"""The replay-age backstop for unadmitted async-completion deliveries (#141893).

An unadmitted durable completion retried on every watcher tick forever: the defer settlement
refunds the delivery attempt, so ``delivery_attempts`` stayed 0 and nothing converged — one
stranded row injected every 2s (~43k log lines/day), with the rows pending for days. The
replay-age cap that would have dropped them only ran at startup replay, so a row inside an
uptime window was never judged. This is the slimmed variant of PR #141959: only the terminal
backstop (the piece the concurrent #141926 backoff-and-durable-cap sweep does not carry) —
a refusal for a row already past the replay age settles it as dropped, the same verdict
startup replay gives its twin.
"""

import asyncio
import queue
import time
from collections import OrderedDict
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

import tools.async_delegation as ad
from gateway.run import GatewayRunner
from gateway.wake import WakeNotAccepted
from tools.process_registry import ProcessRegistry


@pytest.fixture(autouse=True)
def isolated_registry(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    import tools.process_registry as pr_module

    monkeypatch.setattr(pr_module, "CHECKPOINT_PATH", tmp_path / "processes.json")
    monkeypatch.setattr(pr_module, "process_registry", ProcessRegistry())


def _runner(adapter):
    runner = object.__new__(GatewayRunner)
    runner._running = True
    runner.adapters = {__import__("gateway.config", fromlist=["Platform"]).Platform.TELEGRAM: adapter}
    runner.session_store = SimpleNamespace(_ensure_loaded=lambda: None, _entries={})
    runner._session_source_cache = {}
    runner._completion_delivery_lock = __import__("threading").Lock()
    runner._completion_deliveries_inflight = set()
    runner._completion_deliveries_delivered = OrderedDict()
    runner._completion_delivery_retention = 2048
    runner._background_tasks = set()
    return runner


def _async_event(delegation_id="deleg_bounds"):
    now = time.time()
    return {
        "type": "async_delegation",
        "delegation_id": delegation_id,
        "session_key": "agent:main:telegram:dm:12345:678",
        "status": "completed",
        "summary": "Found it",
        "dispatched_at": now - 60.0,
        "completed_at": now - 12.0,
    }


def _persist_pending(event, *, completed_at=None):
    ad._persist_dispatch({
        "delegation_id": event["delegation_id"],
        "session_key": event["session_key"],
        "origin_ui_session_id": "",
        "parent_session_id": event.get("parent_session_id"),
        "dispatched_at": event["dispatched_at"],
    })
    ad._persist_completion(event, {"status": "completed", "summary": event["summary"]})
    if completed_at is not None:
        with ad._DB_LOCK, ad._transaction() as conn:
            conn.execute("UPDATE async_delegations SET completed_at=? WHERE delegation_id=?",
                         (completed_at, event["delegation_id"]))


def _stop_after_sleeps(monkeypatch, runner, count):
    sleep_calls = 0

    async def _bounded_sleep(_delay):
        nonlocal sleep_calls
        sleep_calls += 1
        if sleep_calls >= count:
            runner._running = False

    monkeypatch.setattr(asyncio, "sleep", _bounded_sleep)


def _refusing_admission(monkeypatch):
    """Make every injection raise WakeNotAccepted: the session exists but will not admit."""
    raise_or_admit = {"raise": True}

    async def _refuse(adapter, event):
        if raise_or_admit["raise"]:
            raise WakeNotAccepted("session busy")

    import gateway.wake as wake_module
    monkeypatch.setattr(wake_module, "admit_internal_event", _refuse)
    return raise_or_admit


def test_refusal_within_replay_age_defers_without_spending_attempts(monkeypatch, isolated_registry):
    """A fresh completion refused admission keeps today's contract: pending, budget unspent."""
    event = _async_event()
    _persist_pending(event)
    _refusing_admission(monkeypatch)
    runner = _runner(SimpleNamespace(handle_message=MagicMock()))

    result = asyncio.run(runner._deliver_completion_notification("done", event))

    assert result is False
    row = ad.get_durable_delegation(event["delegation_id"])
    assert (row["delivery_state"], row["delivery_attempts"]) == ("pending", 0)


def test_refusal_past_replay_age_terminally_drops(monkeypatch, isolated_registry):
    """A completion that has been pending longer than the replay-age cap settles as dropped on
    its next refusal — the same verdict startup replay gives it — and leaves the queue."""
    event = _async_event()
    _persist_pending(event, completed_at=time.time() - (ad._MAX_COMPLETION_REPLAY_AGE_S + 3600))
    _refusing_admission(monkeypatch)
    isolated = queue.Queue()
    import tools.process_registry as pr_module
    monkeypatch.setattr(pr_module.process_registry, "completion_queue", isolated)
    isolated.put(event)
    runner = _runner(SimpleNamespace(handle_message=MagicMock()))
    _stop_after_sleeps(monkeypatch, runner, count=3)  # 2 work ticks

    asyncio.run(runner._async_delegation_watcher(interval=0))

    row = ad.get_durable_delegation(event["delegation_id"])
    assert row["delivery_state"] == "dropped"
    assert isolated.empty(), "a terminal drop must not requeue the event"
