# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

"""Characterization oracle for the public ``monarch`` ``Future`` state machine.

Pins the behavior of ``monarch._src.actor.future.Future`` -- its internal
states and the transitions between them, ``as_asyncio()`` and the ``__await__``
shim over it, the residual ``_Tokio`` cross-world guards, the
``get()``-inside-a-loop warning, and the ``_take_inner()`` accessor with its
``_Taken`` terminal state -- so that any change to that behavior is caught here
and made explicit.
"""

import asyncio
import warnings

import pytest
from monarch._rust_bindings.monarch_hyperactor.pytokio import (
    Handle,
    is_tokio_thread,
    PythonTask,
    WouldBlockRuntime,
)
from monarch._src.actor import future as future_mod
from monarch._src.actor.future import Future


async def _value(v):
    return v


async def _raise(exc: BaseException):
    raise exc


async def _sleep_then(seconds: float, v):
    await PythonTask.sleep(seconds)
    return v


async def _await_once(fut: Future):
    return await fut


def _run_in_tokio(coro):
    # Drive ``coro`` to completion on the Tokio runtime. Code inside ``coro``
    # runs on a Tokio worker thread, where ``is_tokio_thread()`` is True and a
    # ``Future`` takes its ``__await__`` tokio branch. ``block_on`` itself runs
    # on the (non-worker) calling thread, so it is allowed to block.
    return PythonTask.from_coroutine(coro).block_on()


# ---------------------------------------------------------------------------
# States and transitions
#
# A fresh Future is _Unawaited; the first access drives the underlying task
# exactly once. Sync get() caches into _Complete/_Exception; as_asyncio() and
# the asyncio __await__ shim spawn a Handle and move to the observe-only
# _Handle; a tokio-thread await still spawns a Shared and moves to _Tokio.
# These pin that lifecycle and its idempotency.
# ---------------------------------------------------------------------------


def test_get_success_transitions_to_complete_and_is_idempotent():
    """First sync get() drives the task, transitions _Unawaited -> _Complete,
    and returns the value; a second get() returns the cached value without
    re-running the coroutine. The coroutine must run exactly once, so
    resolve-and-cache is the contract being pinned."""
    fut: Future[int] = Future(coro=_value(42))
    assert fut.get() == 42
    assert fut.get() == 42


def test_get_exception_transitions_to_exception_and_reraises_stored():
    """Failure mirror of the success terminal: the first get() catches the
    raised exception, transitions _Unawaited -> _Exception (caching the
    object), and re-raises; a second get() re-raises the *same stored* object
    without re-running. Identity is asserted between the two gets (not against
    the constructed ValueError) because the exception crosses the Rust/pyo3
    boundary -- the contract is that _Exception caches one object and re-raises
    it."""
    fut: Future[int] = Future(coro=_raise(ValueError("boom")))
    with pytest.raises(ValueError, match="boom") as first:
        fut.get()
    with pytest.raises(ValueError, match="boom") as second:
        fut.get()
    assert second.value is first.value


def test_get_timeout_raises_timeout_error():
    """get(timeout=...) observes the task through a Handle with the given
    deadline, so an unfinished task surfaces a TimeoutError rather than hanging.
    The timeout is non-cancelling (state becomes _Handle); the sibling
    test_get_timeout_is_non_poisoning_and_reobservable pins re-observation."""
    fut: Future[None] = Future(coro=PythonTask.sleep(3600))
    with pytest.raises(TimeoutError):
        fut.get(timeout=0.1)


def test_get_timeout_is_non_poisoning_and_reobservable():
    """get(timeout=...) that times out routes through a Handle and does NOT
    poison the Future: it transitions to _Handle, and a later get() (no timeout)
    still resolves the same task (the timeout is non-cancelling)."""
    fut: Future[int] = Future(coro=_sleep_then(0.2, 11))
    with pytest.raises(TimeoutError):
        fut.get(timeout=0.01)  # too short: task still running
    assert isinstance(fut._status, future_mod._Handle)
    assert fut.get() == 11  # later get() still observes completion


def test_get_timeout_success_returns_and_transitions_to_handle():
    """get(timeout=...) whose task finishes within the timeout returns the value
    and transitions to the observe-only _Handle (not _Complete): the timed get
    routes through a Handle, so a later no-timeout get() re-observes the result."""
    fut: Future[int] = Future(coro=_sleep_then(0.01, 7))
    assert fut.get(timeout=5) == 7
    assert isinstance(fut._status, future_mod._Handle)
    assert fut.get() == 7


async def test_await_asyncio_transitions_to_handle_and_is_reobservable():
    """Awaiting under an asyncio loop bridges through as_asyncio(), transitions
    _Unawaited -> the observe-only _Handle, and yields the value; re-awaiting
    observes the same Handle (a fresh loop-local future each time)."""
    fut: Future[int] = Future(coro=_value(7))
    assert await fut == 7  # _Unawaited -> _Handle
    assert isinstance(fut._status, future_mod._Handle)
    assert await fut == 7  # re-observe the same Handle


def test_await_tokio_transitions_to_tokio_and_is_idempotent():
    """Awaiting on a tokio thread (inside from_coroutine, where
    is_tokio_thread() is True) transitions _Unawaited -> _Tokio (spawning a
    Shared) and yields the value; re-awaiting reuses it. The _Handle-vs-_Tokio
    split is by which async world the await runs in (asyncio loop vs tokio
    thread)."""
    fut: Future[int] = Future(coro=_value(9))

    async def driver():
        assert is_tokio_thread()
        first = await fut  # _Unawaited -> _Tokio
        second = await fut  # idempotent re-await of the _Tokio state
        return first, second

    assert _run_in_tokio(driver()) == (9, 9)


# ---------------------------------------------------------------------------
# Observation is non-consuming now: get()/as_asyncio()/await mix freely on the
# asyncio+sync side (get after await, await after get, repeated as_asyncio all
# resolve). The only remaining "already converted" guards are the cross-world
# ones involving the residual _Tokio state (and _Handle-awaited-on-a-tokio-
# thread), which remain while the pytokio-driving from_coroutine path exists.
# These pin that: the formerly-erroring same-world cases now resolve; the
# cross-world ones still raise.
# ---------------------------------------------------------------------------


def test_get_after_asyncio_await_returns():
    """get() after an asyncio await (_Handle) now returns the value instead of
    raising -- observation is non-consuming, so the sync get() observes the same
    Handle."""
    fut: Future[int] = Future(coro=_value(1))
    asyncio.run(_await_once(fut))  # -> _Handle
    assert fut.get() == 1


def test_get_after_asyncio_await_error_reraises():
    """Failure mirror of the previous: get() on a _Handle whose live producer
    FAILED re-raises the stored exception through handle.get(). Drives the live
    spawn_handle producer (not a pre-resolved get())."""
    fut: Future[int] = Future(coro=_raise(ValueError("boom")))

    async def bridge():
        # Drive _Unawaited -> _Handle via the live producer; swallow the failure
        # so we can re-observe it via get() below.
        with pytest.raises(ValueError, match="boom"):
            await fut

    asyncio.run(bridge())
    assert isinstance(fut._status, future_mod._Handle)
    with pytest.raises(ValueError, match="boom"):
        fut.get()


def test_get_after_tokio_conversion_raises():
    """get() after a tokio await (_Tokio): once converted to a pytokio Shared,
    get() refuses and points at awaiting inside a from_coroutine coroutine."""
    fut: Future[int] = Future(coro=_value(1))
    _run_in_tokio(_await_once(fut))  # -> _Tokio
    with pytest.raises(
        ValueError, match=r"already converted into a pytokio\.Shared object"
    ):
        fut.get()


def test_await_asyncio_on_tokio_converted_raises():
    """Cross-world: convert on tokio (_Tokio), then await under asyncio -- the
    asyncio path (as_asyncio) refuses a tokio-converted Future."""
    fut: Future[int] = Future(coro=_value(1))
    _run_in_tokio(_await_once(fut))  # -> _Tokio

    async def attempt():
        with pytest.raises(
            ValueError, match=r"already converted into a pytokio\.Shared object"
        ):
            await fut

    asyncio.run(attempt())


def test_await_asyncio_after_get_returns():
    """await under asyncio after a sync get() resolved it (_Complete) now returns
    the value -- as_asyncio() hands back a settled loop future instead of
    raising."""
    fut: Future[int] = Future(coro=_value(1))
    assert fut.get() == 1  # -> _Complete

    async def attempt():
        return await fut

    assert asyncio.run(attempt()) == 1


def test_await_asyncio_after_get_exception_reraises_stored():
    """Failure mirror of the previous: await under asyncio after a failed sync
    get() (_Exception) hands back a settled failed loop future that re-raises the
    *same* stored exception object."""
    boom = ValueError("boom")
    fut: Future[int] = Future(coro=_raise(boom))
    with pytest.raises(ValueError, match="boom"):
        fut.get()  # -> _Exception

    async def attempt():
        with pytest.raises(ValueError, match="boom") as caught:
            await fut
        return caught.value

    assert asyncio.run(attempt()) is boom


def test_await_asyncio_live_handle_error_propagates():
    """A failing coroutine driven LIVE through the Handle producer (spawn_handle
    via as_asyncio, from _Unawaited -- not pre-resolved) surfaces the exception
    through the bridged asyncio future: exercises send_result(Err) -> observer ->
    set_exception, which the pre-resolved settled-future path does not."""
    fut: Future[int] = Future(coro=_raise(ValueError("boom")))

    async def attempt():
        with pytest.raises(ValueError, match="boom"):
            await fut  # _Unawaited -> live _Handle, error via the observer

    asyncio.run(attempt())


def test_await_tokio_on_handle_bridged_raises():
    """Convert on asyncio (_Handle), then await on a tokio thread: the tokio
    branch intentionally refuses a Future already bridged to asyncio."""
    fut: Future[int] = Future(coro=_value(1))
    asyncio.run(_await_once(fut))  # -> _Handle

    async def attempt():
        await fut

    with pytest.raises(ValueError, match="not awaitable on a tokio thread"):
        _run_in_tokio(attempt())


def test_await_tokio_after_get_raises_synchronous_future():
    """await on a tokio thread after a sync get() resolved it (_Complete): the
    tokio branch refuses with the 'already converted into a synchronous future'
    guard."""
    fut: Future[int] = Future(coro=_value(1))
    assert fut.get() == 1  # -> _Complete

    async def attempt():
        await fut

    with pytest.raises(ValueError, match="already converted into a synchronous future"):
        _run_in_tokio(attempt())


def test_await_tokio_after_get_exception_raises_synchronous_future():
    """Failure mirror of test_await_tokio_after_get_raises_synchronous_future:
    await on a tokio thread after a failed sync get() (_Exception) hits the same
    'synchronous future' guard as _Complete."""
    fut: Future[int] = Future(coro=_raise(ValueError("boom")))
    with pytest.raises(ValueError, match="boom"):
        fut.get()  # -> _Exception

    async def attempt():
        await fut

    with pytest.raises(ValueError, match="already converted into a synchronous future"):
        _run_in_tokio(attempt())


def test_await_with_no_event_loop_raises():
    """await with neither an asyncio loop nor a tokio runtime active: there is no
    driver, so __await__ refuses outright."""
    fut: Future[int] = Future(coro=_value(1))
    with pytest.raises(ValueError, match="no active event loop"):
        fut.__await__()


# ---------------------------------------------------------------------------
# get()-inside-a-loop warning -- both branches
#
# The warning fires for asyncio OR tokio callers and forwards a tracing event
# with extra["context"] = "asyncio"/"tokio". We assert the observable seam by
# monkeypatching the module-level log_with_tracing rather than reading Rust
# tracing output.
# ---------------------------------------------------------------------------


def test_get_in_asyncio_loop_warns_and_still_returns(monkeypatch):
    """get() inside an asyncio loop (main thread) warns and forwards a tracing
    event with context='asyncio', but -- because the block is legal on the main
    thread -- still returns the value (backward-compatible)."""
    calls = []
    monkeypatch.setattr(future_mod, "log_with_tracing", lambda *a, **k: calls.append(k))

    async def runner():
        fut: Future[int] = Future(coro=_value(5))
        with pytest.warns(UserWarning, match="active event loop") as record:
            value = fut.get()
        # the asyncio-context advice is as_asyncio()/await (both valid on a loop)
        assert any("as_asyncio" in str(w.message) for w in record)
        return value

    assert asyncio.run(runner()) == 5
    assert len(calls) == 1
    assert calls[0]["extra"]["context"] == "asyncio"


def test_get_in_tokio_thread_raises_would_block_and_is_non_consuming(monkeypatch):
    """On a real tokio thread no-timeout get() raises WouldBlockRuntime up front
    (aligned with Handle.get()) WITHOUT consuming the task: no UserWarning fires,
    the tracing event still forwards (context='tokio'), the Future stays
    _Unawaited, and a later get() from a sync context still drives it to a
    value. (Previously it warned, then block_on() consumed the task and panicked
    'from within a runtime', losing the work and bricking the Future.)"""
    calls = []
    warned = []
    monkeypatch.setattr(future_mod, "log_with_tracing", lambda *a, **k: calls.append(k))
    monkeypatch.setattr(
        future_mod.warnings, "warn", lambda msg, *a, **k: warned.append(str(msg))
    )
    fut: Future[int] = Future(coro=_value(5))

    async def attempt():
        assert is_tokio_thread()
        return fut.get()

    with pytest.raises(WouldBlockRuntime):
        _run_in_tokio(attempt())
    assert len(calls) == 1
    assert calls[0]["extra"]["context"] == "tokio"
    assert warned == []  # tokio get() refuses instead of warning
    assert isinstance(fut._status, future_mod._Unawaited)  # task not consumed
    assert fut.get() == 5  # still drivable from a sync context


def test_get_timeout_in_tokio_thread_raises_would_block_and_is_non_consuming(
    monkeypatch,
):
    """get(timeout=...) on a tokio thread is refused by the in_tokio check up
    front -- BEFORE spawning a Handle -- so it raises WouldBlockRuntime without
    starting work or flipping state: no UserWarning, the trace still forwards
    (context='tokio'), the Future stays _Unawaited, and a later sync get() still
    drives it."""
    calls = []
    warned = []
    monkeypatch.setattr(future_mod, "log_with_tracing", lambda *a, **k: calls.append(k))
    monkeypatch.setattr(
        future_mod.warnings, "warn", lambda msg, *a, **k: warned.append(str(msg))
    )
    fut: Future[int] = Future(coro=_value(5))

    async def attempt():
        assert is_tokio_thread()
        return fut.get(timeout=0.1)

    with pytest.raises(WouldBlockRuntime):
        _run_in_tokio(attempt())
    assert len(calls) == 1
    assert calls[0]["extra"]["context"] == "tokio"
    assert warned == []
    assert isinstance(fut._status, future_mod._Unawaited)  # not spawned/consumed
    assert fut.get() == 5  # still drivable from a sync context


def test_get_invalid_timeout_does_not_spawn_or_mutate():
    """An invalid timeout (NaN/negative/non-finite) is rejected with ValueError
    BEFORE spawning a Handle, so a bad argument never starts work or flips state:
    the Future stays _Unawaited and a later valid get() still drives it."""
    for bad in (float("nan"), -1.0, float("inf")):
        fut: Future[int] = Future(coro=_value(8))
        with pytest.raises(ValueError, match="invalid timeout"):
            fut.get(timeout=bad)
        assert isinstance(fut._status, future_mod._Unawaited)
        assert fut.get() == 8  # still drivable after a rejected bad-timeout call


def test_get_on_handle_in_asyncio_loop_warns_and_forwards_once(monkeypatch):
    """get() on an already-bridged Future (_Handle) from inside an asyncio loop
    forwards exactly one tracing event (context='asyncio') and warns exactly
    once: Future.get() forwards the trace, Handle.get() emits the UserWarning,
    and the _Handle branch does not re-warn."""
    calls = []
    monkeypatch.setattr(future_mod, "log_with_tracing", lambda *a, **k: calls.append(k))

    async def runner():
        fut: Future[int] = Future(coro=_value(5))
        fut.as_asyncio()  # -> _Handle
        with pytest.warns(UserWarning) as caught:
            value = fut.get()
        user_warnings = [w for w in caught if issubclass(w.category, UserWarning)]
        return value, len(user_warnings)

    value, n_user_warnings = asyncio.run(runner())
    assert value == 5
    assert n_user_warnings == 1
    assert len(calls) == 1
    assert calls[0]["extra"]["context"] == "asyncio"


def test_get_in_loop_on_cached_state_traces_without_warning(monkeypatch):
    """get() inside an asyncio loop on an already-cached Future (_Complete or
    _Exception) forwards exactly one tracing event per call and emits NO
    UserWarning: the warning is reserved for paths that actually block the loop,
    and a cached read does not."""
    calls = []
    monkeypatch.setattr(future_mod, "log_with_tracing", lambda *a, **k: calls.append(k))

    done: Future[int] = Future(coro=_value(5))
    assert done.get() == 5  # off loop -> _Complete
    failed: Future[int] = Future(coro=_raise(ValueError("boom")))
    with pytest.raises(ValueError, match="boom"):
        failed.get()  # off loop -> _Exception
    calls.clear()

    async def runner():
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            assert done.get() == 5  # cached _Complete, inside the loop
            with pytest.raises(ValueError, match="boom"):
                failed.get()  # cached _Exception, inside the loop
        return [w for w in caught if issubclass(w.category, UserWarning)]

    assert asyncio.run(runner()) == []
    assert len(calls) == 2  # one tracing forward per in-loop get()
    assert all(c["extra"]["context"] == "asyncio" for c in calls)


def test_handle_and_would_block_runtime_are_importable():
    """``Handle`` and ``WouldBlockRuntime`` import from the pytokio bindings, and
    ``WouldBlockRuntime`` subclasses ``RuntimeError``."""
    assert issubclass(WouldBlockRuntime, RuntimeError)
    assert isinstance(Handle, type)


# ---------------------------------------------------------------------------
# _take_inner() and the _Taken terminal state
#
# _take_inner() requires an unawaited Future -- it fails if the Future was
# already resolved or converted by get()/await. On success it surrenders the
# underlying PythonTask to a caller that drives it directly and transitions the
# Future to the terminal _Taken state, so the Future is spent: a second take,
# or any later get()/await, fails rather than silently re-driving the one-shot
# task.
# ---------------------------------------------------------------------------


def test_take_inner_returns_task_and_marks_future_taken():
    """_take_inner() on an unawaited Future returns the underlying (still
    drivable) PythonTask and transitions the Future to the terminal _Taken
    state."""
    fut: Future[int] = Future(coro=_value(1))
    task = fut._take_inner()
    assert isinstance(task, PythonTask)
    assert isinstance(fut._status, future_mod._Taken)
    assert task.block_on() == 1


def test_take_inner_twice_raises():
    """The Future is spent after the first _take_inner(); a second raises."""
    fut: Future[int] = Future(coro=_value(1))
    fut._take_inner()
    with pytest.raises(ValueError, match="already been awaited"):
        fut._take_inner()


def test_get_after_take_inner_raises():
    """get() after _take_inner() fails at the Future instead of re-driving the
    surrendered task."""
    fut: Future[int] = Future(coro=_value(1))
    fut._take_inner()
    with pytest.raises(ValueError, match="consumed"):
        fut.get()


async def test_await_asyncio_after_take_inner_raises():
    """await under asyncio after _take_inner() fails instead of re-driving."""
    fut: Future[int] = Future(coro=_value(1))
    fut._take_inner()
    with pytest.raises(ValueError, match="consumed"):
        await fut


def test_await_tokio_after_take_inner_raises():
    """await on a tokio thread after _take_inner() fails instead of re-driving."""
    fut: Future[int] = Future(coro=_value(1))
    fut._take_inner()

    async def attempt():
        await fut

    with pytest.raises(ValueError, match="consumed"):
        _run_in_tokio(attempt())


def test_take_inner_after_get_raises():
    """_take_inner() requires an unawaited Future: once get() has resolved it
    (_Complete), taking the inner task is refused."""
    fut: Future[int] = Future(coro=_value(1))
    assert fut.get() == 1  # -> _Complete
    with pytest.raises(ValueError, match="already been awaited"):
        fut._take_inner()


def test_take_inner_after_asyncio_await_raises():
    """Once an asyncio await has bridged the Future (_Handle), _take_inner() is
    refused."""
    fut: Future[int] = Future(coro=_value(1))
    asyncio.run(_await_once(fut))  # -> _Handle
    with pytest.raises(ValueError, match="already been awaited"):
        fut._take_inner()


def test_take_inner_after_tokio_await_raises():
    """Once a tokio await has converted the Future (_Tokio), _take_inner() is
    refused."""
    fut: Future[int] = Future(coro=_value(1))
    _run_in_tokio(_await_once(fut))  # -> _Tokio
    with pytest.raises(ValueError, match="already been awaited"):
        fut._take_inner()


# ---------------------------------------------------------------------------
# as_asyncio() observation semantics
#
# as_asyncio() spawns a Handle once and observes it: observation is
# non-consuming, each call returns a fresh loop-local future, it requires a
# running loop, and it must not consume the task when there is none.
# ---------------------------------------------------------------------------


def test_as_asyncio_off_loop_raises_and_is_non_consuming():
    """Off a running loop, as_asyncio() raises RuntimeError WITHOUT spawning: the
    Future stays _Unawaited, so a later get() still drives the task."""
    fut: Future[int] = Future(coro=_value(3))
    with pytest.raises(RuntimeError):
        fut.as_asyncio()
    assert isinstance(fut._status, future_mod._Unawaited)
    assert fut.get() == 3


async def test_as_asyncio_twice_same_loop_both_resolve():
    """Two as_asyncio() futures from one Future on the same loop both resolve to
    the value (ordinary multi-observer case); the Future is _Handle after the
    first."""
    fut: Future[int] = Future(coro=_value(4))
    f1 = fut.as_asyncio()
    assert isinstance(fut._status, future_mod._Handle)
    f2 = fut.as_asyncio()
    assert await f1 == 4
    assert await f2 == 4


async def test_as_asyncio_cancel_one_then_await_again_resolves():
    """Cancelling one as_asyncio() observer does not poison the Future: a later
    await still resolves (each observer is a fresh loop-local future)."""
    fut: Future[int] = Future(coro=_value(5))
    f1 = fut.as_asyncio()
    f1.cancel()
    assert await fut == 5


def test_as_asyncio_across_two_loops_both_resolve():
    """The same Future observed from two different event loops resolves in each
    (each as_asyncio() binds a fresh future to the current loop)."""
    fut: Future[int] = Future(coro=_value(6))

    async def observe():
        return await fut

    assert asyncio.run(observe()) == 6
    assert asyncio.run(observe()) == 6


async def test_as_asyncio_on_cached_stop_iteration_wraps_in_runtime_error():
    """as_asyncio() on a cached _Exception must wrap a StopIteration in
    RuntimeError before set_exception: asyncio.Future.set_exception rejects
    StopIteration (PEP 479) with TypeError, which would otherwise brick the
    returned future. Mirrors the Handle observer path (HDL-7). A StopIteration
    can't reach _Exception via from_coroutine (it becomes a normal return), so
    the cached state is set directly."""
    fut: Future[int] = Future(coro=_value(1))
    fut._status = future_mod._Exception(StopIteration("done"))
    settled = fut.as_asyncio()
    with pytest.raises(RuntimeError, match="StopIteration"):
        await settled
