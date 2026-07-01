"""
cv_agent.gpu_queue
==================
Single-threaded GPU worker queue for serialising all CUDA inference calls.

HuggingFace pipelines and CUDA are NOT thread-safe — this queue ensures
GPU calls are sequential while letting callers submit work asynchronously.
"""

from __future__ import annotations

import os
import queue
import threading
from typing import Any, Callable, Dict, Optional

from cv_agent.config import logger


# ==============================================================================
# GPU QUEUE
# ==============================================================================

class GPUQueue:
    """
    Serialise all GPU inference calls through a single worker thread.

    FIX C-1 (v7): self-healing watchdog with timeout and restart.
    """

    SUBMIT_TIMEOUT_S: float = float(os.getenv("GPU_SUBMIT_TIMEOUT_S", "300"))

    def __init__(self) -> None:
        self._q:             queue.Queue   = queue.Queue()
        self._thread:        Optional[threading.Thread] = None
        self._lock:          threading.Lock = threading.Lock()
        self._restart_count: int            = 0
        self._stop_event:    threading.Event = threading.Event()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _make_thread(self) -> threading.Thread:
        return threading.Thread(
            target=self._worker, daemon=True,
            name=f"gpu-worker-{self._restart_count}",
        )

    def _ensure_worker(self) -> None:
        """Start or restart the worker thread if it is not alive."""
        with self._lock:
            if self._thread is None or not self._thread.is_alive():
                if self._thread is not None:
                    self._restart_count += 1
                    logger.warning(
                        "GPUQueue: worker thread dead — restarting (restart #%d)",
                        self._restart_count,
                    )
                self._thread = self._make_thread()
                self._thread.start()

    def start(self) -> None:
        """Explicit start (called by run_pipeline; also called lazily by submit)."""
        self._ensure_worker()

    def _worker(self) -> None:
        """
        GPU worker loop.

        Uses get(timeout=1.0) with heartbeat so it never hangs on empty queue.
        Every item's event is always set in the finally block.
        """
        logger.info("GPUQueue worker started (thread=%s)", threading.current_thread().name)
        while not self._stop_event.is_set():
            try:
                fn, args, kwargs, result_holder, event = self._q.get(timeout=1.0)
            except queue.Empty:
                continue  # heartbeat — keep thread alive
            except Exception as exc:
                logger.error("GPUQueue: unexpected error reading queue: %s", exc)
                continue

            try:
                result_holder["result"] = fn(*args, **kwargs)
            except BaseException as exc:          # catch CUDA OOM / SystemExit too
                result_holder["error"] = exc
                logger.error("GPUQueue: inference call failed: %s", exc)
            finally:
                event.set()                       # ALWAYS unblock submit() callers
                self._q.task_done()

        logger.info("GPUQueue worker stopped (thread=%s)", threading.current_thread().name)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def submit(self, fn: Callable, *args: Any, **kwargs: Any) -> Any:
        """
        Submit a callable to the GPU queue and block until complete.

        Raises RuntimeError if the worker does not respond within
        SUBMIT_TIMEOUT_S seconds (default 300 s / 5 min).
        """
        self._ensure_worker()
        holder: Dict[str, Any] = {}
        done   = threading.Event()
        self._q.put((fn, args, kwargs, holder, done))

        finished = done.wait(timeout=self.SUBMIT_TIMEOUT_S)
        if not finished:
            logger.error(
                "GPUQueue: submit timed out after %.0fs — forcing worker restart",
                self.SUBMIT_TIMEOUT_S,
            )
            with self._lock:
                self._thread = None   # mark dead so _ensure_worker restarts it
            raise RuntimeError(
                f"GPU worker did not respond within {self.SUBMIT_TIMEOUT_S}s. "
                "It has been scheduled for restart. Please retry your request."
            )

        if "error" in holder:
            raise holder["error"]
        return holder["result"]

    @property
    def is_alive(self) -> bool:
        """True if the worker thread is running. Used by /health endpoint."""
        return self._thread is not None and self._thread.is_alive()

    def stop(self) -> None:
        """Gracefully stop the worker (useful for clean shutdown)."""
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=5.0)


# Module-level singleton
_gpu_queue = GPUQueue()
