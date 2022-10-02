import ctypes
import multiprocessing as mp
import threading
import time
from dataclasses import dataclass, field
from queue import PriorityQueue
from typing import Any, Generator, List, Optional, Sequence, Tuple

import torch
from hivemind import MPFuture, get_logger, use_hivemind_log_handler
from hivemind.moe.server.task_pool import TaskPoolBase

use_hivemind_log_handler("in_root_logger")
logger = get_logger(__file__)


@dataclass(order=True, frozen=True)
class Task:
    priority: float
    time_submitted: float
    future: MPFuture = field(compare=False)
    args: Sequence[torch.Tensor] = field(compare=False)

    @property
    def uid(self) -> int:
        return self.future._uid


class PrioritizedTaskPool(TaskPoolBase):
    """
    Aggregates requests from multiple ConnectionHandler instances, orders them for processing in Runtime, then
    returns results (or exception) to the corresponding ConnectionHandler. Runs a background process.
    A single PrioritizedTaskPool services a specific function (e.g. layer1.forward, layer2.forward or layer1.backward)

    :note: unlike hivemind.moe TaskPool, this pool does *not* combine incoming requests into batches.
      This would require grouping requests of different length.

    :param process_func: function to be applied to every formed batch; called by Runtime
        Note that process_func should accept only positional args (Tensors) and return a flat tuple of Tensors
    :param max_batch_size: process at most this many inputs in a batch (task contains have one or several inputs)
         Measured in the total number of tokens (i.e. batch size * sequence length)

    :param name: pool name, used for logging
    :param min_batch_size: process at least this many inputs in a batch, otherwise wait for more
    :param start: if True, start automatically at the end of __init__
    """

    def __init__(
        self,
        process_func: callable,
        max_batch_size: int,
        name: str,
        min_batch_size=1,
        daemon=True,
        start=False,
    ):
        super().__init__(process_func, daemon=daemon, name=name)
        self.min_batch_size, self.max_batch_size = min_batch_size, max_batch_size

        self.submitted_tasks = mp.SimpleQueue()  # interaction with ConnectionHandlers
        self._ordered_tasks = PriorityQueue()  # interaction with Runtime - only valid inside Runtime

        self._prioritizer_thread = threading.Thread(
            name=self.name + "_prioritizer",
            target=self._prioritize_tasks,
            args=[self.submitted_tasks, self._ordered_tasks],
            daemon=True,
        )
        self._dispatched_tasks = {}
        self.batch_receiver, self.batch_sender = mp.Pipe(duplex=False)
        self._oldest_undispatched_timestamp = mp.Value(ctypes.c_double, 1.0)
        self.priority = float("inf"), float("inf")  # (first task priority, first task timestamp)
        if start:
            self.start()

    @staticmethod
    def _prioritize_tasks(submitted_tasks: mp.SimpleQueue, ordered_tasks: PriorityQueue):
        """Read tasks from incoming queue and put them into a local priority queue"""
        while True:
            task = submitted_tasks.get()
            if task is None:
                logger.debug("Shutting down prioritizer thread")
                break

            ordered_tasks.put(task, block=True)

    def start(self):
        assert not self.is_alive() and not self._prioritizer_thread.is_alive()
        self._prioritizer_thread.start()
        super().start()

    def shutdown(self, timeout: Optional[float] = None):
        self.submitted_tasks.put(None)
        self.terminate()
        self._prioritizer_thread.join(timeout)

    def submit_task(self, *args: torch.Tensor, priority: float = 0.0) -> MPFuture:
        """Add task to this pool's queue, return Future for its output"""
        task = Task(priority, time.monotonic(), MPFuture(), args)
        if self.get_task_size(task) > self.max_batch_size:
            exc = ValueError(f"Task size greater than max_batch_size ({self.max_batch_size}), it can't be processed")
            task.future.set_exception(exc)
        else:
            self.submitted_tasks.put(task)
            self.batch_sender.send(None)  # use this pipe to count the number of unfinished batches
            if (task.priority, task.time_submitted) < self.priority:
                self.priority = (task.priority, task.time_submitted)
        return task.future

    def get_task_size(self, task: Task) -> int:
        """compute task processing complexity; defaults to the total number of tokens"""
        if task.args and task.args[0].ndim >= 2:
            return task.args[0].shape[0] * task.args[0].shape[1]
        return 1

    def load_batch_to_runtime(
        self, timeout: Optional[float] = None, device: Optional[torch.device] = None
    ) -> Tuple[Any, List[torch.Tensor]]:
        """receive next batch of arrays"""
        task = self._ordered_tasks.get(block=True, timeout=timeout)
        batch_inputs = [
            tensor.detach().to(device, non_blocking=True).requires_grad_(tensor.requires_grad) for tensor in task.args
        ]
        self._dispatched_tasks[task.uid] = task
        self.batch_receiver.recv()  # reduce the number of active batches
        if not self._ordered_tasks.empty():
            first_remaining_task: Task = self._ordered_tasks.queue[0]
            self.priority = (first_remaining_task.priority, first_remaining_task.time_submitted)
        return task.uid, batch_inputs

    def send_outputs_from_runtime(self, uid: int, batch_outputs: List[torch.Tensor]):
        """send results for a processed batch, previously loaded through load_batch_to_runtime"""
        batch_outputs = [
            tensor.to(device="cpu").share_memory_().detach().requires_grad_(tensor.requires_grad)
            for tensor in batch_outputs
        ]

        task = self._dispatched_tasks.pop(uid, None)
        if task is None:
            logger.error(
                f"Internal error: task task with index {uid} is missing from the dictionary; " f"Could not set result"
            )
        else:
            task.future.set_result(batch_outputs)

    def send_exception_from_runtime(self, uid: int, exception: BaseException):
        task = self._dispatched_tasks.pop(uid, None)
        if task is None:
            logger.error(
                f"Internal error: task task with index {uid} is missing from the dictionary; "
                f"Could not set exception {exception}"
            )
        else:
            task.future.set_exception(exception)

    def run(self, *args, **kwargs):
        mp.Event().wait()

    @property
    def empty(self):
        return not self.batch_receiver.poll()

    @property
    def priority(self) -> Tuple[float, float]:
        """The priority of this pool equals the (priority, timestamp) of the most important task in it."""
        return float(self._priority.value), float(self._oldest_undispatched_timestamp.value)

    @priority.setter
    def priority(self, item: Tuple[float, float]):
        assert len(item) == 2
        self._priority.value = float(item[0])
        self._oldest_undispatched_timestamp.value = float(item[1])
