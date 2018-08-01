import multiprocessing
from abc import ABCMeta, abstractmethod
from datetime import datetime
from typing import MutableSequence, MutableMapping, Tuple
from asyncio import sleep
from .option_result import OptionResult


class Worker(multiprocessing.Process, metaclass=ABCMeta):
    @property
    @abstractmethod
    def done(self) -> bool:
        pass

    @abstractmethod
    def reincarnate(self) -> 'Worker':
        pass


class MessageBus:
    __slots__ = '_manager', '_queued', '_started', '_result', '_2nd_moment_work_time', '_mean_work_time'

    def __init__(self):
        self._manager = multiprocessing.Manager()
        self._queued: MutableSequence[OptionResult] = self._manager.list()
        self._started: MutableMapping[OptionResult, Tuple[datetime, int]] = self._manager.dict()
        self._result: MutableSequence[Tuple[OptionResult, float]] = self._manager.list()

        self._mean_work_time = self._manager.Value('f', 100)
        self._2nd_moment_work_time = self._manager.Value('f', 10)

    def queue(self, opt_res: OptionResult):
        self._queued.append(opt_res)

    def started(self, opt_res: OptionResult, worker: Worker):
        self._started[opt_res] = datetime.now(), worker.pid

    def add_result(self, opt_res: OptionResult, f: float):
        now = datetime.now()

        start_time, pid = self._started.pop(opt_res)
        duration = (now - start_time).total_seconds()
        with self._mean_work_time.get_lock():
            self._mean_work_time.value = .99 * self._mean_work_time.value + .01 * duration
        with self._2nd_moment_work_time.get_lock():
            self._2nd_moment_work_time.value = .99 * self._2nd_moment_work_time.value + .01 * duration ** 2

        self._result.append((opt_res, f))


    async def kill_zombie(self, *workers: Worker):
        workers = list(workers)

        while len(workers) > 0:
            for w in tuple(workers):
                if w.done:
                    workers.remove(w)

            now = datetime.now()
            mean = self._mean_work_time.value
            threshold = mean + 2 * (self._2nd_moment_work_time.value - mean**2) ** .5
            pids = tuple(w.pid for w in workers)
            for opt_res, (dt, pid) in tuple(self._started.items()):
                if pid not in pids:
                    self._started.pop(opt_res)
                    continue

                duration = (now - dt).total_seconds()
                if threshold < duration:
                    worker = workers[pids.index(pid)]
                    worker.terminate()
                    new = worker.reincarnate()
                    new.start()
                    workers[workers.index(worker)] = new

            sleep(mean / 2)






