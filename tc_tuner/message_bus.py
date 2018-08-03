import multiprocessing
from abc import ABCMeta, abstractmethod
from asyncio import sleep
from datetime import datetime
from typing import MutableSequence, MutableMapping, Tuple, TypeVar, List
from .stats import TunerStats
from .option_result import OptionResult

T = TypeVar('T')


class Worker(multiprocessing.Process, metaclass=ABCMeta):
    @property
    @abstractmethod
    def done(self) -> bool:
        pass

    @abstractmethod
    def reincarnate(self: T) -> T:
        pass


class MessageBus:
    __slots__ = '_manager', '_queued', '_started', '_result', '_2nd_moment_work_time', '_mean_work_time', '_stats'

    _MEAN = 100

    def __init__(self, name: str):
        self._manager = multiprocessing.Manager()
        self._queued: MutableSequence[OptionResult] = self._manager.list()
        self._started: MutableMapping[OptionResult, Tuple[datetime, int]] = self._manager.dict()
        self._result: MutableSequence[OptionResult] = self._manager.list()
        self._mean_work_time = self._manager.Value('f', 100)
        self._2nd_moment_work_time = self._manager.Value('f', 100 ** 2 + 10 ** 2)

        self._stats = TunerStats(name=name)

    def queue(self, opt_res: OptionResult):
        self._queued.append(opt_res)
        self._stats.queue_passage('queued')

    def started(self, opt_res: OptionResult, worker: Worker):
        self._started[opt_res] = datetime.now(), worker.pid
        self._stats.queue_passage('started')

    def add_result(self, opt_res: OptionResult):
        now = datetime.now()

        start_time, pid = self._started.pop(opt_res)
        duration = (now - start_time).total_seconds()
        with self._mean_work_time.get_lock():
            self._mean_work_time.value = .99 * self._mean_work_time.value + .01 * duration
        with self._2nd_moment_work_time.get_lock():
            self._2nd_moment_work_time.value = .99 * self._2nd_moment_work_time.value + .01 * duration ** 2

        self._result.append(opt_res)
        self._stats.queue_passage('result')

    def pop_result(self, idx: int = 0):
        return self._result.pop(idx)

    def result_len(self):
        return len(self._result)

    async def monitor(self):
        while True:
            self._stats.queue_size('queued', len(self._queued))
            self._stats.queue_size('started', len(self._started))
            self._stats.queue_size('result', len(self._result))

            await sleep(.2)

    async def kill_zombie(self, *workers: Worker):
        workers: List[Worker] = list(workers)

        while len(workers) > 0:
            for w in tuple(workers):
                if w.done:
                    workers.remove(w)

            now = datetime.now()
            mean = self._mean_work_time.value
            threshold = mean + 2 * (self._2nd_moment_work_time.value - mean ** 2) ** .5
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

            await sleep(mean / 2)
