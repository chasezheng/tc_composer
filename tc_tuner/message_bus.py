import asyncio
import multiprocessing
import time
from abc import ABCMeta, abstractmethod
from asyncio import sleep
from multiprocessing import sharedctypes
from typing import MutableSequence, Tuple, TypeVar, List

from .gpu_lock import gpu_lock
from .option_result import OptionResult
from .settings import get_configured_logger
from .stats import TunerStats
from . import async_util
T = TypeVar('T')


class Worker(multiprocessing.Process, metaclass=ABCMeta):
    @property
    @abstractmethod
    def done(self) -> bool:
        pass

    @property
    @abstractmethod
    def started_at(self) -> float:
        pass

    @abstractmethod
    def reincarnate(self: T) -> T:
        pass


class MessageBus:
    __slots__ = '_queued', '_result', '_compile_time', '_stats', \
                'aget_queue_config'

    _MEAN = 100

    def __init__(self, name: str):
        manager = multiprocessing.Manager()
        self._queued: MutableSequence[OptionResult] = manager.list()
        self._result: MutableSequence[OptionResult] = manager.list()
        self._compile_time: Tuple[float, float] = multiprocessing.sharedctypes.RawArray('f', 2)
        self._compile_time[0], self._compile_time[1] = 100, 10 ** 2 + 100 ** 2

        self._stats = TunerStats(name=name)

        self.aget_queue_config = [.1, 1.005, 3]

    @property
    def logger(self):
        return get_configured_logger(__name__)

    def queue(self, opt_res: OptionResult):
        self._queued.append(opt_res)
        self._stats.queue_passage('queued')

    def queue_msg_num(self):
        return len(self._queued)

    def pop_queue(self, idx: int = 0):
        return self._queued.pop(idx)

    async def aget_queue(self, idx: int = 0):
        delay, backoff, recovery_bias = self.aget_queue_config

        while True:
            try:
                while True:
                    if len(self._queued) > 0:
                        return self._queued.pop(idx)
                    else:
                        await asyncio.sleep(delay)
                        delay *= backoff
            except IndexError:
                pass
            finally:
                self.aget_queue_config[0] = delay / (backoff ** recovery_bias)

    def add_result(self, opt_res: OptionResult):
        if opt_res.compile_time not in (float('inf'), float('nan')):
            mean, _2nd_moment = self._compile_time
            mean = .98 * mean + .02 * opt_res.compile_time  # todo test
            _2nd_moment = .98 * _2nd_moment + .02 * opt_res.compile_time ** 2
            self._compile_time[:2] = mean, _2nd_moment

        self._result.append(opt_res)
        self.logger.info(f'Option speed: {opt_res.speed}. Compile time: {opt_res.compile_time}')
        self._stats.queue_passage('result')

    def pop_result(self, idx: int = 0):
        self.logger.info(f'Popping result')
        return self._result.pop(idx)

    def result_len(self):
        return len(self._result)

    @async_util.repeat(recovery_bias=16)
    async def find_and_pop(self, opt_res: OptionResult) -> OptionResult:
        opt_str = str(opt_res.option)

        with self._result._mutex:  # todo test
            strs = tuple(str(r.option) for r in tuple(self._result))
            if opt_str in strs:
                return self._result.pop(strs.index(opt_str))
            else:
                return async_util.Repeat.DelayFactor(1)

    async def monitor(self):
        while True:
            self._stats.queue_size('queued', len(self._queued))
            self._stats.queue_size('result', len(self._result))

            await sleep(.2)

    async def kill_zombie(self, *workers: Worker):
        workers: List[Worker] = list(workers)

        while len(workers) > 0:
            for w in tuple(workers):
                if w.done:
                    workers.remove(w)

            mean, _2nd_moment = self._compile_time
            sd = (_2nd_moment - mean ** 2) ** .5
            threshold = max(mean + 10 * sd, 100)
            self.logger.info(f'Mean compile time: {mean}. sd: {sd}')

            self._stats.gauge('compile_time', mean, tags=[f'key:mean'])
            self._stats.gauge('compile_time', sd, tags=[f'key:sd'])

            now = int(time.perf_counter())
            for n, w in enumerate(tuple(workers)):
                duration = now - w.started_at
                if threshold < duration or not w.is_alive():
                    self.logger.info(f'Terminating worker {w.pid} with compile time {duration}.')
                    w.terminate()
                    new = w.reincarnate()
                    async with gpu_lock:
                        new.start()
                    workers[workers.index(w)] = new
                    self.logger.info('Worker died')
                    self._stats.increment('worker_died')

            await sleep(mean / 3)
