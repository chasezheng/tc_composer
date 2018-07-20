import asyncio
from functools import lru_cache
from multiprocessing import queues, Queue
from typing import MutableMapping, Generic, TypeVar, Callable, Optional


T = TypeVar('T')


class AsyncQueue(Generic[T]):
    __slots__ = '_queue', '_get_retry_delay', '_put_retry_delay', '_exponential_backoff', '_collected'

    Empty = queues.Empty
    Full = queues.Full

    RECOVERY_BIAS = 3

    def __init__(self, maxsize: int = -1):
        self._get_retry_delay: float = .1
        self._put_retry_delay: float = .1
        self._exponential_backoff: float = 1.001
        self._queue = Queue(maxsize=maxsize)
        self._collected: MutableMapping = {}

    @property
    @lru_cache(maxsize=None)
    def empty(self) -> Callable[[], bool]:
        return self._queue.empty

    @property
    @lru_cache(maxsize=None)
    def full(self) -> Callable[[], bool]:
        return self._queue.full

    @property
    @lru_cache(maxsize=None)
    def get_nowait(self) -> Callable[[], T]:
        return self._queue.get_nowait

    @property
    @lru_cache(maxsize=None)
    def put_nowait(self) -> Callable[[T], None]:
        return self._queue.put_nowait

    @property
    @lru_cache(maxsize=None)
    def get(self) -> Callable[[Optional[bool], Optional[float]], T]:
        return self._queue.get

    @property
    @lru_cache(maxsize=None)
    def put(self) -> Callable[[T, Optional[bool], Optional[float]], None]:
        return self._queue.put

    @property
    @lru_cache(maxsize=None)
    def qsize(self) -> Callable[[], int]:
        return self._queue.qsize

    async def aget(self) -> T:
        delay = self._get_retry_delay
        backoff = self._exponential_backoff
        recover_bias = self.RECOVERY_BIAS

        while True:
            if not self.empty():
                try:
                    return self.get(False)
                except queues.Empty:
                    pass
                finally:
                    self._get_retry_delay = delay / (backoff ** recover_bias)
            await asyncio.sleep(delay)
            delay *= backoff

    async def aput(self, obj: T) -> None:
        delay = self._put_retry_delay
        backoff = self._exponential_backoff
        recover_bias = self.RECOVERY_BIAS

        while True:
            if not self.full():
                try:
                    return self.put(obj, False)
                except queues.Full:
                    pass
                finally:
                    self._put_retry_delay = delay / (backoff ** recover_bias)
            await asyncio.sleep(delay)
            delay *= backoff
