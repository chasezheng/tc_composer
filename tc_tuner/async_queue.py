import multiprocessing
from multiprocessing import queues
from typing import Generic, TypeVar

from .async_util import repeat, Repeat

T = TypeVar('T')
DelayFactor = Repeat.DelayFactor


class AsyncQueue(Generic[T], queues.Queue):
    __slots__ = ()

    Empty = queues.Empty
    Full = queues.Full

    def __init__(self, maxsize: int = -1):
        super(AsyncQueue, self).__init__(maxsize=maxsize, ctx=multiprocessing.get_context())

    @repeat(delay=.001)
    async def aget(self) -> T:
        try:
            return self.get(False)
        except self.Empty:
            return DelayFactor(1)

    @repeat(delay=.001)
    async def aput(self, obj: T):
        try:
            self.put(obj, False)
        except self.Full:
            return DelayFactor(1)
