import asyncio
from collections import abc
from functools import singledispatch, partial
from typing import Callable, Coroutine, MutableSet

from .stats import TunerStats


async def chain(*fs: abc.Awaitable) -> None:
    for f in fs:
        await f


def check_exception(f: asyncio.Future) -> None:
    if f.cancelled() or (not f.done()):
        return
    e = f.exception()
    if e is not None:
        raise e


class Repeat(Callable[..., Coroutine]):
    __slots__ = 'f', 'retry_delay', 'backoff', 'recovery_bias', '__wrapped__'

    _STATS = TunerStats('Repeat')
    _INSTANCES: MutableSet['Repeat'] = set()

    class DelayFactor(int):
        __slots__ = ()
        pass

    def __init__(self, f: Callable,
                 retry_delay: float = .1, backoff: float = 1.005, recovery_bias: float = 3):
        super(Repeat, self).__init__()
        self.f = f
        self.retry_delay = retry_delay
        self.backoff = backoff
        self.recovery_bias = recovery_bias
        self._INSTANCES.add(self)

    async def __call__(self, *args, **kwargs):
        sleep = asyncio.sleep
        delay, backoff, recovery_bias = self.retry_delay, self.backoff, self.recovery_bias

        last_exponent = float('nan')
        try:
            while True:
                out = self.f(*args, **kwargs)
                if isinstance(out, abc.Awaitable):
                    out = await out

                if not isinstance(out, self.DelayFactor):
                    delay /= backoff ** recovery_bias
                    if last_exponent < 0:
                        # Same sign
                        backoff **= 1.1
                    elif last_exponent > 0:
                        backoff **= 1 / 1.1
                    return out

                await sleep(delay)

                if out < 0:
                    exponent = out * recovery_bias
                else:
                    exponent = out

                delay *= backoff ** exponent

                if exponent * last_exponent > 0:
                    # Same sign
                    backoff **= 1.1
                elif exponent * last_exponent < 0:
                    backoff **= 1 / 1.1
                last_exponent = exponent

                self.retry_delay = delay
                self.backoff = backoff
        finally:
            self.retry_delay = delay
            self.backoff = backoff

    def __get__(self, instance, owner):
        return partial(self.__call__, instance)

    @classmethod
    async def monitor(cls):
        while True:
            for i in cls._INSTANCES:
                cls._STATS.async_stats(key=i.f.__name__, delay=i.retry_delay,
                                       backoff=i.backoff, recovery_bias=i.recovery_bias)
            await asyncio.sleep(1)


@singledispatch
def repeat(*args, **kwargs):
    raise NotImplementedError


@repeat.register(abc.Callable)
def __(f: Callable):
    return Repeat(f)


@repeat.register(float)
def __(delay: float = .1, backoff: float = 1.005, recovery_bias: float = 3):
    return lambda f: Repeat(f=f, retry_delay=delay, backoff=backoff, recovery_bias=recovery_bias)
