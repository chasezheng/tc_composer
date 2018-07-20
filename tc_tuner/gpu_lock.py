import asyncio

import torch

from .async_queue import AsyncQueue


class _GPULock:
    __slots__ = ()  # todo allow nested with statements

    _LOCK_QUEUE = AsyncQueue(maxsize=1)
    _LOCK_QUEUE.put_nowait('gpu_lock')

    @classmethod
    async def __aenter__(cls):
        await cls._LOCK_QUEUE.aget()

    @classmethod
    async def __aexit__(cls, exc_type, exc, tb):
        torch.cuda.synchronize()
        cls._LOCK_QUEUE.put_nowait('gpu_lock')

    @classmethod
    def __enter__(cls):
        asyncio.wait((asyncio.Task(cls.__aenter__()),))

    @classmethod
    def __exit__(cls, exc_type, exc_val, exc_tb):
        asyncio.wait((asyncio.Task(cls.__aexit__(exc_type, exc_val, exc_tb)),))


gpu_lock = _GPULock()

del _GPULock
