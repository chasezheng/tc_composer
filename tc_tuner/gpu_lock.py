import asyncio

import torch
from .settings import EVENT_LOOP
from .async_queue import AsyncQueue


class _GPULock:
    __slots__ = ()

    _LOCK_QUEUE = AsyncQueue(maxsize=1)
    _LOCK_QUEUE.put_nowait('gpu_lock')

    _GOTTEN: int = 0
    # todo duplicate logic
    @classmethod
    async def __aenter__(cls):
        if cls._GOTTEN == 0:
            await cls._LOCK_QUEUE.aget()

        cls._GOTTEN += 1

    @classmethod
    async def __aexit__(cls, exc_type, exc, tb):
        torch.cuda.synchronize()
        cls._GOTTEN -= 1

        if cls._GOTTEN == 0:
            cls._LOCK_QUEUE.put_nowait('gpu_lock')

    @classmethod
    def __enter__(cls):
        if cls._GOTTEN == 0:
            cls._LOCK_QUEUE.get()

        cls._GOTTEN += 1

    @classmethod
    def __exit__(cls, exc_type, exc_val, exc_tb):
        torch.cuda.synchronize()
        cls._GOTTEN -= 1

        if cls._GOTTEN == 0:
            cls._LOCK_QUEUE.put_nowait('gpu_lock')


gpu_lock = _GPULock()

del _GPULock
