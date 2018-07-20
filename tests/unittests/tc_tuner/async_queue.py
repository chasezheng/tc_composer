import sys
from multiprocessing import Process
from typing import Callable

import torch
from torch import Tensor

from tc_composer.func.affine_transform import AffineTransform
from tc_tuner.async_queue import AsyncQueue
from tc_tuner.settings import EVENT_LOOP
from ..torch_test_case import TorchTestCase


class TestAsyncQueue(TorchTestCase):
    class MyProcess(Process):
        def __init__(self,
                     target: Callable,
                     in_queue: AsyncQueue[Tensor],
                     out_queue: AsyncQueue[BaseException]):
            super(TestAsyncQueue.MyProcess, self).__init__(target=target)
            self.in_queue = in_queue
            self.out_queue = out_queue
            self._target = target

        def run(self):
            try:
                out = self._target(self.in_queue.get())
            except:
                self.out_queue.put(sys.exc_info()[1])
                raise
            else:
                self.out_queue.put(out)

    def test_exceptions(self):
        queue = AsyncQueue(maxsize=1)
        queue.put(1)

        with self.assertRaises(queue.Full):
            queue.put_nowait(1)

        queue.get()

        with self.assertRaises(queue.Empty):
            queue.get_nowait()

    def test_serialize(self):
        in_queue = AsyncQueue()
        out_queue: AsyncQueue[BaseException] = AsyncQueue()
        f = torch.sigmoid

        process = TestAsyncQueue.MyProcess(f, in_queue, out_queue)
        inp = torch.randn(1)
        in_queue.put(inp)
        process.start()

        try:
            res = out_queue.get()

            if torch.is_tensor(res):
                self.assert_allclose(res, f(inp))
            else:
                raise res
        finally:
            process.terminate()

    @staticmethod
    def compile_and_run(x):
        aff, inp = x
        aff.recompile(inp)
        return aff(inp)

    def test_serialize_tc_composer(self):
        aff = AffineTransform(3, 4)
        tc_inp = torch.randn(2, 3)
        aff.recompile(tc_inp)

        in_queue = AsyncQueue()
        out_queue = AsyncQueue()

        process = TestAsyncQueue.MyProcess(self.compile_and_run, in_queue, out_queue)
        process.start()
        try:
            in_queue.put((aff, tc_inp))

            res = out_queue.get()

            if torch.is_tensor(res):
                self.assert_allclose(res, aff(tc_inp))
            else:
                raise res
        finally:
            process.terminate()

    def test_aget(self):
        queue = AsyncQueue()
        queue.put(1)

        f = queue.aget()
        EVENT_LOOP.run_until_complete(f)

    def test_aput(self):
        queue = AsyncQueue()

        f = queue.aput(1)
        EVENT_LOOP.run_until_complete(f)
