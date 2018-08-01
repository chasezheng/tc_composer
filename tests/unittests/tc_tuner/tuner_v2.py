import tensor_comprehensions as tc
import torch

from tc_composer.func.affine_transform import AffineTransform
from tc_tuner.async_queue import AsyncQueue
from tc_tuner.modules import Vectorizer
from tc_tuner.tuner_v2 import Worker
from ..torch_test_case import TorchTestCase


class TestWorker(TorchTestCase):
    def setUp(self):
        self.batch = 2
        self.in_n = 3
        self.out_n = 4
        self.inp = torch.randn(self.batch, self.in_n)
        self.f = AffineTransform(self.in_n, self.out_n)
        self.f.recompile(self.inp)
        self.correct_out = self.f(self.inp)

    def test_tune(self):
        num = 2
        progress_queue = AsyncQueue()
        result_queue = AsyncQueue()
        worker = Worker(num, f=self.f, inps=(self.inp,), correct_outs=(self.correct_out,),
                        start_option_res=Vectorizer.parse_option_str(str(tc.MappingOptions('naive'))),
                        progress_queue=progress_queue, result_queue=result_queue)
        worker.start()

        for _ in range(num):
            opt_attr, f = result_queue.get()
            self.assertIsInstance(f, float)
