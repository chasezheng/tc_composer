import time

import tensor_comprehensions as tc
import torch

from tc_composer.func.affine_transform import AffineTransform
from tc_tuner.message_bus import MessageBus
from tc_tuner.option_result import OptionResult
from tc_tuner.tuner_v2 import ChildTuner
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

        self.msg_bus = MessageBus('test')

    def test_tune(self):
        num = 2
        worker = ChildTuner(num, f=self.f, inps=(self.inp,), correct_outs=(self.correct_out,),
                            start_option_res=OptionResult(tc.MappingOptions('naive')),
                            msg_bus=self.msg_bus)
        worker.start()

        while worker.is_alive():
            time.sleep(.1)

        for _ in range(num):
            opt_res = self.msg_bus.pop_result()
            self.assertIsInstance(opt_res, OptionResult)
            self.assertIsInstance(opt_res.speed, float)
            self.assertIsNone(opt_res.exc)
