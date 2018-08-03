import asyncio
import time

import torch

from tc_composer.func.affine_transform import AffineTransform
from tc_tuner import settings, async_util
from tc_tuner.option_result import OptionResult
from tc_tuner.tuner import ChildTuner, Tuner
from ..tc_tuner.message_bus import MessageBus
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
        msg_bus = MessageBus('test')
        worker = ChildTuner(
            f=self.f, inps=(self.inp,),
            correct_outs=(self.correct_out,),
            msg_bus=msg_bus)
        worker.start()

        for _ in range(num):
            msg_bus.queue(OptionResult.make_naive())

            while len(msg_bus._result) == 0 and worker.is_alive():
                time.sleep(.1)

            opt_res = msg_bus.pop_result()
            self.assertIsInstance(opt_res.speed, float)


class TestTuner(TorchTestCase):
    def test_evaluate_option(self):
        in_n = 3
        out_n = 5
        inp = torch.randn(1, in_n)
        aff = AffineTransform(in_n=in_n, out_n=out_n)
        aff.recompile(inp)
        tuner = Tuner(aff, inps=(inp,), correct_outs=(aff(inp),))

        ts, opts = tuner._proposer(tuner._in_sizes)

        task = asyncio.ensure_future(tuner.evaluate_option(ts[0], opts[0]))
        task1 = asyncio.ensure_future(tuner.evaluate_option(ts[1], opts[1]))

        def done_task(*x):
            tuner._proposer.apply_grad()
            tuner._evaluator.apply_grad()

        task.add_done_callback(done_task)
        task1.add_done_callback(done_task)

        settings.EVENT_LOOP.run_until_complete(asyncio.sleep(.3))


        opt_res = tuner._msg_bus.pop_queue()
        opt_res.speed = 1
        tuner._msg_bus.add_result(opt_res)
        opt_res = tuner._msg_bus.pop_queue()
        opt_res.speed = tuner.NOT_VIABLE
        tuner._msg_bus.add_result(opt_res)

        settings.EVENT_LOOP.run_until_complete(task)
        settings.EVENT_LOOP.run_until_complete(task1)

        async_util.check_exception(task)
        async_util.check_exception(task1)

