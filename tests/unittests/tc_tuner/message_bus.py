import time

import tensor_comprehensions as tc

from tc_tuner.message_bus import MessageBus, Worker
from tc_tuner.option_result import OptionResult
from tc_tuner.settings import EVENT_LOOP
from ..torch_test_case import TorchTestCase


class TestMessageBus(TorchTestCase):
    class TestWorker(Worker):
        def __init__(self):
            super(TestMessageBus.TestWorker, self).__init__()
            self._started_at = int(time.time())

        @property
        def done(self):
            return False

        @property
        def started_at(self):
            return self._started_at

        def run(self):
            EVENT_LOOP.stop()
            time.sleep(60)

        def reincarnate(self):
            return type(self)()

    def setUp(self):
        self.msg_bus = MessageBus(name='test')
        self.worker = TestMessageBus.TestWorker()
        self.opt_res = OptionResult(tc.MappingOptions('naive'))

    def test_started(self):
        self.worker.start()

        with self.assertRaises(IndexError):
            self.msg_bus.pop_result()

        self.msg_bus.add_result(self.opt_res)
        self.assertEqual(self.opt_res, self.msg_bus.pop_result())

    def test_kill_zombie(self):
        self.worker.start()
        self.msg_bus._compile_time[0], self.msg_bus._compile_time[1] = .5, .05 ** 2 + .5 ** 2
        time.sleep(1)
        next(self.msg_bus.kill_zombie(self.worker).__await__())

        self.worker.join()
        self.assertTrue(not self.worker.is_alive())
