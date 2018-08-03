from multiprocessing import Process, sharedctypes

from tc_tuner.gpu_lock import gpu_lock
from ..torch_test_case import TorchTestCase


class TestGPULock(TorchTestCase):
    @classmethod
    def gotten(cls):
        print(gpu_lock._GOTTEN)
        assert (gpu_lock._GOTTEN) == 0

    def test_child_process(self):
        try:
            type(gpu_lock)._GOTTEN = 10
            p = Process(target=self.gotten)
            p.start()
            p.join()
            self.assertEqual(p.exitcode, 0)
        finally:
            type(gpu_lock)._GOTTEN = 0

    def test_nested(self):
        self.assertEqual(gpu_lock._GOTTEN, 0)
        with gpu_lock:
            self.assertEqual(gpu_lock._GOTTEN, 1)
            with gpu_lock:
                self.assertEqual(gpu_lock._GOTTEN, 2)
            self.assertEqual(gpu_lock._GOTTEN, 1)
        self.assertEqual(gpu_lock._GOTTEN, 0)
