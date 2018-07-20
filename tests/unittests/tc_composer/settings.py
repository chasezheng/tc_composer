from tc_composer.settings import tc_type
from ..torch_test_case import TorchTestCase


class TestTCType(TorchTestCase):
    def test_tc_type(self):
        self.assertEqual(tc_type('torch.HalfTensor'), 'half')
        self.assertEqual(tc_type('torch.FloatTensor'), 'float')
        self.assertEqual(tc_type('torch.DoubleTensor'), 'double')
        self.assertEqual(tc_type('torch.LongTensor'), 'long')

        self.assertEqual(tc_type('torch.cuda.HalfTensor'), 'half')
        self.assertEqual(tc_type('torch.cuda.FloatTensor'), 'float')
        self.assertEqual(tc_type('torch.cuda.DoubleTensor'), 'double')
        self.assertEqual(tc_type('torch.cuda.LongTensor'), 'long')
