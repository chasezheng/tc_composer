import pickle
import tempfile

from tc_tuner.option_result import OptionResult
from ..torch_test_case import TorchTestCase


class TestOptionResult(TorchTestCase):
    def test_pickle(self):
        o = OptionResult.make_naive()

        with tempfile.NamedTemporaryFile() as tmp:
            pickle.dump(o, tmp)
            tmp.seek(0)
            o2 = pickle.load(tmp)

        self.assertEqual(str(o.option), str(o2.option))
