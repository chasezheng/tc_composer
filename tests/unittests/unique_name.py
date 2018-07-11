from tc_composer.unique_name import UniqueName
from .torch_test_case import TorchTestCase
from uuid import uuid4


class TestUniqueName(TorchTestCase):
    def test_str(self):
        unique_str = str(uuid4()) + 'T'

        self.assertEqual(str(UniqueName(prefix=unique_str)), unique_str)

    def test_uniqueness(self):
        name = 'Test'
        n0 = UniqueName(prefix=name)
        n1 = UniqueName(prefix=name)

        self.assertNotEqual(str(n0), str(n1))
        self.assertTrue(str(n0).startswith(name))
        self.assertTrue(str(n1).startswith(name))

    def test_lower_case(self):
        with self.assertRaises(AssertionError):
            UniqueName('single_lower_case_letter'[0])

# todo test TensorName