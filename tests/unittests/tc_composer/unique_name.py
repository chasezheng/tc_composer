from uuid import uuid4

from tc_composer.unique_name import UniqueName, Size, TensorName
from ..torch_test_case import TorchTestCase
from torch import Tensor

class TestUniqueName(TorchTestCase):
    def test_str(self):
        unique_str = str(uuid4()) + 'T'
        unique_name = UniqueName(prefix=unique_str)

        self.assertEqual(unique_name, unique_str)
        self.assertIsInstance(unique_name, str)
        self.assertEqual(hash(unique_str), hash(unique_name))
        self.assertEqual(unique_name.capitalize(), unique_str.capitalize())

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


class TestSize(TorchTestCase):
    def test_num(self):
        num = 10
        size = Size(num)
        self.assertEqual(str(size), str(num))

        with self.assertRaises(AssertionError):
            size.num = num + 1

        size1 = Size()
        size1.num = num
        self.assertEqual(str(size1), str(num))

    def test_add(self):
        size = Size()
        size1 = Size()
        size_num = Size(11)

        self.assertEqual(size.add(10), f'{size} + 10')
        self.assertEqual(size.add(size1), f'{size} + {size1}')
        self.assertEqual(size.add(size_num), f'{size} + {size_num}')
        self.assertEqual(size.add(size1, size_num), f'{size} + {size_num} + {size1}')
        self.assertEqual(size.add(size1, size_num, 10), f'{size} + {10 + size_num.num} + {size1}')

        self.assertEqual(size_num.add(10), f'{size_num.num + 10}')
        self.assertEqual(size_num.add(size1), f'{size_num} + {size1}')
        self.assertEqual(size_num.add(size), f'{size_num} + {size}')
        self.assertEqual(size_num.add(size1, size), f'{size_num} + {size1} + {size}')
        self.assertEqual(size_num.add(size1, size, 10), f'{10 + size_num.num} + {size1} + {size}')

    def test_sub(self):
        size = Size()
        size1 = Size()
        size_num = Size(11)

        self.assertEqual(size.sub(10), f'{size} - 10')
        self.assertEqual(size.sub(size1), f'{size} - {size1}')
        self.assertEqual(size.sub(size_num), f'{size} - {size_num}')
        self.assertEqual(size.sub(size1, size_num), f'{size} - {size_num} - {size1}')
        self.assertEqual(size.sub(size1, size_num, 10), f'{size} - {10 + size_num.num} - {size1}')

        self.assertEqual(size_num.sub(10), f'{size_num.num - 10}')
        self.assertEqual(size_num.sub(size1), f'{size_num} - {size1}')
        self.assertEqual(size_num.sub(size), f'{size_num} - {size}')
        self.assertEqual(size_num.sub(size1, size), f'{size_num} - {size1} - {size}')
        self.assertEqual(size_num.sub(size1, size, 10), f'{size_num.num - 10} - {size1} - {size}')


class TestTensorName(TorchTestCase):
    def test_dim(self):
        dim = 5
        t = TensorName(dim=dim)
        self.assertEqual(t.dim, dim)
        self.assertEqual(len(t.sizes), dim)
        self.assertEqual(len(t.indices), dim)

        with self.assertRaises(AssertionError):
            TensorName(dim=dim, sizes=tuple(Size() for _ in range(dim + 1)))

    def test_type(self):
        from tc_composer import settings

        self.assertEqual(TensorName(dim=1).type, settings.DEFAULT_TYPE)
        self.assertEqual(TensorName(dim=1, type='float').type, 'float')
        with self.assertRaises(AssertionError):
            TensorName(dim=1, type='myfloat')

    def test_new_from(self):
        sizes = (1, 2)
        t = Tensor(*sizes)

        self.assertEqual(tuple(s.num for s in TensorName.new_from(t).sizes), sizes)

    def test_make_pair(self):
        sizes = (1, 2)
        n, t = TensorName.make_pair(sizes)

        self.assertEqual(tuple(s.num for s in n.sizes), sizes)
        self.assertEqual(tuple(s.num for s in n.sizes), t.shape)
