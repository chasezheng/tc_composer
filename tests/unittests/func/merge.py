import torch

from tc_composer.func.merge import Sum, Concat
from ..torch_test_case import TorchTestCase


class TestSum(TorchTestCase):
    def setUp(self):
        self.size = tuple(range(1, 4))
        self.t0 = torch.randn(*self.size)
        self.t1 = torch.randn(*self.size)

    def test_sum(self):
        sum = Sum(num_ins=2, in_dim=len(self.size))
        sum.recompile(self.t0, self.t1)

        self.assert_allclose(actual=sum(self.t0, self.t1), desired=self.t0 + self.t1)

    def test_branch(self):
        sums = tuple(Sum(num_ins=1, in_dim=len(self.size)) for _ in range(3))
        func = sum(sums) << Sum(num_ins=3, in_dim=len(self.size))
        func.recompile(self.t0)
        self.assert_allclose(actual=func(self.t1), desired=3 * self.t1)


class TestConcat(TorchTestCase):
    def setUp(self):
        self.size = tuple(range(1, 4))
        self.t0 = torch.randn(*self.size)
        self.t1 = torch.randn(*self.size)

    def test_concat(self):
        concat = Concat(num_ins=2, in_dim=len(self.size))
        self.logger.info(concat.tc_def)
        concat.recompile(self.t0, self.t1)

        self.assert_allclose(actual=concat(self.t0, self.t1), desired=torch.cat((self.t0, self.t1), dim=0))

    def test_branch(self):
        sums = (Sum(num_ins=1, in_dim=len(self.size)) for _ in range(3))
        func = sum(sums) << Concat(num_ins=3, in_dim=len(self.size))
        func.recompile(self.t0)
        self.assert_allclose(actual=func(self.t1), desired=torch.cat(3 * [self.t1]))
