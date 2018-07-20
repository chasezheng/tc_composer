import unittest

import torch

from tc_composer.func.merge import Sum, Concat
from ...torch_test_case import TorchTestCase


class TestSum(TorchTestCase):
    def setUp(self):
        self.size = tuple(range(1, 4))
        self.t0 = torch.randn(*self.size)
        self.t1 = torch.randn(*self.size)

    def test_sum(self):
        sum = Sum(num_ins=2, in_dim=len(self.size))
        self.logger.info(sum.tc_def(self.t0, self.t1))
        sum.recompile(self.t0, self.t1)

        self.assert_allclose(actual=sum(self.t0, self.t1), desired=self.t0 + self.t1)

    def test_branch(self):
        sums = tuple(Sum(num_ins=1, in_dim=len(self.size)) for _ in range(3))
        func = sum(sums) << Sum(num_ins=3, in_dim=len(self.size))
        func.recompile(self.t0)
        self.assert_allclose(actual=func(self.t1), desired=3 * self.t1)


class TestConcat(TorchTestCase):
    def setUp(self):
        self.dim = 3
        self.ins = (torch.randn(2, 2, 3), torch.randn(2, 2, 3))

    @unittest.skip
    def test_concat(self):
        stack = Concat(num_ins=len(self.ins), in_dim=self.dim)
        stack.recompile(*self.ins)

        self.assert_allclose(actual=stack(*self.ins), desired=torch.cat(self.ins, dim=-1))

    @unittest.skip
    def test_branch(self):
        sums = tuple(Sum(num_ins=1, in_dim=self.dim) for _ in range(3))
        func = sum(sums) << Concat(num_ins=len(sums), in_dim=self.dim)
        func.recompile(self.ins[0])
        self.assert_allclose(actual=func(self.ins[0]), desired=torch.cat(3 * [self.ins[0]], dim=-1))
