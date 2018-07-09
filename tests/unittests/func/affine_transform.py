import torch
from torch import nn
from torch.autograd import Variable

from tc_composer.func.affine_transform import AffineTransform
from ..torch_test_case import TorchTestCase


class TestAffineTransform(TorchTestCase):
    def setUp(self):
        self.batch_size = 3
        self.in_n = 10
        self.out_n = 11

        self.input = torch.randn(self.batch_size, self.in_n)

    def test_with_bias(self):
        tc_aff = AffineTransform(in_n=self.in_n, out_n=self.out_n)
        torch_aff = nn.Linear(self.in_n, self.out_n)

        torch_aff.weight.data = tc_aff.params[0].data.view_as(torch_aff.weight)
        torch_aff.bias.data = tc_aff.params[1].data.view_as(torch_aff.bias)

        tc_aff.recompile(self.input)
        self.assert_allclose(tc_aff(self.input).squeeze(), torch_aff(self.input).squeeze())


    def test_without_bias(self):
        tc_aff = AffineTransform(in_n=self.in_n, out_n=self.out_n, bias=False)
        torch_aff = nn.Linear(self.in_n, self.out_n, bias=False)

        torch_aff.weight.data = tc_aff.params[0].data.view_as(torch_aff.weight)

        tc_aff.recompile(self.input)
        self.assert_allclose(tc_aff(self.input).squeeze(), torch_aff(self.input).squeeze())

