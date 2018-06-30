import torch
from torch import nn
from torch.autograd import Variable

from tc_composer.layer.affine_transform import AffineTransform
from ..torch_test_case import TorchTestCase


class TestAffineTransform(TorchTestCase):
    def setUp(self):
        self.in_n = 10
        self.out_n = 10

        self.input = Variable(torch.randn(3, 10))

    def test_with_bias(self):
        tc_aff = AffineTransform(self.in_n, self.out_n)
        torch_aff = nn.Linear(self.in_n, self.out_n)

        torch_aff.weight.data = tc_aff.weight.data.view_as(torch_aff.weight)
        torch_aff.bias.data = tc_aff.bias.data.view_as(torch_aff.bias)
        self.assert_allclose(tc_aff(self.input).squeeze(), torch_aff(self.input).squeeze())

        # One dimension vector
        self.assert_allclose(tc_aff(self.input[0].squeeze()).squeeze(), torch_aff(self.input[0].squeeze()).squeeze())

    def test_without_bias(self):
        tc_aff = AffineTransform(self.in_n, self.out_n, bias=False)
        torch_aff = nn.Linear(self.in_n, self.out_n, bias=False)

        torch_aff.weight.data = tc_aff.weight.data.view_as(torch_aff.weight)
        self.assert_allclose(tc_aff(self.input).squeeze(), torch_aff(self.input).squeeze())

        # One dimension vector
        self.assert_allclose(tc_aff(self.input[0].squeeze()).squeeze(), torch_aff(self.input[0].squeeze()).squeeze())
