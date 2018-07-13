import torch
from torch import nn

from tc_composer.func.affine_transform import AffineTransform, FusedAffineTransform
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


class TestFusedAffineTransform(TorchTestCase):
    def setUp(self):
        self.batch_size = 2
        self.in_n = 3
        self.hiddens = tuple(range(5, 8))
        self.activations = tuple('sigmoid' for _ in self.hiddens)

        self.input = torch.randn(self.batch_size, self.in_n)

    def test_fused_affine_transfom(self):
        tc_aff = FusedAffineTransform(self.in_n, hiddens=self.hiddens, activations=self.activations)
        self.logger.info(tc_aff.tc_def(self.input))
        tc_aff.recompile(self.input)


        def yield_torch():
            ins = (self.in_n,) + self.hiddens[:-1]
            for in_n, out, activation in zip(ins, self.hiddens, self.activations):
                yield nn.Linear(in_n, out)
                if activation.lower() == 'sigmoid':
                    yield nn.Sigmoid()
                elif activation.lower() == 'relu':
                    yield nn.ReLU()
                elif activation.lower() == 'tanh':
                    yield nn.Tanh()
                elif activation.lower() == 'softmax':
                    yield nn.Softmax(dim=-1)

        torch_aff = nn.Sequential(*yield_torch())

        for p, t in zip(torch_aff.parameters(), tc_aff.params):
            p.data = t.detach().view_as(p)

        self.assert_allclose(actual=tc_aff(self.input), desired=torch_aff(self.input))
