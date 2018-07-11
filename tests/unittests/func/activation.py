import torch
from torch import nn

from tc_composer.func.activation import Activation, Softmax
from ..torch_test_case import TorchTestCase


class TestActivation(TorchTestCase):

    def setUp(self):
        self.in_n = 6
        self.m = torch.randn(4, self.in_n)

    def test_tanh(self):
        tanh = Activation(input_dim=2, func='tanh')
        tanh.recompile(self.m)
        self.assert_allclose(tanh(self.m), nn.Tanh()(self.m))

    def test_relu(self):
        relu = Activation(input_dim=2, func='relu')
        relu.recompile(self.m)
        self.assert_allclose(relu(self.m), nn.ReLU()(self.m))

    def test_sigmoid(self):
        sigmoid = Activation(input_dim=2, func='sigmoid')
        sigmoid.recompile(self.m)
        self.assert_allclose(sigmoid(self.m), nn.Sigmoid()(self.m))

    def test_compose(self):
        composed = Activation(input_dim=2, func='sigmoid') << Activation(input_dim=2, func='relu')
        composed.recompile(self.m)
        self.assert_allclose(actual=composed(self.m), desired=nn.ReLU()(nn.Sigmoid()(self.m)))

    def test_branch(self):
        branch = Activation(input_dim=2, func='sigmoid') \
                 + Activation(input_dim=2, func='relu') \
                 + Softmax(input_dim=2)
        branch.recompile(self.m)

        a, b, c = branch(self.m)
        self.assert_allclose(actual=a, desired=nn.Sigmoid()(self.m))
        self.assert_allclose(actual=b, desired=nn.ReLU()(self.m))
        self.assert_allclose(actual=c, desired=nn.Softmax(dim=-1)(self.m))

    def test_softmax(self):
        tc_softmax = Softmax(input_dim=2)
        torch_softmax = nn.Softmax(dim=-1)

        tc_softmax.recompile(self.m)
        # tc_softmax.train(False), torch_softmax.train(False)
        self.assert_allclose(tc_softmax(self.m), torch_softmax(self.m))
