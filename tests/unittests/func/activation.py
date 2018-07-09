import torch
from torch import nn
from torch.autograd import Variable
from ..torch_test_case import TorchTestCase

from tc_composer.func.activation import Activation, Softmax


class TestActivation(TorchTestCase):

    def setUp(self):
        self.v = Variable(torch.randn(20), requires_grad=True)
        self.m = Variable(torch.randn(4, 6), requires_grad=True)

    def test_tanh(self):
        tanh = Activation('tanh')
        tanh.recompile(self.v)
        self.assert_allclose(tanh(self.v), nn.Tanh()(self.v))

    def test_relu(self):
        relu = Activation('relu')
        relu.recompile(self.v)
        self.assert_allclose(relu(self.v), nn.ReLU()(self.v))

    def test_sigmoid(self):
        sigmoid = Activation('sigmoid')
        sigmoid.recompile(self.v)
        self.assert_allclose(sigmoid(self.v), nn.Sigmoid()(self.v))

    def test_softmax(self):
        tc_softmax = Softmax()
        torch_softmax = nn.Softmax(dim=-1)

        tc_softmax.recompile(self.m)
        #tc_softmax.train(False), torch_softmax.train(False)
        self.assert_allclose(tc_softmax(self.m), torch_softmax(self.m))
