import torch
from torch import nn
from torch.autograd import Variable
from ..torch_test_case import TorchTestCase

from tc_composer.layer.activation import Activation, Softmax


class TestActivation(TorchTestCase):

    def setUp(self):
        self.v = Variable(torch.randn(20), requires_grad=True)
        self.m = Variable(torch.randn(4, 5), requires_grad=True)

    def test_tanh(self):
        self.assert_allclose(Activation('tanh')(self.v), nn.Tanh()(self.v))
        self.assert_allclose(Activation('tanh')(self.m), nn.Tanh()(self.m))

    def test_relu(self):
        self.assert_allclose(Activation('relu')(self.v), nn.ReLU()(self.v))
        self.assert_allclose(Activation('relu')(self.m), nn.ReLU()(self.m))

    def test_sigmoid(self):
        self.assert_allclose(Activation('sigmoid')(self.v), nn.Sigmoid()(self.v))
        self.assert_allclose(Activation('sigmoid')(self.m), nn.Sigmoid()(self.m))

    def test_softmax(self):
        tc_softmax = Softmax()
        torch_softmax = nn.Softmax(dim=-1)

        tc_softmax.train(False), torch_softmax.train(False)
        self.assert_allclose(tc_softmax(self.m), torch_softmax(self.m))
