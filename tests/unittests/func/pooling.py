import torch
from torch import nn
from torch.autograd import Variable

from tc_composer.func.pooling import MaxPooling, AveragePooling
from ..torch_test_case import TorchTestCase


class TestPooling(TorchTestCase):
    def setUp(self):
        self.stride = (2, 3)
        self.kernel_size = (5, 7)
        self.in_channels = 3
        self.image_size = (13, 17)
        self.batch_size = 2

    def test_max_pooling(self):
        tc_max_pooling = MaxPooling(stride=self.stride, kernel_size=self.kernel_size)
        torch_max_pooling = nn.MaxPool2d(kernel_size=self.kernel_size, stride=self.stride)
        image = torch.rand(self.batch_size, self.in_channels, *self.image_size)

        tc_max_pooling.recompile(image)
        self.assert_allclose(tc_max_pooling(image), torch_max_pooling(image))

    def test_avg_pooling(self):
        tc_avg_pooling = AveragePooling(stride=self.stride, kernel_size=self.kernel_size)
        torch_avg_pooling = nn.AvgPool2d(kernel_size=self.kernel_size, stride=self.stride)
        image = torch.rand(self.batch_size, self.in_channels, *self.image_size)

        tc_avg_pooling.recompile(image)
        self.assert_allclose(tc_avg_pooling(image), torch_avg_pooling(image))
