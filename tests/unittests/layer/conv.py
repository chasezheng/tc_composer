import torch
from torch.autograd import Variable
from torch.nn import Conv2d

from tc_composer.layer.conv import Convolution
from ..torch_test_case import TorchTestCase


class TestConv(TorchTestCase):
    RTOL = 1e-10

    def setUp(self):
        self.batch_size = 2
        self.groups = 1
        self.in_channels = 3
        self.in_height = 17
        self.in_width = 19
        self.tc_image = Variable(
            torch.randn(self.batch_size, self.groups, self.in_channels, self.in_height, self.in_width),
            requires_grad=True)
        self.torch_image = Variable(
            torch.Tensor(self.batch_size, self.groups * self.in_channels, self.in_height, self.in_width),
            requires_grad=True)
        self.torch_image.data.copy_(self.tc_image.data.view_as(self.torch_image))

        self.out_channels = 3
        self.kernel_size = (11, 11)

    def test_simple(self):
        tc_conv = Convolution(self.in_channels, self.out_channels, kernel_size=self.kernel_size)
        torch_conv = Conv2d(self.in_channels, self.out_channels, self.kernel_size)

        torch_conv.weight.data.copy_(tc_conv.weight.data.view_as(torch_conv.weight))
        torch_conv.bias.data.copy_(tc_conv.bias.data.view_as(torch_conv.bias))

        tc_conv.train(False), torch_conv.train(False)
        self.assert_allclose(actual=tc_conv(self.tc_image).squeeze(),
                             desired=torch_conv(self.torch_image).squeeze())

        tc_conv.train(True), torch_conv.train(True)
        tc_out = tc_conv(self.tc_image).squeeze()
        torch_out = torch_conv(self.torch_image).squeeze()
        self.assert_allclose(actual=tc_out, desired=torch_out)

        tc_out.sum().backward()
        torch_out.sum().backward()
        self.assert_allclose(actual=tc_conv.weight.grad.squeeze(),
                             desired=torch_conv.weight.grad.squeeze())
        self.assert_allclose(actual=tc_conv.bias.grad.squeeze(),
                             desired=torch_conv.bias.grad.squeeze())
        self.assert_allclose(actual=self.tc_image.grad.view(-1),
                             desired=self.torch_image.grad.view(-1))
        self.assertIsNotNone(tc_conv.weight.grad)
        self.assertIsNotNone(tc_conv.bias.grad)

    def test_bias(self):
        tc_conv = Convolution(self.in_channels, self.out_channels, kernel_size=self.kernel_size, bias=False)
        torch_conv = Conv2d(self.in_channels, self.out_channels, self.kernel_size, bias=False)

        # Their bias remain different
        torch_conv.weight.data.copy_(tc_conv.weight.data.view_as(torch_conv.weight))

        tc_conv.train(False), torch_conv.train(False)
        self.assert_allclose(actual=tc_conv(self.tc_image).squeeze(),
                             desired=torch_conv(self.torch_image).squeeze())

        tc_conv.train(True), torch_conv.train(True)
        tc_out = tc_conv(self.tc_image).squeeze()
        torch_out = torch_conv(self.torch_image).squeeze()
        self.assert_allclose(actual=tc_out, desired=torch_out)

        tc_out.sum().backward()
        torch_out.sum().backward()
        self.assert_allclose(actual=tc_conv.weight.grad.squeeze(),
                             desired=torch_conv.weight.grad.squeeze())
        self.assert_allclose(actual=self.tc_image.grad.view(-1),
                             desired=self.torch_image.grad.view(-1))
        if tc_conv.bias is not None and tc_conv.bias.grad is not None:
            self.assert_allclose(actual=tc_conv.bias.grad,
                                 desired=torch.zeros(*tc_conv.bias.shape))

    def test_group(self):
        groups = 3
        tc_image = Variable(
            torch.randn(self.batch_size, groups, self.in_channels, self.in_height, self.in_width),
            requires_grad=True
        )
        torch_image = Variable(
            torch.Tensor(self.batch_size, groups * self.in_channels, self.in_height, self.in_width),
            requires_grad=True
        )
        torch_image.data.copy_(tc_image.data.view_as(torch_image))

        tc_conv = Convolution(self.in_channels, self.out_channels, kernel_size=self.kernel_size, groups=groups)
        torch_conv = Conv2d(self.in_channels * groups, out_channels=self.out_channels * groups,
                            kernel_size=self.kernel_size, groups=groups)
        torch_conv.weight.data.copy_(tc_conv.weight.data.view_as(torch_conv.weight))
        torch_conv.bias.data.copy_(tc_conv.bias.data.view_as(torch_conv.bias))

        tc_conv.train(False), torch_conv.train(False)
        self.assert_allclose(actual=tc_conv(tc_image).view(-1),
                             desired=torch_conv(torch_image).view(-1))

        tc_conv.train(True), torch_conv.train(True)
        tc_out = tc_conv(tc_image).view(-1)
        torch_out = torch_conv(torch_image).view(-1)
        self.assert_allclose(actual=tc_out, desired=torch_out)

        tc_out.sum().backward()
        torch_out.sum().backward()
        self.assert_allclose(actual=tc_conv.weight.grad.view(-1),
                             desired=torch_conv.weight.grad.view(-1))
        self.assert_allclose(actual=tc_conv.bias.grad.view(-1),
                             desired=torch_conv.bias.grad.view(-1))
        self.assert_allclose(actual=tc_image.grad.view(-1),
                             desired=torch_image.grad.view(-1))
        self.assertIsNotNone(tc_conv.weight.grad)
        self.assertIsNotNone(tc_conv.bias.grad)

    def test_stride(self):
        stride = (2, 3)
        tc_conv = Convolution(self.in_channels, self.out_channels, kernel_size=self.kernel_size,
                              stride=stride)
        torch_conv = Conv2d(self.in_channels, self.out_channels, self.kernel_size, stride=stride)

        torch_conv.weight.data.copy_(tc_conv.weight.data.view_as(torch_conv.weight))
        torch_conv.bias.data.copy_(tc_conv.bias.data.view_as(torch_conv.bias))

        tc_conv.train(False), torch_conv.train(False)
        self.assert_allclose(actual=tc_conv(self.tc_image).squeeze(),
                             desired=torch_conv(self.torch_image).squeeze())

        tc_conv.train(True), torch_conv.train(True)
        tc_out = tc_conv(self.tc_image).squeeze()
        torch_out = torch_conv(self.torch_image).squeeze()
        self.assert_allclose(actual=tc_out, desired=torch_out)

        tc_out.sum().backward()
        torch_out.sum().backward()
        self.assert_allclose(actual=tc_conv.weight.grad.squeeze(),
                             desired=torch_conv.weight.grad.squeeze())
        self.assert_allclose(actual=tc_conv.bias.grad.squeeze(),
                             desired=torch_conv.bias.grad.squeeze())

        self.assert_allclose(actual=self.tc_image.grad.view(-1),
                             desired=self.torch_image.grad.view(-1))
        self.assertIsNotNone(tc_conv.weight.grad)
        self.assertIsNotNone(tc_conv.bias.grad)

    def test_padding(self):
        padding = (2, 3)
        tc_conv = Convolution(self.in_channels, self.out_channels, kernel_size=self.kernel_size,
                              padding=padding)
        torch_conv = Conv2d(self.in_channels, self.out_channels, self.kernel_size, padding=padding)

        torch_conv.weight.data.copy_(tc_conv.weight.data.view_as(torch_conv.weight))
        torch_conv.bias.data.copy_(tc_conv.bias.data.view_as(torch_conv.bias))

        tc_conv.train(False), torch_conv.train(False)
        self.assert_allclose(actual=tc_conv(self.tc_image).squeeze(),
                             desired=torch_conv(self.torch_image).squeeze())

        tc_conv.train(True), torch_conv.train(True)
        tc_out = tc_conv(self.tc_image).squeeze()
        torch_out = torch_conv(self.torch_image).squeeze()
        self.assert_allclose(actual=tc_out, desired=torch_out)

        tc_out.sum().backward()
        torch_out.sum().backward()
        self.assert_allclose(actual=tc_conv.weight.grad.squeeze(),
                             desired=torch_conv.weight.grad.squeeze())
        self.assert_allclose(actual=tc_conv.bias.grad.squeeze(),
                             desired=torch_conv.bias.grad.squeeze())
        self.assert_allclose(actual=self.tc_image.grad.view(-1),
                             desired=self.torch_image.grad.view(-1))
        self.assertIsNotNone(tc_conv.weight.grad)
        self.assertIsNotNone(tc_conv.bias.grad)

    def test_stirde_padding(self):
        stride = (4, 4)
        padding = (2, 2)
        tc_conv = Convolution(self.in_channels, self.out_channels, kernel_size=self.kernel_size,
                              padding=padding, stride=stride)
        torch_conv = Conv2d(self.in_channels, self.out_channels, self.kernel_size, stride=stride,
                            padding=padding)

        torch_conv.weight.data.copy_(tc_conv.weight.data.view_as(torch_conv.weight))
        torch_conv.bias.data.copy_(tc_conv.bias.data.view_as(torch_conv.bias))

        tc_conv.train(False), torch_conv.train(False)
        self.assert_allclose(actual=tc_conv(self.tc_image).squeeze(),
                             desired=torch_conv(self.torch_image).squeeze())

        tc_conv.train(True), torch_conv.train(True)
        tc_out = tc_conv(self.tc_image).squeeze()
        torch_out = torch_conv(self.torch_image).squeeze()
        self.assert_allclose(actual=tc_out, desired=torch_out)

        tc_out.sum().backward()
        torch_out.sum().backward()
        self.assert_allclose(actual=tc_conv.weight.grad.squeeze(),
                             desired=torch_conv.weight.grad.squeeze())
        self.assert_allclose(actual=tc_conv.bias.grad.squeeze(),
                             desired=torch_conv.bias.grad.squeeze())
        self.assert_allclose(actual=self.tc_image.grad.view(-1),
                             desired=self.torch_image.grad.view(-1))
        self.assertIsNotNone(tc_conv.weight.grad)
        self.assertIsNotNone(tc_conv.bias.grad)
