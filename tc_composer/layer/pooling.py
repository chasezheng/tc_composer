from typing import Sequence

from torch.autograd import Variable

from .layer import Layer
from ..settings import TYPE_NAME


class AveragePooling(Layer):
    def __init__(self, stride: Sequence[int], kernel_size: Sequence[int], name: str = None):
        super(AveragePooling, self).__init__(name=name)
        self.stride = stride
        self.kernel_size = kernel_size

    @property
    def lang(self) -> str:
        return (f"def {self.forward_name}({TYPE_NAME}(B, C, H, W) input) -> (output) {'{'}\n"
                f"    output(b, c, h, w) +=! input(b, c, {self.stride[0]}*h + kh, {self.stride[1]}*w + kw)\n"
                f"        where kh in 0:{self.kernel_size[0]}, kw in 0:{self.kernel_size[1]}\n"
                f"    output(b, c, h, w) = fdivide(output(b, c, h, w), {self.kernel_size[0]*self.kernel_size[1]})"
                f"        where exists output(b, c, h, w)"
                "}")

    def forward(self, input: Variable, outputs: Sequence[Variable] = None, **kwargs):
        return self.tc_unit(input, outputs=outputs, **kwargs)


class MaxPooling(Layer):
    def __init__(self, stride: Sequence[int], kernel_size: Sequence[int], name: str = None):
        super(MaxPooling, self).__init__(name=name)
        self.stride = stride
        self.kernel_size = kernel_size

    @property
    def lang(self) -> str:
        return (f"def {self.forward_name}({TYPE_NAME}(B, C, H, W) input) -> (output) {'{'}\n"
                f"  output(b, c, h, w) max=! input(b, c, h*{self.stride[0]} + kh, w*{self.stride[1]} + kw)\n"
                f"      where kh in 0:{self.kernel_size[0]}, kw in 0:{self.kernel_size[1]}\n"
                "}")

    def forward(self, input: Variable, outputs: Sequence[Variable] = None, **kwargs):
        return self.tc_unit(input, outputs=outputs, **kwargs)
