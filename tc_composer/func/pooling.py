from typing import Sequence

from torch.autograd import Variable

from .function_with_params import FunctionWithParams
from ..settings import TYPE_NAME


class AveragePooling(FunctionWithParams):
    def __init__(self, stride: Sequence[int], kernel_size: Sequence[int]):
        super(AveragePooling, self).__init__()
        self.stride = stride
        self.kernel_size = kernel_size

    @property
    def params(self):
        return ()

    @property
    def tc_def(self) -> str:
        return (f"def AveragePooling({TYPE_NAME}(B, C, H, W) input) -> (output) {'{'}\n"
                f"    output(b, c, h, w) +=! input(b, c, {self.stride[0]}*h + kh, {self.stride[1]}*w + kw)\n"
                f"        where kh in 0:{self.kernel_size[0]}, kw in 0:{self.kernel_size[1]}\n"
                f"    output(b, c, h, w) = fdivide(output(b, c, h, w), {self.kernel_size[0]*self.kernel_size[1]})"
                f"        where exists output(b, c, h, w)"
                "}")



class MaxPooling(FunctionWithParams):
    def __init__(self, stride: Sequence[int], kernel_size: Sequence[int]):
        super(MaxPooling, self).__init__()
        self.stride = stride
        self.kernel_size = kernel_size

    @property
    def params(self):
        return ()

    @property
    def tc_def(self) -> str:
        return (f"def MaxPooling({TYPE_NAME}(B, C, H, W) input) -> (output) {'{'}\n"
                f"  output(b, c, h, w) max=! input(b, c, h*{self.stride[0]} + kh, w*{self.stride[1]} + kw)\n"
                f"      where kh in 0:{self.kernel_size[0]}, kw in 0:{self.kernel_size[1]}\n"
                "}")
