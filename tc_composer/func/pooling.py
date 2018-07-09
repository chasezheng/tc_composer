from typing import Sequence

from .function_with_params import FunctionWithParams
from ..unique_name import TensorName


class AveragePooling(FunctionWithParams):  # todo dim
    def __init__(self, stride: Sequence[int], kernel_size: Sequence[int]):
        in_name = TensorName(dim=4, prefix='input')
        out_name = TensorName(dim=4, prefix='output')
        out_name.sizes[0] = in_name.sizes[0]  # Batch_size
        out_name.sizes[1] = in_name.sizes[1]  # In_channels
        super(AveragePooling, self).__init__(in_names=[in_name], outs_to_keep=[out_name])
        self.stride = stride
        self.kernel_size = kernel_size

    @property
    def named_params(self):
        return ()

    @property
    def def_body(self) -> str:
        input, = self.in_names
        output, = self.outs_to_keep

        return (f"{output}(b, c, h, w) +=! {input}(b, c, {self.stride[0]}*h + kh, {self.stride[1]}*w + kw)\n"
                f"    where kh in 0:{self.kernel_size[0]}, kw in 0:{self.kernel_size[1]}\n"
                f"{output}(b, c, h, w) = fdivide({output}(b, c, h, w), {self.kernel_size[0]*self.kernel_size[1]})\n"
                f"    where exists {output}(b, c, h, w)")


class MaxPooling(FunctionWithParams):
    def __init__(self, stride: Sequence[int], kernel_size: Sequence[int]):
        in_name = TensorName(dim=4, prefix='input')
        out_name = TensorName(dim=4, prefix='output')
        out_name.sizes[0] = in_name.sizes[0]  # Batch_size
        out_name.sizes[1] = in_name.sizes[1]  # In_channels
        super(MaxPooling, self).__init__(in_names=[in_name], outs_to_keep=[out_name])
        self.stride = stride
        self.kernel_size = kernel_size

    @property
    def named_params(self):
        return ()

    @property
    def def_body(self) -> str:
        input, = self.in_names
        output, = self.outs_to_keep

        return (f"{output}(b, c, h, w) max=! {input}(b, c, h*{self.stride[0]} + kh, w*{self.stride[1]} + kw)\n"
                f"    where kh in 0:{self.kernel_size[0]}, kw in 0:{self.kernel_size[1]}")
