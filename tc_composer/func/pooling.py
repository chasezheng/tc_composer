from typing import Sequence

from .function_with_params import FunctionWithParams
from ..unique_name import TensorName


class AveragePooling(FunctionWithParams):  # todo dim
    def __init__(self, stride: Sequence[int], kernel_size: Sequence[int]):

        super(AveragePooling, self).__init__()
        self.stride = stride
        self.kernel_size = kernel_size

    @property
    def named_params(self):
        return ()

    def def_components(self, in_names: Sequence[TensorName] = None):
        if in_names is None:
            input = TensorName(dim=4, prefix='input')
        else:
            input, = in_names

        output = TensorName(dim=4, sizes=(*input.sizes[:2], 'OH', 'OW'), prefix='output')

        body = (f"{output}(b, c, h, w) +=! {input}(b, c, {self.stride[0]}*h + kh, {self.stride[1]}*w + kw)\n"
                f"    where kh in 0:{self.kernel_size[0]}, kw in 0:{self.kernel_size[1]}\n"
                f"{output}(b, c, h, w) = fdivide({output}(b, c, h, w), {self.kernel_size[0]*self.kernel_size[1]})\n"
                f"    where exists {output}(b, c, h, w)")
        return body, (input,), (output,), ()


class MaxPooling(FunctionWithParams):
    def __init__(self, stride: Sequence[int], kernel_size: Sequence[int]):
        super(MaxPooling, self).__init__()
        self.stride = stride
        self.kernel_size = kernel_size

    @property
    def named_params(self):
        return ()

    def def_components(self, in_names: Sequence[TensorName] = None):
        if in_names is None:
            input = TensorName(dim=4, prefix='input')
        else:
            input, = in_names

        output = TensorName(dim=4, sizes=(*input.sizes[:2], 'OH', 'OW'), prefix='output')
        body = (f"{output}(b, c, h, w) max=! {input}(b, c, h*{self.stride[0]} + kh, w*{self.stride[1]} + kw)\n"
                f"    where kh in 0:{self.kernel_size[0]}, kw in 0:{self.kernel_size[1]}")

        return body, (input,), (output,), ()
