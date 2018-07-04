from typing import Sequence

import tensor_comprehensions as tc
from torch import Tensor

from .function_with_params import FunctionWithParams
from ..settings import TYPE_NAME


class Activation(FunctionWithParams):
    def __init__(self, func: str):
        super(Activation, self).__init__()
        assert func.lower() in ('tanh', 'sigmoid', 'relu')
        self._func = func.lower()

    def __call__(self, t: Tensor, outputs: Sequence[Tensor] = ()) -> Tensor:
        return super(Activation, self).__call__(t.view(-1), outputs=outputs).view_as(t)

    @property
    def params(self):
        return ()

    @property
    def tc_def(self) -> str:

        if self._func == 'tanh':
            return (f"def activation({TYPE_NAME}(I) input) -> (output) {'{'}\n"
                    f"    output(i) = tanh(input(i))\n"
                    "}")
        elif self._func == 'sigmoid':
            return (f"def sigmoid({TYPE_NAME}(I) input) -> (output) {'{'}\n"
                    f"    output(i) = 1 / (1 + exp(-input(i)))\n"
                    "}")
        elif self._func == 'relu':
            return (f"def relu({TYPE_NAME}(I) input) -> (output) {'{'}\n"
                    "    output(i) = fmax(input(i), 0)\n"
                    "}")
        else:
            raise Exception(f"Unexpected func {self._func}")

    def recompile(self, input: Tensor, option: tc.MappingOptions = None):
        super(Activation, self).recompile(input.view(-1), option=option)


class Softmax(FunctionWithParams):
    """Perform softmax on dim=-1
    """

    def __init__(self):
        super(Softmax, self).__init__()

    @property
    def params(self):
        return ()

    @property
    def tc_def(self) -> str:
        return (f"def softmax({TYPE_NAME}(N, D) input) -> (output, maxVal, tmp, l1norm) {'{'}\n"
                "   maxVal(n) max=! input(n, d)\n"
                "   tmp(n, d) = exp(input(n, d) - maxVal(n))\n"
                "   l1norm(n) +=! tmp(n, d)\n"
                "   output(n, d) = tmp(n, d) / l1norm(n)\n"
                "}")

    def __call__(self, t: Tensor, outputs: Sequence[Tensor] = ()) -> Tensor:
        return super(Softmax, self).__call__(t, outputs=outputs)[0]
