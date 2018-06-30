from typing import Sequence

from torch.autograd import Variable

from .layer import Layer
from ..settings import TYPE_NAME


class Activation(Layer):
    def __init__(self, func: str):
        super(Activation, self).__init__(name=func.lower().capitalize())
        assert func.lower() in ('tanh', 'sigmoid', 'relu')
        self._func = func.lower()

    @property
    def lang(self) -> str:

        if self._func == 'tanh':
            return (f"def {self.forward_name}({TYPE_NAME}(I) input) -> (output) {'{'}\n"
                    f"    output(i) = tanh(input(i))\n"
                    "}")
        elif self._func == 'sigmoid':
            return (f"def {self.forward_name}({TYPE_NAME}(I) input) -> (output) {'{'}\n"
                    f"    output(i) = 1 / (1 + exp(-input(i)))\n"
                    "}")
        elif self._func == 'relu':
            return (f"def {self.forward_name}({TYPE_NAME}(I) input) -> (output) {'{'}\n"
                    "    output(i) = fmax(input(i), 0)\n"
                    "}")
        else:
            raise Exception(f"Unexpected func {self._func}")

    def forward(self, input: Variable, outputs: Sequence[Variable] = None, **kwargs):
        return self.tc_unit(input.view(-1), outputs=outputs, **kwargs).view_as(input)


class Softmax(Layer):
    """Perform softmax on dim=-1
    """
    def __init__(self, name: str = None):
        super(Softmax, self).__init__(name=name)

    @property
    def lang(self) -> str:
        return (f"def {self.forward_name}({TYPE_NAME}(N, D) input) -> (output, maxVal, tmp, l1norm) {'{'}\n"
                "   maxVal(n) max= input(n, d)\n"
                "   tmp(n, d) = exp(input(n, d) - maxVal(n))\n"
                "   l1norm(n) +=! tmp(n, d)\n"
                "   output(n, d) = tmp(n, d) / l1norm(n)\n"
                "}")

    def forward(self, input: Variable, outputs: Sequence[Variable] = None, **kwargs):
        return self.tc_unit(input.view(-1, input.shape[-1]), outputs=outputs, **kwargs)[0].view_as(input)
