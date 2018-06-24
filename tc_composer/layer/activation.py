from typing import Sequence

from torch.autograd import Variable

from .layer import Layer


class Activation(Layer):
    def __init__(self, func: str, in_n: str, name: str = None):
        super(Activation, self).__init__(name=name)
        assert func in ('tanh', 'sigmoid', 'relu')
        self._func = func
        self.in_n = in_n

    @property
    def lang(self) -> str:
        if self._func == 'tanh':
            return (f"def {self.id}(float(batch_size, in_n) input) -> (output) {'{'}\n"
                    f"    output(b, n) = tanh(I(b, n))\n"
                    "}")
        elif self._func == 'sigmoid':
            return (f"def {self.id}(float(batch_size, in_n) input) -> (output) {'{'}\n"
                    f"    output(b, n) = 1 / (1 + exp(-input(b, n)))\n"
                    "}")
        elif self._func == 'relu':
            return (f"def {self.id}(float(batch_size, in_n) input) -> (output) {'{'}\n"
                    "    output(b, n) = fmax(input(b, n), 0)\n"
                    "}")
        else:
            raise Exception(f"Unexpected func {self._func}")

    def forward(self, input: Variable, outputs: Sequence[Variable] = None, **kwargs):
        return self.tc_unit(input.view(-1, self.in_n), outputs=outputs, **kwargs).view_as(input)


class Tanh(Layer):
    def __init__(self, in_n: str, name: str = None):
        super(Tanh, self).__init__(name=name)
        self.in_n = in_n

    @property
    def lang(self) -> str:
        return (f"def {self.id}(float(I, J) input) -> (output, tmp){'{'}"
                "   output(r, c) max= input(r, reduced) where c in 0:J"
                "   output(r, c) = exp(input(r, c) - output(r, c))"
                "   tmp(r) +=! output(r, r_r)"
                "   output(r, c) = output(r, c) / tmp(r)"
                "}")

    def forward(self, input: Variable, outputs: Sequence[Variable] = None, **kwargs):
        return self.tc_unit(input.view(-1, self.in_n), outputs=outputs, **kwargs)[0].view_as(input)