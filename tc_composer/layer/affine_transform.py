import torch
from torch.autograd import Variable
from torch.nn import Parameter
from typing import Sequence
from .layer import Layer
from ..settings import TYPE_NAME


class AffineTransform(Layer):
    def __init__(self, in_n: int, out_n: int, bias: bool = True, name: str = None):
        super(AffineTransform, self).__init__(name=name)
        self.in_n = in_n
        self._use_bias = bias

        self.weight = Parameter(torch.normal(torch.zeros(out_n, in_n), .1))
        self.bias = Parameter(torch.normal(torch.zeros(out_n), .1))

    @property
    def lang(self) -> str:
        if self._use_bias:
            return (f"def {self.forward_name}({TYPE_NAME}(batch_size, in_n) input,\n"
                f"                      {TYPE_NAME}(out_n, in_n) weight,\n"
                f"                      {TYPE_NAME}(out_n) bias) -> (output) {'{'}\n"
                "    output(b, n) +=! input(b, i) * weight(n, i)\n" +
                "    output(b, n) = output(b, n) + bias(n)\n"
                "}")
        else:
            return (f"def {self.forward_name}({TYPE_NAME}(batch_size, in_n) input,\n"
                f"                      {TYPE_NAME}(out_n, in_n) weight) -> (output) {'{'}\n"
                "    output(b, n) +=! input(b, i) * weight(n, i)\n"
                "}")

    def forward(self, input: Variable, outputs: Sequence[Variable] = None, **kwargs):
        if self._use_bias:
            return self.tc_unit(input.view(-1, self.in_n), self.weight, self.bias, outputs=outputs, **kwargs)
        else:
            return self.tc_unit(input.view(-1, self.in_n), self.weight, outputs=outputs, **kwargs)