import torch
from torch.nn import Parameter

from .base import Layer


class AffineTransform(Layer):
    def __init__(self, in_n: int, out_n: int, activation: str = 'tanh', bias: bool = True, name: str = None):
        super(AffineTransform, self).__init__(in_n=in_n, out_n=out_n, name=name)
        assert activation in ('tanh', 'sigmoid',)  # todo more

        self.bias = bias

        self.W1 = Parameter(torch.normal(torch.zeros(out_n, in_n), .1))
        self.B1 = Parameter(torch.normal(torch.zeros(out_n), .1))

    @property
    def lang(self) -> str:
        return (f"def {self.id}(float(batch_size, in_n) I, float(out_n, in_n) W1, float(out_n) B1) -> (O1) {'{'}\n"
                "    O1(b, n) +=! I(b, i) * W1(n, i)\n"
                "    O1(b, n) = O1(b, n) + B1(n)\n" if self.bias else ''  # todo Can replace with this? `O1(b, n) += B1(n)\n`
                                                                      "}")

    def tc_call_args(self, tensor, **kwargs):
        return (tensor, self.W1, self.B1), kwargs
