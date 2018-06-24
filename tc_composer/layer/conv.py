from typing import Sequence, Tuple

import torch
from torch.autograd import Variable
from torch.nn import Parameter

from .layer import Layer


class Convolution(Layer):
    def __init__(self,
                 filter_num: int,
                 filter_size: Tuple[int, int],
                 color_num: int = 3,
                 name: str = None):
        super(Convolution, self).__init__(name=name)
        assert len(filter_size) == 2

        self.filter_num = filter_num
        self.color_num = color_num
        self.filter_size = filter_size
        self.weight = Parameter(torch.normal(torch.zeros(filter_num, color_num, *filter_size), .1))
        self.bias = Parameter(torch.normal(torch.zeros(filter_num), .1))

    @property
    def lang(self) -> str:
        return (f"def {self.id}(float(N, C, H, W) I, float(M, C, KH, KW) W, float(M) B) -> (O) {'{'}"
                "   O(n, m, h, w) +=! I(n, c, h + kh, w + kw) * W(m, c, kh, kw)"
                "   O(n, m, h, w) = O(n, m, h, w) + B(m)"
                "}")

    def forward(self, input: Variable, outputs: Sequence[Variable] = None, **kwargs):
        return self.tc_unit(input, self.weight, self.bias, outputs=outputs, **kwargs)
