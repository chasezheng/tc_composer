from typing import Sequence

from torch.autograd import Variable

from .layer import Layer


class AveragePooling(Layer):
    def __init__(self, sH: float, sW: float, kH, kW, name: str = None):
        super(AveragePooling, self).__init__(name=name)
        self.sH = sH
        self.sW = sW
        self.kH = kH
        self.kW = kW

    @property
    def lang(self) -> str:
        return (f"def {self.id}(float(B, C, H, W) input) -> (output) {'{'}\n"
                f"  output(b, c, h, w) += input(b, c, h * {self.sH} + kh, w * {self.sW} + kw) where kh in 0:{self.kH}, kw in 0:{self.kW}\n"
                "}")

    def forward(self, input: Variable, outputs: Sequence[Variable] = None, **kwargs):
        return self.tc_unit(input, outputs=outputs, **kwargs)


class MaxPooling(Layer):
    def __init__(self, sH: float, sW: float, kH, kW, name: str = None):
        super(MaxPooling, self).__init__(name=name)
        self.sH = sH
        self.sW = sW
        self.kH = kH
        self.kW = kW

    @property
    def lang(self) -> str:
        return (f"def {self.id}(float(B, C, H, W) input) -> (output) {'{'}\n"
                f"  output(b, c, h, w) max= input(b, c, h * {self.sH} + kh, w * {self.sW} + kw) where kh in 0:{self.kH}, kw in 0:{self.kW}\n"
                "}")

    def forward(self, input: Variable, outputs: Sequence[Variable] = None, **kwargs):
        return self.tc_unit(input, outputs=outputs, **kwargs)
