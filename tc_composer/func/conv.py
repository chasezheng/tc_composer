from typing import Tuple

import torch
from functools import lru_cache
from .function_with_params import FunctionWithParams
from ..settings import DEFAULT_TYPE
from ..unique_name import TensorName


class Convolution(FunctionWithParams):  # todo dilation, different dimensions
    __slots__ = '_named_params', 'in_channels', 'out_channels', 'kernel_size', 'padding', 'stride', 'groups', 'use_bias'

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: Tuple[int, int],
                 stride: Tuple[int, int] = (1, 1),
                 padding: Tuple[int, int] = (0, 0),
                 groups: int = 1,
                 bias: bool = True):
        assert len(kernel_size) == len(padding) == len(stride), f"Found: len(kernel_size) == {len(kernel_size)}, " \
                                                                f"len(padding) == {len(padding)}, " \
                                                                f"len(stride) == {len(stride)}"

        in_name = TensorName(dim=5, prefix='input', sizes='B G C H W'.split())
        out_name = TensorName(dim=5, prefix='output', sizes='B G M OH OW'.split())
        in_name.sizes[1] = out_name.sizes[1] = groups
        in_name.sizes[2] = in_channels
        out_name.sizes[0] = in_name.sizes[0]
        out_name.sizes[2] = out_channels
        out_name.sizes[3], out_name.sizes[4] = kernel_size

        super(Convolution, self).__init__(in_names=[in_name], outs_to_keep=[out_name])

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride
        self.groups = groups
        self.use_bias = bias

    @property
    @lru_cache(maxsize=None)
    def named_params(self):
        if self.use_bias:
            return TensorName.make_pair(sizes=(self.groups, self.out_channels, self.in_channels, *self.kernel_size), prefix='weight'), \
                   TensorName.make_pair(sizes=(self.groups, self.out_channels), prefix='bias')
        else:
            return TensorName.make_pair(sizes=(self.groups, self.out_channels, self.in_channels, *self.kernel_size), prefix='weight'),

    @property
    def def_body(self) -> str:
        input = self.in_names[0]
        output = self.outs_to_keep[0]
        weight, _ = self.named_params[0]
        if self.use_bias:
            bias, _ = self.named_params[1]
        else:
            bias = None

        H, W = input.sizes[-2:]
        KH, KW = weight.sizes[-2:]

        input_h_index = f"({self.stride[0]}*h + kh - {self.padding[0]})"
        input_w_index = f"({self.stride[1]}*w + kw - {self.padding[1]})"
        output_h_constraint = f"h in 0:1+({H}+{2*self.padding[0]}-{KH})/{self.stride[0]}"
        output_w_constraint = f"w in 0:1+({W}+{2*self.padding[1]}-{KW})/{self.stride[1]}"

        forward = (
                f"{output}(n, g, m, h, w) +=! {input}(n, g, c, max(min({input_h_index}, {H}-1), 0), max(min({input_w_index}, {W}-1), 0)) * {weight}(g, m, c, kh, kw) \n"
                f"                              * fmin(1.0, fmax(0.0, (1 + {input_h_index}) * ({H} - ({input_h_index}))))\n"  # Setting zero at the padding boundaries. 
                f"                              * fmin(1.0, fmax(0.0, (1 + {input_w_index}) * ({W} - ({input_w_index}))))\n"
                f"    where kh in 0:{KH}, kw in 0:{KW},\n"
                f"        {output_h_constraint}, {output_w_constraint}\n" +
                (f"{output}(n, g, m, h, w) = {output}(n, g, m, h, w) + {bias}(g, m)\n"
                 f"     where {output_h_constraint}, {output_w_constraint}" if self.use_bias else ''))

        return forward

    def back(self):
        type_name = DEFAULT_TYPE

        d_output_h_index = f"((h + {self.padding[0]})/{self.stride[0]} - kh)"
        d_output_w_index = f"((w + {self.padding[1]})/{self.stride[1]} - kw)"
        weight_h_index = f"{self.stride[0]}*kh + (h + {self.padding[0]}) - {self.stride[0]}*((h + {self.padding[0]})/{self.stride[0]})"
        weight_w_index = f"{self.stride[1]}*kw + (w + {self.padding[1]}) - {self.stride[1]}*((w + {self.padding[1]})/{self.stride[1]})"
        input_h_index = f"{self.stride[0]}*oh + kh - {self.padding[0]}"
        input_w_index = f"{self.stride[1]}*ow + kw - {self.padding[1]}"

        backward = (
                f"def Convolution({type_name}(N, G, C, H, W) input, \n"
                f"                         {type_name}(G, M, C, KH, KW) weight, \n"
                f"                         {type_name}(G, M) bias, \n"
                f"                         {type_name}(N, G, M, OH, OW) d_output) -> (d_input, d_weight, d_bias) {'{'}\n"
                f"    d_input(n, g, c, h, w) +=! d_output(n, g, m, {d_output_h_index}, {d_output_w_index}) * weight(g, m, c, {weight_h_index}, {weight_w_index})\n"
                f"                                      * fmin(1.0, fmax(0.0, (1 + {d_output_h_index}) * (OH - ({d_output_h_index}))))\n"  # Setting zero at the padding boundaries. 
                f"                                      * fmin(1.0, fmax(0.0, (1 + {d_output_w_index}) * (OW - ({d_output_w_index}))))\n"
                f"                                      * fmin(1.0, fmax(0.0, KH - ({weight_h_index})))\n"  # Setting zero at the padding boundaries. 
                f"                                      * fmin(1.0, fmax(0.0, KW - ({weight_w_index})))\n"
                f"        where h in 0:H, w in 0:W,\n"
                f"              kh in 0:(KH + {self.stride[0] - 1})/{self.stride[0]},\n"
                f"              kw in 0:(KW + {self.stride[1] - 1})/{self.stride[1]}\n"
                f"    d_weight(g, m, c, kh, kw) +=! d_output(n, g, m, oh, ow) * input(n, g, c, {input_h_index},  {input_w_index})\n"
                f"                                      * fmin(1.0, fmax(0.0, (1 + {input_h_index}) * (H - ({input_h_index}))))\n"  # Setting zero at the padding boundaries. 
                f"                                      * fmin(1.0, fmax(0.0, (1 + {input_w_index}) * (W - ({input_w_index}))))\n"
                f"        where kh in 0:KH, kw in 0:KW, oh in 0:OH, ow in 0:OW\n" +
                (f"    d_bias(g, m) +=! d_output(n, g, m, h, w)\n" if self.use_bias else
                 f"    d_bias(g, m) = 0 where exists bias(g, m)\n") +
                "}")
        return backward
