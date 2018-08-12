from functools import lru_cache
from typing import Sequence
from typing import Tuple

from .function_with_params import FunctionWithParams
from ..settings import DEFAULT_TYPE
from ..unique_name import TensorName


class Convolution(FunctionWithParams):  # todo dilation, different dimensions
    __slots__ = 'in_channels', 'out_channels', 'kernel_size', 'padding', 'stride', 'groups', 'use_bias'

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: Tuple[int, int],
                 stride: Tuple[int, int] = (1, 1),
                 padding: Tuple[int, int] = (0, 0),
                 groups: int = 1,
                 bias: bool = True,
                 entry_point: str = None):
        super(Convolution, self).__init__(entry_point=entry_point)
        assert len(kernel_size) == len(padding) == len(stride), f"Found: len(kernel_size) == {len(kernel_size)}, " \
                                                                f"len(padding) == {len(padding)}, " \
                                                                f"len(stride) == {len(stride)}"

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
        if self.groups > 1:
            if self.use_bias:
                return TensorName.make_pair(sizes=(self.groups, self.out_channels, self.in_channels, *self.kernel_size),
                                            prefix='weight'), \
                       TensorName.make_pair(sizes=(self.groups, self.out_channels), prefix='bias')
            else:
                return TensorName.make_pair(sizes=(self.groups, self.out_channels, self.in_channels, *self.kernel_size),
                                            prefix='weight'),
        else:
            if self.use_bias:
                return TensorName.make_pair(sizes=(self.out_channels, self.in_channels, *self.kernel_size),
                                            prefix='weight'), \
                       TensorName.make_pair(sizes=(self.out_channels,), prefix='bias')
            else:
                return TensorName.make_pair(sizes=(self.out_channels, self.in_channels, *self.kernel_size),
                                            prefix='weight'),

    def def_components(self, in_names: Sequence[TensorName] = None):
        assert in_names is None or len(in_names) == 1
        if in_names is not None:
            input = in_names[0]
        else:
            if self.groups > 1:
                input = TensorName(dim=5, prefix='input', sizes=('B', self.groups, self.in_channels, 'H', 'W'))
            else:
                input = TensorName(dim=4, prefix='input', sizes=('B', self.in_channels, 'H', 'W'))

        if input.sizes[-2].num is not None:
            OH = (input.sizes[-2].num + 2*self.padding[0] + self.stride[0] - self.kernel_size[0]) // self.stride[0]
        else:
            OH = 'OH'
        if input.sizes[-1].num is not None:
            OW = (input.sizes[-1].num + 2*self.padding[1] + self.stride[1] - self.kernel_size[1]) // self.stride[1]
        else:
            OW = 'OW'

        if self.groups > 1:
            assert input.dim == 5, f"input=({input.arg})"
            input.sizes[1].num = self.groups
            input.sizes[2].num = self.in_channels
            output_sizes = (input.sizes[0], self.groups, self.out_channels, OH, OW)
        else:
            input.sizes[1].num = self.in_channels
            output_sizes = (input.sizes[0], self.out_channels, OH, OW)

        output = TensorName(dim=len(output_sizes), prefix='output', sizes=output_sizes)

        if self.use_bias:
            weight, _ = self.named_params[0]
            bias, _ = self.named_params[1]
        else:
            weight, _ = self.named_params[0]
            bias = None

        H, W = input.sizes[-2:]
        KH, KW = self.kernel_size

        input_h_index = f"h + kh"
        input_w_index = f"w + kw"

        if self.padding[0] > 0:
            input_h_index += f" - {self.padding[0]}"
        if self.padding[1] > 0:
            input_w_index += f" - {self.padding[1]}"
        if self.stride[0] > 1:
            input_h_index = f"{self.stride[0]}*" + input_h_index
            output_h_upper_bound = f"({H.add(2*self.padding[0] - KH + self.stride[0])})/{self.stride[0]}"
        else:
            output_h_upper_bound = f"{H.add(2*self.padding[0] - KH + self.stride[0])}"
        if self.stride[1] > 1:
            input_w_index = f"{self.stride[1]}*" + input_w_index
            output_w_upper_bound = f"({W.add(2*self.padding[1] - KW + self.stride[1])})/{self.stride[1]}"
        else:
            output_w_upper_bound = f"{W.add(2*self.padding[1] - KW + self.stride[1])}"

        output_h_upper_bound = output_sizes[-2] if isinstance(output_sizes[-2], int) else output_h_upper_bound
        output_w_upper_bound = output_sizes[-1] if isinstance(output_sizes[-1], int) else output_w_upper_bound

        output_h_constraint = f"h in 0:{output_h_upper_bound}"
        output_w_constraint = f"w in 0:{output_w_upper_bound}"

        input_h_mask_cond = f"(({input_h_index}) * ({H.sub(1)} - ({input_h_index})) >= 0)"
        input_w_mask_cond = f"(({input_w_index}) * ({W.sub(1)} - ({input_w_index})) >= 0)"
        conds = (input_h_mask_cond, input_w_mask_cond)
        masks = (c for c,p in zip(conds, self.padding) if p > 0)

        g_ = 'g, ' if self.groups > 1 else ''
        forward = (
                f"{output}(n, {g_}m, h, w) +=! " +
                (
                    ' && '.join(masks) + ' ?\n' if any(n > 0 for n in self.padding) else ''
                ) +
                f"    {input}(n, {g_}c, {input_h_index}, {input_w_index}) * {weight}({g_}m, c, kh, kw)" +
                (" : 0.0\n" if any(n > 0 for n in self.padding) else '\n') +
                f"        where kh in 0:{KH}, kw in 0:{KW}, {output_h_constraint}, {output_w_constraint}" +
                (f"\n"
                 f"{output}(n, {g_}m, h, w) = {output}(n, {g_}m, h, w) + {bias}({g_}m)" if self.use_bias else ''))

        return forward, (input,), (output,), ()

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
