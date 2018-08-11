from functools import lru_cache
from typing import Sequence

from .func.function_with_params import FunctionWithParams
from .unique_name import TensorName


class AlexNet(FunctionWithParams):
    __slots__ = ()

    @property
    @lru_cache(maxsize=None)
    def named_params(self):
        return TensorName.make_pair(sizes=(64, 3, 11, 11), prefix='weight'), TensorName.make_pair(sizes=(64,),prefix='bias'), \
               TensorName.make_pair(sizes=(192, 64, 5, 5), prefix='weight'), TensorName.make_pair(sizes=(192,),prefix='bias'), \
               TensorName.make_pair(sizes=(384, 192, 3, 3), prefix='weight'), TensorName.make_pair(sizes=(384,),prefix='bias'), \
               TensorName.make_pair(sizes=(256, 384, 3, 3), prefix='weight'), TensorName.make_pair(sizes=(256,),prefix='bias'), \
               TensorName.make_pair(sizes=(256, 256, 3, 3), prefix='weight'), TensorName.make_pair(sizes=(256,),prefix='bias'), \
               TensorName.make_pair(sizes=(4096, 9216), prefix='weight'), TensorName.make_pair(sizes=(4096,),prefix='bias'), \
               TensorName.make_pair(sizes=(4096, 4096), prefix='weight'), TensorName.make_pair(sizes=(4096,),prefix='bias'), \
               TensorName.make_pair(sizes=(1000, 4096), prefix='weight'), TensorName.make_pair(sizes=(1000,),prefix='bias')

    def def_components(self, in_names: Sequence[TensorName] = None):
        if in_names is not None:
            assert len(in_names) == 1
            assert tuple(s.num for s in in_names[0].sizes[1:]) == (3, 227, 227), f'{tuple(s.num for s in in_names[0].sizes[1:])}'
            input = in_names[0]
        else:
            input = TensorName(dim=4, sizes=('batches', 3, 227, 227), prefix='input')
        batches = input.sizes[0]
        output = TensorName(dim=4, sizes=(batches, 64, 56, 56), prefix='output')
        output1 = TensorName(dim=4, sizes=(batches, 64, 27, 27), prefix='output')
        output2 = TensorName(dim=4, sizes=(batches, 192, 27, 27), prefix='output')
        output3 = TensorName(dim=4, sizes=(batches, 192, 13, 13), prefix='output')
        output4 = TensorName(dim=4, sizes=(batches, 384, 13, 13), prefix='output')
        output5 = TensorName(dim=4, sizes=(batches, 256, 13, 13), prefix='output')
        output6 = TensorName(dim=4, sizes=(batches, 256, 13, 13), prefix='output')
        output7 = TensorName(dim=4, sizes=(batches, 256, 6, 6), prefix='output')
        output8 = TensorName(dim=2, sizes=(batches, 4096), prefix='output')
        output9 = TensorName(dim=2, sizes=(batches, 4096), prefix='output')
        output10 = TensorName(dim=2, sizes=(batches, 1000), prefix='output')

        weight, bias, \
        weight1, bias1, \
        weight2, bias2, \
        weight3, bias3, \
        weight4, bias4, \
        weight5, bias5, \
        weight6, bias6, \
        weight7, bias7 = tuple(n for n, _ in self.named_params)

        body = (
            f"""{output}(n, m, h, w) +=! {input}(n, c, max(min(4*h + kh - 2, 226), 0), max(min(4*w + kw - 2, 226), 0)) * {weight}(m, c, kh, kw) 
                              * fmin(1.0, fmax(0.0, (1 + 4*h + kh - 2) * (227 - (4*h + kh - 2))))
                              * fmin(1.0, fmax(0.0, (1 + 4*w + kw - 2) * (227 - (4*w + kw - 2))))
    where kh in 0:11, kw in 0:11, h in 0:56, w in 0:56

{output1}(b, c, h, w) max=! fmax({output}(b, c, h*2 + kh, w*2 + kw) + {bias}(c), 0)
    where kh in 0:3, kw in 0:3

{output2}(n, m, h, w) +=! {output1}(n, c, max(min(h + kh - 2, 26), 0), max(min(w + kw - 2, 26), 0)) * {weight1}(m, c, kh, kw) 
                              * fmin(1.0, fmax(0.0, (1 + h + kh - 2) * (27 - (h + kh - 2))))
                              * fmin(1.0, fmax(0.0, (1 + w + kw - 2) * (27 - (w + kw - 2))))
    where kh in 0:5, kw in 0:5, h in 0:27, w in 0:27

{output3}(b, c, h, w) max=! fmax({output2}(b, c, h*2 + kh, w*2 + kw) + {bias1}(c), 0)
    where kh in 0:3, kw in 0:3

{output4}(n, m, h, w) +=! {output3}(n, c, max(min(h + kh - 1, 12), 0), max(min(w + kw - 1, 12), 0)) * {weight2}(m, c, kh, kw) 
                              * fmin(1.0, fmax(0.0, (1 + h + kh - 1) * (13 - (h + kh - 1))))
                              * fmin(1.0, fmax(0.0, (1 + w + kw - 1) * (13 - (w + kw - 1))))
    where kh in 0:3, kw in 0:3, h in 0:13, w in 0:13

{output5}(n, m, h, w) +=! fmax({output4}(n, c, max(min(h + kh - 1, 12), 0), max(min(w + kw - 1, 12), 0)) + {bias2}(c), 0) * {weight3}(m, c, kh, kw) 
                              * fmin(1.0, fmax(0.0, (1 + h + kh - 1) * (13 - (h + kh - 1))))
                              * fmin(1.0, fmax(0.0, (1 + w + kw - 1) * (13 - (w + kw - 1))))
    where kh in 0:3, kw in 0:3, h in 0:13, w in 0:13

{output6}(n, m, h, w) +=! fmax({output5}(n, c, max(min(h + kh - 1, 12), 0), max(min(w + kw - 1, 12), 0)) + {bias3}(c), 0) * {weight4}(m, c, kh, kw) 
                              * fmin(1.0, fmax(0.0, (1 + h + kh - 1) * (13 - (h + kh - 1))))
                              * fmin(1.0, fmax(0.0, (1 + w + kw - 1) * (13 - (w + kw - 1))))
    where kh in 0:3, kw in 0:3, h in 0:13, w in 0:13

{output7}(b, c, h, w) max=! fmax({output6}(b, c, h*2 + kh, w*2 + kw) + {bias4}(c), 0)
    where kh in 0:3, kw in 0:3

{output8}(b, n) +=! {output7}(b, i/{output7.sizes[-2].num*output7.sizes[-1].num}, (i%{output7.sizes[-2].num*output7.sizes[-1].num})/{output7.sizes[-1]}, i%{output7.sizes[-1]}) * {weight5}(n, i)
    where i in 0:{output7.sizes[-3].num*output7.sizes[-2].num*output7.sizes[-1].num}, n in 0:{weight5.sizes[0]}

{output9}(b, n) +=! fmax({output8}(b, i) + {bias5}(i), 0) * {weight6}(n, i)

{output10}(b, n) +=! fmax({output9}(b, i) + {bias6}(i), 0) * {weight7}(n, i)

{output10}(b, n) = {output10}(b, n) + {bias7}(n)""")

        return body, (input,), (output10,), (output, output1, output2, output3, output4, output5, output6, output7, output8, output9)


class AlexNetConv(FunctionWithParams):
    __slots__ = ()

    @property
    @lru_cache(maxsize=None)
    def named_params(self):
        return TensorName.make_pair(sizes=(64, 3, 11, 11), prefix='weight'), TensorName.make_pair(sizes=(64,),prefix='bias'), \
               TensorName.make_pair(sizes=(192, 64, 5, 5), prefix='weight'), TensorName.make_pair(sizes=(192,),prefix='bias'), \
               TensorName.make_pair(sizes=(384, 192, 3, 3), prefix='weight'), TensorName.make_pair(sizes=(384,),prefix='bias'), \
               TensorName.make_pair(sizes=(256, 384, 3, 3), prefix='weight'), TensorName.make_pair(sizes=(256,),prefix='bias'), \
               TensorName.make_pair(sizes=(256, 256, 3, 3), prefix='weight'), TensorName.make_pair(sizes=(256,),prefix='bias')

    def def_components(self, in_names: Sequence[TensorName] = None):
        if in_names is not None:
            assert len(in_names) == 1
            assert tuple(s.num for s in in_names[0].sizes[1:]) == (3, 227, 227), f'{tuple(s.num for s in in_names[0].sizes[1:])}'
            input = in_names[0]
        else:
            input = TensorName(dim=4, sizes=('batches', 3, 227, 227), prefix='input')
        batches = input.sizes[0]
        output = TensorName(dim=4, sizes=(batches, 64, 56, 56), prefix='output')
        output1 = TensorName(dim=4, sizes=(batches, 64, 27, 27), prefix='output')
        output2 = TensorName(dim=4, sizes=(batches, 192, 27, 27), prefix='output')
        output3 = TensorName(dim=4, sizes=(batches, 192, 13, 13), prefix='output')
        output4 = TensorName(dim=4, sizes=(batches, 384, 13, 13), prefix='output')
        output5 = TensorName(dim=4, sizes=(batches, 256, 13, 13), prefix='output')
        output6 = TensorName(dim=4, sizes=(batches, 256, 13, 13), prefix='output')
        output7 = TensorName(dim=4, sizes=(batches, 256, 6, 6), prefix='output')

        weight, bias, \
        weight1, bias1, \
        weight2, bias2, \
        weight3, bias3, \
        weight4, bias4 = tuple(n for n, _ in self.named_params)

        body = (
            f"""{output}(n, m, h, w) +=! {input}(n, c, max(min(4*h + kh - 2, 226), 0), max(min(4*w + kw - 2, 226), 0)) * {weight}(m, c, kh, kw) 
                              * fmin(1.0, fmax(0.0, (1 + 4*h + kh - 2) * (227 - (4*h + kh - 2))))
                              * fmin(1.0, fmax(0.0, (1 + 4*w + kw - 2) * (227 - (4*w + kw - 2))))
    where kh in 0:11, kw in 0:11, h in 0:56, w in 0:56

{output1}(b, c, h, w) max=! fmax({output}(b, c, h*2 + kh, w*2 + kw) + {bias}(c), 0)
    where kh in 0:3, kw in 0:3

{output2}(n, m, h, w) +=! {output1}(n, c, max(min(h + kh - 2, 26), 0), max(min(w + kw - 2, 26), 0)) * {weight1}(m, c, kh, kw) 
                              * fmin(1.0, fmax(0.0, (1 + h + kh - 2) * (27 - (h + kh - 2))))
                              * fmin(1.0, fmax(0.0, (1 + w + kw - 2) * (27 - (w + kw - 2))))
    where kh in 0:5, kw in 0:5, h in 0:27, w in 0:27

{output3}(b, c, h, w) max=! fmax({output2}(b, c, h*2 + kh, w*2 + kw) + {bias1}(c), 0)
    where kh in 0:3, kw in 0:3

{output4}(n, m, h, w) +=! {output3}(n, c, max(min(h + kh - 1, 12), 0), max(min(w + kw - 1, 12), 0)) * {weight2}(m, c, kh, kw) 
                              * fmin(1.0, fmax(0.0, (1 + h + kh - 1) * (13 - (h + kh - 1))))
                              * fmin(1.0, fmax(0.0, (1 + w + kw - 1) * (13 - (w + kw - 1))))
    where kh in 0:3, kw in 0:3, h in 0:13, w in 0:13

{output5}(n, m, h, w) +=! fmax({output4}(n, c, max(min(h + kh - 1, 12), 0), max(min(w + kw - 1, 12), 0)) + {bias2}(c), 0) * {weight3}(m, c, kh, kw) 
                              * fmin(1.0, fmax(0.0, (1 + h + kh - 1) * (13 - (h + kh - 1))))
                              * fmin(1.0, fmax(0.0, (1 + w + kw - 1) * (13 - (w + kw - 1))))
    where kh in 0:3, kw in 0:3, h in 0:13, w in 0:13

{output6}(n, m, h, w) +=! fmax({output5}(n, c, max(min(h + kh - 1, 12), 0), max(min(w + kw - 1, 12), 0)) + {bias3}(c), 0) * {weight4}(m, c, kh, kw) 
                              * fmin(1.0, fmax(0.0, (1 + h + kh - 1) * (13 - (h + kh - 1))))
                              * fmin(1.0, fmax(0.0, (1 + w + kw - 1) * (13 - (w + kw - 1))))
    where kh in 0:3, kw in 0:3, h in 0:13, w in 0:13

{output7}(b, c, h, w) max=! fmax({output6}(b, c, h*2 + kh, w*2 + kw) + {bias4}(c), 0)
    where kh in 0:3, kw in 0:3""")

        return body, (input,), (output7,), (output, output1, output2, output3, output4, output5, output6)


class AlexNetFeedForward(FunctionWithParams):
    __slots__ = ()

    @property
    @lru_cache(maxsize=None)
    def named_params(self):
        return TensorName.make_pair(sizes=(4096, 9216), prefix='weight'), TensorName.make_pair(sizes=(4096,),prefix='bias'), \
               TensorName.make_pair(sizes=(4096, 4096), prefix='weight'), TensorName.make_pair(sizes=(4096,),prefix='bias'), \
               TensorName.make_pair(sizes=(1000, 4096), prefix='weight'), TensorName.make_pair(sizes=(1000,),prefix='bias')

    def def_components(self, in_names: Sequence[TensorName] = None):
        if in_names is not None:
            assert len(in_names) == 1
            assert tuple(s.num for s in in_names[0].sizes[1:]) == (256, 6, 6), f'{tuple(s.num for s in in_names[0].sizes[1:])}'
            input = in_names[0]
        else:
            input = TensorName(dim=4, sizes=('batches', 256, 6, 6), prefix='input')
        batches = input.sizes[0]
        output8 = TensorName(dim=2, sizes=(batches, 4096), prefix='output')
        output9 = TensorName(dim=2, sizes=(batches, 4096), prefix='output')
        output10 = TensorName(dim=2, sizes=(batches, 1000), prefix='output')

        weight5, bias5, \
        weight6, bias6, \
        weight7, bias7 = tuple(n for n, _ in self.named_params)

        body = (
            f"""
{output8}(b, n) +=! {input}(b, i/{input.sizes[-2].num*input.sizes[-1].num}, (i%{input.sizes[-2].num*input.sizes[-1].num})/{input.sizes[-1]}, i%{input.sizes[-1]}) * {weight5}(n, i)
    where i in 0:{input.sizes[-3].num*input.sizes[-2].num*input.sizes[-1].num}, n in 0:{weight5.sizes[0]}

{output9}(b, n) +=! fmax({output8}(b, i) + {bias5}(i), 0) * {weight6}(n, i)

{output10}(b, n) +=! fmax({output9}(b, i) + {bias6}(i), 0) * {weight7}(n, i)

{output10}(b, n) = {output10}(b, n) + {bias7}(n)""")

        return body, (input,), (output10,), (output8, output9)
