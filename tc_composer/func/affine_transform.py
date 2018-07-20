from functools import lru_cache
from typing import Sequence

from .function_with_params import FunctionWithParams
from ..unique_name import TensorName


class AffineTransform(FunctionWithParams):
    __slots__ = '_use_bias', 'in_n', 'out_n'

    def __init__(self, in_n: int, out_n: int, bias: bool = True):
        super(AffineTransform, self).__init__(entry_point='Affine' if bias else 'Linear')

        self.in_n = in_n
        self.out_n = out_n
        self._use_bias = bias

    @property
    @lru_cache(maxsize=None)
    def named_params(self):
        if self._use_bias:
            return TensorName.make_pair(sizes=(self.out_n, self.in_n), prefix='weight'), \
                   TensorName.make_pair(sizes=(self.out_n,), prefix='bias')
        else:
            return TensorName.make_pair(sizes=(self.out_n, self.in_n), prefix='weight'),

    def def_components(self, in_names: Sequence[TensorName] = None):
        if in_names is None:
            input = TensorName(dim=2, sizes=('batches', self.in_n), prefix='input')
        else:
            assert len(in_names) == 1
            input = in_names[0]
            assert len(input.sizes) == 2
        input.sizes[1].num = self.in_n
        output = TensorName(dim=2, sizes=(input.sizes[0], self.out_n), prefix='output')

        if self._use_bias:
            weight, bias = tuple(name for name, _ in self.named_params)
            body = (f"{output}(b, n) +=! {input}(b, i) * {weight}(n, i)\n"
                    f"{output}(b, n) = {output}(b, n) + {bias}(n)")
        else:
            weight, = tuple(name for name, _ in self.named_params)
            body = f"{output}(b, n) +=! {input}(b, i) * {weight}(n, i)"

        return body, (input,), (output,), ()


class FusedAffineTransform(FunctionWithParams):
    __slots__ = 'in_n', 'hiddens', 'activations'

    def __init__(self, in_n: int,
                 hiddens: Sequence[int],
                 activations: Sequence[str]):
        super(FusedAffineTransform, self).__init__()
        assert set(activations).issubset({'tanh', 'relu', 'sigmoid'})
        self.activations = activations
        self.in_n = in_n
        self.hiddens = hiddens

    @property
    @lru_cache(maxsize=None)
    def named_params(self):
        def yielder():
            ins = (self.in_n,) + self.hiddens[:-1]

            for i, o in zip(ins, self.hiddens):
                yield TensorName.make_pair(sizes=(o, i), prefix='weight')
                yield TensorName.make_pair(sizes=(o,), prefix='bias')

        return tuple(yielder())

    @lru_cache(maxsize=None)
    def def_components(self, in_names: Sequence[TensorName] = None):
        if in_names is None:
            input = TensorName(dim=2, sizes=('batches', self.in_n), prefix='input')
        else:
            assert len(in_names) == 1
            input = in_names[0]
        input.sizes[1].num = self.in_n

        def output_yielder():
            batch = input.sizes[0]
            for h in self.hiddens:
                yield TensorName(dim=2, sizes=(batch, h), prefix='output')

        param_names = tuple(n for n, _ in self.named_params)
        weight_names = tuple(n for k, n in enumerate(param_names) if k % 2 == 0)
        bias_names = tuple(n for k, n in enumerate(param_names) if k % 2 == 1)

        out_names = tuple(output_yielder())

        def pt_wise_operated():
            for output, bias, activation in zip(out_names, bias_names, self.activations):
                operated = f'({output}(b, i) + {bias}(i))'
                if activation == 'tanh':
                    operated = f"tanh({operated})"
                elif activation == 'sigmoid':
                    operated = f"1 / (1 + exp(-{operated}))"
                elif activation == 'relu':
                    operated = f"fmax({operated}, 0)"
                yield operated

        operated = tuple(pt_wise_operated())

        def statement_yielder():
            yield f"{out_names[0]}(b, n) +=! {input}(b, i) * {weight_names[0]}(n, i)"
            for output, inp, weight in zip(out_names[1:], operated, weight_names[1:]):
                yield f"{output}(b, n) +=! {inp} * {weight}(n, i)"

            yield (f"{out_names[-1]}(b, n) = {operated[-1].replace('i)', 'n)')}")

        return '\n'.join(statement_yielder()), (input,), out_names[-1:], out_names[:-1]
