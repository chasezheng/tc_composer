from functools import lru_cache
from typing import Sequence

from .function_with_params import FunctionWithParams
from ..unique_name import TensorName, Size


class Sum(FunctionWithParams):
    __slots__ = '_num_ins', '_in_dim'

    def __init__(self,
                 num_ins: int = None,
                 in_dim: int = None,
                 entry_point: str = None):
        super(Sum, self).__init__(entry_point=entry_point)
        self._num_ins = num_ins
        self._in_dim = in_dim

    @lru_cache(maxsize=None)
    def def_components(self, in_names: Sequence[TensorName] = None):
        if in_names is None:
            assert self._num_ins is not None
            assert self._in_dim is not None
            sizes = tuple(Size() for _ in range(self._in_dim))
            in_names = tuple(TensorName(dim=self._in_dim, sizes=sizes, prefix='summant') for _ in range(self._num_ins))

        sizes = in_names[0].sizes
        output = TensorName(dim=len(sizes), sizes=sizes, prefix='summed')
        indices = ', '.join(output.indices)
        summation = ' + '.join(f'{n}({indices})' for n in in_names)
        existence = ', '.join(f'exists {n}({indices})' for n in in_names)
        body = (f"{output}({indices}) = {summation}\n"
                f"    where {existence}")
        return body, in_names, (output,), ()

    @property
    def named_params(self):
        return ()


class Concat(FunctionWithParams):
    __slots__ = '_num_ins', '_in_dim'

    def __init__(self, num_ins: int, in_dim: int, ):
        super(Concat, self).__init__()
        self._num_ins = num_ins
        self._in_dim = in_dim

    def def_components(self, in_names: Sequence[TensorName] = None):
        if in_names is None:
            assert self._num_ins is not None
            assert self._in_dim is not None

            sizes = tuple(Size() for _ in range(self._in_dim))
            concat_dim = self._in_dim - 1

            def new_sizes():
                return tuple(s if n != concat_dim else Size() for n, s in enumerate(sizes))

            in_names = tuple(
                TensorName(dim=self._in_dim, sizes=new_sizes(), prefix='input') for _ in range(self._num_ins))
        else:
            sizes = in_names[0]._sizes
            concat_dim = len(sizes) - 1

        concat_sizes = tuple(s if n != concat_dim else 'S' for n, s in enumerate(sizes))
        output = TensorName(dim=len(sizes), sizes=concat_sizes, prefix='stacked')

        def statement_yielder():
            indices_list = output.indices
            offsetted_indices = list(indices_list)

            upper_bound = sum(n._sizes[concat_dim] for n in in_names if isinstance(n._sizes[concat_dim], int))
            if upper_bound > 0:
                upper_bound = str(upper_bound) + ' + ' + ' + '.join(
                    str(n._sizes[concat_dim]) for n in in_names if not isinstance(n._sizes[concat_dim], int))
            else:
                upper_bound = ' + '.join(
                    str(n._sizes[concat_dim]) for n in in_names if not isinstance(n._sizes[concat_dim], int))

            yield (f"{output}({', '.join(indices_list)}) = {in_names[0]}(0,0,0)\n"
                   f"    where {indices_list[concat_dim]} in 0:{upper_bound}, {indices_list[0]} in 0:{output.sizes[0]}, {indices_list[1]} in 0:{output.sizes[1]}")

            lower_bound = '0'
            upper_bound = '0'
            for inp in in_names:
                upper_bound += '+' + str(inp._sizes[concat_dim])
                yield (
                    f"{output}({', '.join(indices_list)}) +=! {inp}({', '.join(offsetted_indices)})\n"
                    f"    where {indices_list[concat_dim]} in {lower_bound}:{upper_bound}")
                offsetted_indices[concat_dim] += f'-{inp.sizes[concat_dim]}'
                lower_bound += '+' + str(inp._sizes[concat_dim])

            yield (f"{output}({', '.join(indices_list)}) += 0.0\n"
                   f"    where {indices_list[concat_dim]} in 0:{upper_bound}, {indices_list[0]} in 0:{output.sizes[0]}, {indices_list[1]} in 0:{output.sizes[1]}")

        body = '\n'.join(statement_yielder())
        return body, in_names, (output,), ()

    @property
    def named_params(self):
        return ()
