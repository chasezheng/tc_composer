from functools import lru_cache
from typing import Sequence

from .function_with_params import FunctionWithParams
from ..unique_name import TensorName


class Activation(FunctionWithParams):
    __slots__ = '_func', '_input_dim'

    def __init__(self, func: str, input_dim: int = None):
        super(Activation, self).__init__(entry_point=func.capitalize())
        assert func.lower() in ('tanh', 'sigmoid', 'relu')
        self._func = func.lower()
        self._input_dim = input_dim

    @property
    def named_params(self):
        return ()

    @lru_cache(maxsize=None)
    def def_components(self, in_names: Sequence[TensorName] = None):
        if in_names is None:
            assert self._input_dim is not None, "We don't know the dimension of input."
            input = TensorName(dim=self._input_dim, prefix='input')
        else:
            assert len(in_names) == 1
            input = in_names[0]

        output = TensorName(dim=input.dim, sizes=input.sizes, prefix='output')
        indices = ', '.join(output.indices)

        if self._func == 'tanh':
            body = f"{output}({indices}) = tanh({input}({indices}))"
        elif self._func == 'sigmoid':
            body = f"{output}({indices}) = 1 / (1 + exp(-{input}({indices})))"
        elif self._func == 'relu':
            body = f"{output}({indices}) = fmax({input}({indices}), 0)"
        else:
            raise Exception(f"Unexpected func {self._func}")

        return body, (input,), (output,), ()


class Softmax(FunctionWithParams):
    __slots__ = '_aggregation_dim', '_input_dim'

    def __init__(self,
                 input_dim: int = None,
                 aggregation_dim: int = -1,
                 entry_point: str = None):
        super(Softmax, self).__init__(entry_point=entry_point)
        self._input_dim = input_dim
        self._aggregation_dim = aggregation_dim

    @property
    def named_params(self):
        return ()

    @lru_cache(maxsize=None)
    def def_components(self, in_names: Sequence[TensorName] = None):
        if in_names is None:
            assert self._input_dim is not None, "We don't know the dimension of input."
            input = TensorName(dim=self._input_dim, prefix='input')
        else:
            assert len(in_names) == 1
            input = in_names[0]

        dim = self._aggregation_dim % input.dim
        reduced_sizes = tuple(s for n, s in enumerate(input.sizes) if n != dim)
        max_val, translated, l1norm, output = TensorName(dim=input.dim - 1, sizes=reduced_sizes, prefix='max_val'), \
                                              TensorName(dim=input.dim, sizes=input.sizes, prefix='translated'), \
                                              TensorName(dim=input.dim - 1, sizes=reduced_sizes, prefix='l1norm'), \
                                              TensorName(dim=input.dim, sizes=input.sizes, prefix='output')
        indices = ', '.join(i for i in output.indices)
        reduced_indices = ', '.join(s for n, s in enumerate(output.indices) if n != dim)
        body = (f"{max_val}({reduced_indices}) max=! {input}({indices})\n"
                f"{translated}({indices}) = exp({input}({indices}) - {max_val}({reduced_indices}))\n"
                f"{l1norm}({reduced_indices}) +=! {translated}({indices})\n"
                f"{output}({indices}) = {translated}({indices}) / {l1norm}({reduced_indices})")

        return body, (input,), (output,), (max_val, translated, l1norm)
