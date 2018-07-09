from .function_with_params import FunctionWithParams
from ..settings import DEFAULT_TYPE
from ..unique_name import TensorName


class Activation(FunctionWithParams):
    __slots__ = '_func',

    def __init__(self, func: str):
        assert func.lower() in ('tanh', 'sigmoid', 'relu')
        super(Activation, self).__init__(
            in_names=[TensorName(dim=1, type=DEFAULT_TYPE, prefix=f'input')],
            outs_to_keep=[TensorName(dim=1, type=DEFAULT_TYPE, prefix=f'output')],
            entry_point=func
        )
        self._func = func.lower()

    @property
    def named_params(self):
        return ()

    @property
    def def_body(self) -> str:
        input, = self.in_names
        output, = self.outs_to_keep

        if self._func == 'tanh':
            return f"{output}(i) = tanh({input}(i))"
        elif self._func == 'sigmoid':
            return f"{output}(i) = 1 / (1 + exp(-{input}(i)))"
        elif self._func == 'relu':
            return f"{output}(i) = fmax({input}(i), 0)"
        else:
            raise Exception(f"Unexpected func {self._func}")


class Softmax(FunctionWithParams):
    """Perform softmax on dim=-1
    """
    __slots__ = ()

    def __init__(self):
        super(Softmax, self).__init__(
            in_names=[TensorName(dim=2, type=DEFAULT_TYPE, prefix='input')],
            outs_to_keep=[TensorName(dim=2, type=DEFAULT_TYPE, prefix='output')],
            outs_to_discard=[TensorName(dim=1, type=DEFAULT_TYPE, prefix='max_val'),
                             TensorName(dim=2, type=DEFAULT_TYPE, prefix='translated'),
                             TensorName(dim=1, type=DEFAULT_TYPE, prefix='l1norm')]
        )

    @property
    def named_params(self):
        return ()

    @property
    def def_body(self) -> str:
        input, = self.in_names
        max_val, translated, l1norm, output = self.outs_to_keep

        return (f"{max_val}(n) max=! {input}(n, d)\n"
                f"{translated}(n, d) = exp({input}(n, d) - {max_val}(n))\n"
                f"{l1norm}(n) +=! {translated}(n, d)\n"
                f"{output}(n, d) = {translated}(n, d) / {l1norm}(n)")
