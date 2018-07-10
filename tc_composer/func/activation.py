from .function_with_params import FunctionWithParams
from ..settings import DEFAULT_TYPE
from ..unique_name import TensorName


class Activation(FunctionWithParams):
    __slots__ = '_func',

    def __init__(self, in_n: int, func: str):
        in_name = TensorName(dim=2, sizes=('batches', in_n), prefix=f'input')
        super(Activation, self).__init__(
            in_names=[in_name],
            outs_to_keep=[TensorName(dim=2, sizes=in_name.sizes, prefix=f'output')],
            entry_point=func.capitalize()
        )
        assert func.lower() in ('tanh', 'sigmoid', 'relu')
        self._func = func.lower()

    @property
    def named_params(self):
        return ()

    @property
    def def_body(self) -> str:
        input, = self.in_names
        output, = self.outs_to_keep

        if self._func == 'tanh':
            return f"{output}(b, i) = tanh({input}(b, i))"
        elif self._func == 'sigmoid':
            return f"{output}(b, i) = 1 / (1 + exp(-{input}(b, i)))"
        elif self._func == 'relu':
            return f"{output}(b, i) = fmax({input}(b, i), 0)"
        else:
            raise Exception(f"Unexpected func {self._func}")


class Softmax(FunctionWithParams):
    """Perform softmax on dim=-1
    """
    __slots__ = ()

    def __init__(self, in_n: int):
        in_name = TensorName(dim=2, sizes=('batch_size', in_n), prefix='input')
        super(Softmax, self).__init__(
            in_names=[in_name],
            outs_to_keep=[TensorName(dim=2, sizes=in_name.sizes, prefix='input')],
            outs_to_discard=[TensorName(dim=1, sizes=in_name.sizes[:1], prefix='max_val'),
                             TensorName(dim=2, sizes=in_name.sizes, prefix='translated'),
                             TensorName(dim=1, sizes=in_name.sizes[:1], prefix='l1norm')])

    @property
    def named_params(self):
        return ()

    @property
    def def_body(self) -> str:
        input, = self.in_names
        max_val, translated, l1norm, output = *self.outs_to_discard, *self.outs_to_keep

        return (f"{max_val}(n) max=! {input}(n, d)\n"
                f"{translated}(n, d) = exp({input}(n, d) - {max_val}(n))\n"
                f"{l1norm}(n) +=! {translated}(n, d)\n"
                f"{output}(n, d) = {translated}(n, d) / {l1norm}(n)")
