from .base import Layer


class Activation(Layer):
    __slots__ = 'func', 'layer'

    def __init__(self, func: str, layer: Layer, name: str = None):
        super(Activation, self).__init__(in_n=layer.out_n, out_n=layer.out_n, name=name)
        assert func in ('tanh', 'sigmoid',)  # todo more
        self.func = func
        self.layer = layer

    @property
    def lang(self) -> str:
        return (f"def {self.id}(float(batch_size, in_n) input) -> (output) {'{'}\n"
                f"    output(b, n) = {self.func}(I(b, n))\n"
                "}")

    def tc_call_args(self, *args, **kwargs):
        return args, kwargs
