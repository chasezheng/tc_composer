import tensor_comprehensions as tc

from .modules import Vectorizer


class OptionResult:
    __slots__ = 'option', 'speed', 'exc', 'compile_time'

    NOT_VIABLE = float('inf')

    def __init__(self, option: tc.MappingOptions, speed: float = None, exc: BaseException = None):
        self.option: tc.MappingOptions = option
        self.speed = speed
        self.exc = exc
        self.compile_time = self.NOT_VIABLE

    def __getstate__(self):
        return Vectorizer.parse_option_str(str(self.option)), self.speed, self.exc, self.compile_time

    def __eq__(self, other):
        if not isinstance(other, type(self)):
            return False
        return (str(self.option), self.speed, self.exc) == (str(other.option), other.speed, other.exc)

    def __hash__(self):
        return self.__getstate__().__hash__()

    def __setstate__(self, state):
        opt_attr, speed, exc, compile_time = state
        self.option = Vectorizer.from_attr_to_opt(opt_attr)
        self.speed = speed
        self.exc = exc
        self.compile_time = compile_time

    @staticmethod
    def make_naive():
        return OptionResult(tc.MappingOptions('naive'))

    def to_tensor(self):
        return Vectorizer.from_mapping_options(self.option)