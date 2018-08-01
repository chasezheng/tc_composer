import tensor_comprehensions as tc

from .modules import Vectorizer


class OptionResult:
    __slots__ = 'option', 'speed', 'exc'

    NOT_VIABLE = float('inf')

    def __init__(self, option: tc.MappingOptions, speed: float = None, exc: BaseException = None):
        self.option: tc.MappingOptions = option
        self.speed = speed
        self.exc = exc

    def __getstate__(self):
        return Vectorizer.parse_option_str(str(self.option)), self.speed, self.exc

    def __hash__(self):
        return self.__getstate__().__hash__()

    def __setstate__(self, state):
        opt_attr, speed, exc = state
        self.option = Vectorizer.from_attr_to_opt(opt_attr)
        self.speed = speed
        self.exc = exc

    @staticmethod
    def make_naive():
        return OptionResult(tc.MappingOptions('naive'))
