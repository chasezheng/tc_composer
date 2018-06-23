import functools
from typing import Callable, Type


class cached_property:
    """A property that is only computed once per instance and then replaces
    itself with an ordinary attribute. Deleting the attribute resets the
    property.
    Credit to Marcel Hellkamp, author of bottle.py.
    """

    def __init__(self, func: Callable):
        self.__doc__ = getattr(func, '__doc__')
        self.func = func

    def __get__(self, obj: object, cls: Type):
        if obj is None:
            return self
        value = obj.__dict__[self.func.__name__] = self.func(obj)
        return value


def memoize(f: Callable) -> Callable:
    cache = f.cache = {}

    @functools.wraps(f)
    def memoizer(*args, **kwargs):
        key = tuple(list(args) + sorted(kwargs.items()))
        if key not in cache:
            cache[key] = f(*args, **kwargs)
        return cache[key]

    return memoizer
