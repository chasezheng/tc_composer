import typing
from collections import Counter


class UniqueId:
    __slots__ = '__id',

    __NAMES: typing.Counter[str] = Counter()

    def __init__(self, name: str = None):
        name = name or type(self).__name__[:1]

        if name in UniqueId.__NAMES:
            UniqueId.__NAMES[name] += 1
            name = name + str(UniqueId.__NAMES[name])

        assert name not in UniqueId.__NAMES
        self.__id: str = name
        UniqueId.__NAMES[name] += 1

    @property
    def id(self):
        return self.__id

    def __str__(self):
        return self.__id.__str__()