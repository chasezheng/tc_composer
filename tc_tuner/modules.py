import ast
import logging
import re
import traceback
from collections import abc
from collections import deque
from functools import lru_cache
from itertools import repeat
from typing import Iterable, Tuple, TypeVar, Sequence, MutableSequence, Mapping, Any, MutableSet

import tensor_comprehensions as tc
import torch
from torch import nn, Tensor, optim
from torch.nn.functional import softmax

from .settings import get_configured_logger


class Module(torch.nn.Module):
    __slots__ = 'optimizers', 'loss'

    def __init__(self):
        super(Module, self).__init__()
        self.optimizers = []
        self.loss = deque()

    @property
    @lru_cache(maxsize=None)
    def logger(self) -> logging.Logger:
        return get_configured_logger(type(self).__name__)

    def _iterate_loss(self) -> Iterable[deque]:
        yield self.loss
        for _, sub in self.named_sub():
            if len(sub.loss) > 0:
                yield sub.loss
            yield from sub._iterate_loss()

    def apply_grad(self):
        if self.training:
            loss = 0
            for queue in self._iterate_loss():
                loss = loss + sum(queue)
                queue.clear()
            if loss is not 0:
                loss.backward()
            self.step()
        self.zero_grad()

    def named_sub(self) -> Iterable[Tuple[str, 'Module']]:
        """Immediate sub-modules.
        """
        for name, module in self.named_children():
            # self.children yields immediate sub nn.Modules
            if module is not self and isinstance(module, Module):
                yield name, module

    def step(self):
        for op in self.optimizers:
            op.step()
        for _, m in self.named_sub():
            m.step()


class Decorrelation(Module):
    __slots__ = '_2nd_moment', 'in_n'

    def __init__(self, in_n: int,
                 coef: float = 1e-4):
        super(Decorrelation, self).__init__()
        self.in_n = in_n
        self._2nd_moment: Tensor = 0
        self.coef = coef

    def forward(self, input: Tensor):
        input = input.view(-1, self.in_n)
        if self.training:
            self._2nd_moment = self._2nd_moment + torch.matmul(input.t(), input)
        if len(input) == 1:
            return input[0]
        return input

    def _iterate_loss(self):
        try:
            d = self.scaled_abs_det(self._2nd_moment)
        except RuntimeError:
            self.logger.error(traceback.format_exc())
            pass
        else:
            self.loss.append(d.log().neg().mul(self.coef))
        finally:
            self._2nd_moment: Tensor = 0

        yield from super(Decorrelation, self)._iterate_loss()

    @staticmethod
    def scaled_abs_det(m: Tensor) -> Tensor:
        with torch.set_grad_enabled(False):
            U, s, V = torch.svd(m.inverse())
        s = V.t().matmul(m.matmul(U)).diag()

        return s.div(s.mean()).prod(0)


class Proposer(Module):
    __slots__ = '_proposer', '_num_proposals', '_decorrelation'

    def __init__(self,
                 in_features: int,
                 num_proposals: int = 10,
                 start_option: tc.MappingOptions = None):
        super(Proposer, self).__init__()
        start_option = start_option or tc.MappingOptions('naive')

        self._num_proposals = num_proposals
        self._proposer = nn.Sequential(
            nn.Linear(in_features=in_features, out_features=128),
            nn.Dropout(),
            nn.Tanh(),
            nn.Linear(in_features=128, out_features=32),
            nn.Tanh(),
            Decorrelation(32),
            nn.Linear(in_features=32, out_features=num_proposals * Vectorizer.LEN),
            Decorrelation(Vectorizer.LEN)
        )
        self.optimizers.append(optim.RMSprop(self.parameters(), lr=1e-3))

        initial_bias = Vectorizer.from_mapping_options(start_option)
        initial_bias = torch.cat(tuple(repeat(initial_bias, num_proposals)))
        tuple(self._proposer.parameters())[-1].data = initial_bias

    def forward(self, inp: Tensor) -> Tensor:
        return self._proposer(inp).view(-1, Vectorizer.LEN)


class Evaluator(Module):
    __slots__ = ()

    def __init__(self):
        super(Evaluator, self).__init__()
        self.guesser = nn.Sequential(
            nn.Linear(in_features=Vectorizer.LEN, out_features=10 * Vectorizer.LEN),
            nn.Dropout(),
            nn.Tanh(),
            nn.Linear(in_features=10 * Vectorizer.LEN, out_features=32),
            Decorrelation(32),
            nn.Linear(in_features=32, out_features=1))

        self.optimizers.append(optim.RMSprop(self.parameters()))

    def forward(self, *inp):
        return self.guesser(*inp)


T = TypeVar('T')


class Vectorizer:
    __slots__ = ()

    CONFIG = (
        # attribute_name, attribute_type, vectorized_length
        ('fixParametersBeforeScheduling', (True, False), 2),
        ('intraTileScheduleFusionStrategy', ('Max', 'Min', 'Preserve3Coincident'), 3),
        ('outerScheduleFusionStrategy', ('Max', 'Min', 'Preserve3Coincident'), 3),
        ('mapToBlocks', int, 3),
        ('mapToThreads', int, 3),
        ('matchLibraryCalls', (True, False), 2),
        ('tile', int, 10),
        ('unroll', int, 1),
        # ('maxSharedMemory', int, 1),
        ('unrollCopyShared', (True, False), 2),
        ('usePrivateMemory', (True, False), 2),
        ('useReadOnlyCache', (True, False), 2),
        ('useSharedMemory', (True, False), 2)
    )
    OptionAttr = Sequence[Tuple[str, Any]]

    LOGGER = get_configured_logger('Vectorizer')
    LEN = sum(c for a, b, c in CONFIG)

    @classmethod
    def to_bool(cls, t: Tensor) -> bool:
        t = t.view(-1)
        assert len(t) == 2

        return cls.to_class(t, (False, True))

    @classmethod
    def to_ints(cls, t: Tensor) -> Sequence[int]:
        return t.pow(2).round().long().tolist()

    @classmethod
    def to_class(cls, t: Tensor, classes: Sequence[T]) -> T:
        idx = torch.multinomial(softmax(t, dim=-1), 1).item()
        mask = torch.zeros_like(t)
        mask[idx] = 1
        t.mul_(mask)
        return classes[idx]

    @classmethod
    def from_bool(cls, b: bool) -> Tensor:
        return cls.from_class(b, (True, False))

    @classmethod
    def from_ints(cls, seq: Sequence[int]) -> Tensor:
        assert all(i >= 0 for i in seq), f"seq = {seq}"
        return Tensor(seq).sqrt_()

    @classmethod
    def from_class(cls, this: T, classes: Sequence[T]) -> Tensor:
        out = torch.zeros(len(classes)).fill_(-1e10)
        out[classes.index(this)] = 1e10
        return out

    @staticmethod
    def remove_trailing_zeros(seq):
        if 0 not in seq:
            return seq
        return seq[:seq.index(0)]

    @classmethod
    def to_mapping_options(cls, t: Tensor) -> tc.MappingOptions:
        assert t.dim() == 1, f"t.shape = {t.shape}"
        option = tc.MappingOptions('naive')
        start = 0
        try:
            for attr, mytype, length in cls.CONFIG:
                if mytype is int:
                    if length == 1:
                        getattr(option, attr)(*cls.to_ints(t[start:start + length]))
                    else:
                        getattr(option, attr)(  # todo discontinuity
                            cls.remove_trailing_zeros(cls.to_ints(t[start:start + length])))
                else:
                    # assert isinstance(mytype, abc.Sequence), f"type(mytype) = {type(mytype)}"
                    getattr(option, attr)(cls.to_class(t[start:start + length], mytype))

                start += length
        except:
            cls.LOGGER.error(f"attr = {attr}; mytype = {mytype.__name__}; length = {length}.")
            raise

        return option

    @classmethod
    def from_mapping_options(cls, option: tc.MappingOptions) -> Tensor:
        return cls.from_attr_to_tensor(cls.parse_option_str(str(option)))

    @classmethod
    def from_attr_to_tensor(cls, option_attr: 'Vectorizer.OptionAttr') -> Tensor:
        attr_dict = dict(option_attr)

        def tensor_yielder():
            for attr, mytype, length in cls.CONFIG:
                if mytype is int:
                    out = cls.from_ints((attr_dict[attr],) if isinstance(attr_dict[attr], int) else attr_dict[attr])
                    if len(out) < length:
                        out = torch.cat((out, torch.zeros(length - len(out))))
                else:
                    # assert isinstance(mytype, abc.Sequence), f"type(mytype) = {type(mytype)}"
                    out = cls.from_class(attr_dict[attr], mytype)
                    # assert len(out) == length
                yield out

        return torch.cat(tuple(tensor_yielder()))

    @classmethod
    def from_attr_to_opt(cls, pairs: Sequence[Tuple[str, Any]], opt: tc.MappingOptions = None) -> tc.MappingOptions:
        opt = opt or tc.MappingOptions('naive')
        for k, v in pairs:
            getattr(opt, k)(v)
        return opt

    @classmethod
    def parse_option_str(cls, option_str: str) -> 'Vectorizer.OptionAttr':
        """Return a dictionary that maps attributes of an option to their values.
        """
        # Formatting
        option_str = option_str.replace('tiling {', 'tiling: [').replace('}\n  unroll', ']\n  unroll') \
            .replace(' sizes: ', ' ')
        option_str = option_str.replace('block {', 'block: [').replace('}\ngrid', ']\ngrid')
        option_str = option_str.replace('grid {', 'grid: [').replace('}\nuse_', ']\nuse_')
        option_str = option_str.replace(' x: ', '').replace(' y: ', '').replace('z: ', '')
        option_str = option_str.replace(' {', ': {')
        option_str = re.sub(r'\b', '"', option_str)
        option_str = option_str.replace('"\n', '",\n').replace('}', '},').replace(']\n', '],\n')
        option_str = option_str.replace('"false"', 'False').replace('"true"', 'True')
        mapping = ast.literal_eval('{' + option_str + '}')

        # Interpreting integers
        if 'max_shared_memory' in mapping:
            mapping['max_shared_memory'] = int(mapping['max_shared_memory'])
        mapping['generic_mapping_options']['unroll'] = int(mapping['generic_mapping_options']['unroll'])
        mapping['generic_mapping_options']['tiling'] = tuple(
            int(n) for n in mapping['generic_mapping_options']['tiling'])
        mapping['block'] = tuple(int(n) for n in mapping['block'])
        mapping['grid'] = tuple(int(n) for n in mapping['grid'])

        # Flattening nested dictionaries
        mapping.update(mapping['generic_mapping_options'])
        del mapping['generic_mapping_options']

        # Formatting key names, filtering keys, further flattening nested dictionaries
        def yielder():
            for k, v in mapping.items():
                k = re.sub(r'_.', lambda s: s.group(0).upper().replace('_', ''), k)
                if k == 'grid':
                    k = 'mapToBlocks'
                elif k == 'block':
                    k = 'mapToThreads'
                elif k == 'tiling':
                    k = 'tile'
                elif k in ('outerScheduleOptions', 'intraTileScheduleOptions'):
                    k = k.replace('Options', '') + 'FusionStrategy'
                    v = v['fusion_strategy']
                elif k == 'useReadonlyCache':
                    k = 'useReadOnlyCache'

                if k in ('allow_skewing', 'positive_orthant', 'tile_imperfectly_nested',
                         'allowSkewing', 'positiveOrthant', 'tileImperfectlyNested'):
                    pass
                else:
                    yield k, v

        return tuple(yielder())

    # Sanity checks
    for attr, mytype, length in CONFIG:
        assert length > 0, f"attr = {attr}; mytype = {mytype}; length = {length}"
        assert mytype is int or isinstance(mytype, abc.Sequence), f"attr = {attr}; mytype = {mytype}; length = {length}"
        assert isinstance(attr, str), f"attr = {attr}; mytype = {mytype}; length = {length}"

        del attr, mytype, length
