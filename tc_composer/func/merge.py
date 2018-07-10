from .function_with_params import FunctionWithParams
from ..unique_name import TensorName, UniqueName

# todo test these
class Sum(FunctionWithParams):
    __slots__ = ()

    def __init__(self,
                 num_ins: int = None,
                 in_dim: int = None,
                 entry_point: str = None):
        if num_ins is not None and in_dim is not None:
            assert num_ins > 0  #todo error msg
            assert in_dim >= 0
            sizes = tuple(UniqueName() for _ in range(in_dim))
            in_names = tuple(TensorName(dim=in_dim, sizes=sizes, prefix='summant') for _ in range(num_ins))
            outs_to_keep = [TensorName(dim=in_dim, sizes=sizes, prefix='summed')]
        else:
            assert num_ins is None  #todo error msg
            assert in_dim is None
            in_names = ()
            # Dimension of a tensor_name can be reset when needed.
            outs_to_keep = [TensorName(dim=0, prefix='summed')]

        super(Sum, self).__init__(in_names=in_names, outs_to_keep=outs_to_keep,
                                  entry_point=entry_point)

    @property
    def def_body(self):
        if len(self.in_names) == 0:
            raise Exception("There isn't any input.")

        indices_list = ','.join('i' + str(i) for i in range(len(self.in_names[0].sizes)))
        output, = self.outs_to_keep

        summation = ' + '.join(f'{n}({indices_list})' for n in self.in_names)
        existence = ', '.join(f'exists {n}' for n in self.in_names)
        return (f"{output}({indices_list}) = {summation}\n"
                f"    where {existence}")

    @property
    def named_params(self):
        return ()


class Concat(FunctionWithParams):
    __slots__ = 'concat_dim',

    def __init__(self, concat_dim=0, num_ins: int = None, in_dim: int = None, entry_point: str = None):
        if num_ins is not None and in_dim is not None:
            assert num_ins > 0  #todo error msg
            assert in_dim >= 0
            sizes = tuple(UniqueName() for _ in range(in_dim))
            in_names = tuple(TensorName(dim=in_dim, sizes=sizes, prefix='input') for _ in range(num_ins))
            outs_to_keep = [TensorName(dim=in_dim, sizes=sizes, prefix='concated')]
        else:
            assert num_ins is None  #todo error msg
            assert in_dim is None
            in_names = ()
            # Dimension of a tensor_name can be reset when needed.
            outs_to_keep = [TensorName(dim=0, prefix='concated')]

        super(Concat, self).__init__(in_names=in_names, outs_to_keep=outs_to_keep,
                                     entry_point=entry_point)
        self.concat_dim = concat_dim

    @property
    def def_body(self):
        if len(self.in_names) == 0:
            raise Exception("There isn't any input.")

        def statement_yielder():
            indices_list = tuple('i' + str(i) for i in range(len(self.in_names[0].sizes)))
            offsetted_indices = list(indices_list)
            output, = self.outs_to_keep
            for inp in self.in_names:
                yield (f"{output}({','.join(indices_list)}) = {inp}({','.join(offsetted_indices)})\n"
                       f"   where exists {inp}({offsetted_indices})")
                offsetted_indices[self.concat_dim] += f' - {inp.sizes[self.concat_dim]}'

        return '\n'.join(statement_yielder())

    @property
    def named_params(self):
        return ()