from .function_with_params import FunctionWithParams
from ..unique_name import TensorName, UniqueName


class Sum(FunctionWithParams):
    __slots__ = ()

    def __init__(self,
                 num_ins: int,
                 in_dim: int,
                 entry_point: str = None):
        sizes = tuple(UniqueName() for _ in range(in_dim))
        in_names = tuple(TensorName(dim=in_dim, sizes=sizes, prefix='summant') for _ in range(num_ins))
        outs_to_keep = [TensorName(dim=in_dim, sizes=sizes, prefix='summed')]
        super(Sum, self).__init__(in_names=in_names, outs_to_keep=outs_to_keep,
                                  entry_point=entry_point)

    @property
    def def_body(self):
        if len(self.in_names) == 0:
            raise Exception("There isn't any input.")

        indices_list = ','.join('i' + str(i) for i in range(len(self.in_names[0].sizes)))
        output, = self.outs_to_keep

        summation = ' + '.join(f'{n}({indices_list})' for n in self.in_names)
        existence = ', '.join(f'exists {n}({indices_list})' for n in self.in_names)
        return (f"{output}({indices_list}) = {summation}\n"
                f"    where {existence}")

    @property
    def named_params(self):
        return ()


class Concat(FunctionWithParams):
    __slots__ = 'concat_dim',

    def __init__(self, num_ins: int, in_dim: int, concat_dim=0, entry_point: str = None):
        sizes = tuple(UniqueName() for _ in range(in_dim))
        in_names = tuple(TensorName(dim=in_dim, sizes=sizes, prefix='input') for _ in range(num_ins))
        outs_to_keep = [TensorName(dim=in_dim, sizes=sizes, prefix='concated')]
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
                if isinstance(inp.sizes[self.concat_dim], int):
                    upper_bound = inp.sizes[self.concat_dim]*len(self.in_names)
                else:
                    upper_bound = f'{inp.sizes[self.concat_dim]}*{len(self.in_names)}'
                yield (f"{output}({','.join(indices_list)}) +=! {inp}({','.join(offsetted_indices)})\n"
                       f"    where {indices_list[self.concat_dim]} in 0:{upper_bound}, exists {inp}({','.join(offsetted_indices)})")
                offsetted_indices[self.concat_dim] += f' - {inp.sizes[self.concat_dim]}'

        return '\n'.join(statement_yielder())

    @property
    def named_params(self):
        return ()
