from torch import Tensor
from torch import nn

from .decorrelation import Decorrelation
from .option_vectorizer import Vectorizer


class Proposer(nn.Module):
    __slots__ = '_proposer', '_num_proposals', '_decorrelation'

    def __init__(self, in_features: int, num_proposals: int = 10):
        self._num_proposals = num_proposals
        self._proposer = nn.Sequential(
            nn.Linear(in_features=in_features, out_features=64*num_proposals),
            nn.Dropout(),
            nn.Tanh(),
            nn.Linear(in_features=64*num_proposals, out_features=32*num_proposals),
            nn.Tanh(),
            Decorrelation(),
            nn.Linear(in_features=32*num_proposals, out_features=Vectorizer.LEN*num_proposals))
        self._decorrelation = Decorrelation()

    def forward(self, input: Tensor) -> Tensor:
        out = self._proposer(input)
        for t in out.view(self._num_proposals, Vectorizer.LEN):
            self._decorrelation(t)
        return out


