from torch import nn

from .decorrelation import Decorrelation
from .option_vectorizer import Vectorizer


class Evaluator(nn.Module):
    __slots__ = ()

    def __init__(self):
        super(Evaluator, self).__init__()
        self.guesser = nn.Sequential(
            nn.Linear(in_features=Vectorizer.LEN, out_features=10 * Vectorizer.LEN),
            nn.Dropout(),
            nn.Tanh(),
            nn.Linear(in_features=10 * Vectorizer.LEN, out_features=32),
            Decorrelation(),
            nn.Linear(in_features=32, out_features=1))

    def forward(self, *input):
        return self.guesser(*input)
