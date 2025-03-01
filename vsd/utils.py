"""Utilities for variational search distributions."""

import math
from itertools import batched
from typing import Iterable, NewType, Optional, Sequence

import numpy as np
import torch
from torch import Tensor

SequenceArray = NewType("SequenceArray", Sequence[str])
SequenceTensor = NewType("SequenceTensor", torch.IntTensor)


def batch_indices(
    n: int, batchsize: int, seed: Optional[int] = None
) -> Iterable[np.ndarray]:
    rnd = np.random.RandomState(seed)
    while True:
        rinds = rnd.permutation(n)
        for b in batched(rinds, batchsize):
            yield np.array(b)


#
# Various Sequential NN components
#


class Transpose(torch.nn.Module):

    def __init__(self, dim0=-1, dim1=-2) -> None:
        super().__init__()
        self.dim0 = dim0
        self.dim1 = dim1

    def forward(self, X: Tensor) -> Tensor:
        return X.transpose(dim0=self.dim0, dim1=self.dim1)


class Max(torch.nn.Module):

    def __init__(self, dim: int = -1) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, X: Tensor) -> Tensor:
        return X.max(self.dim)[0]


class Average(torch.nn.Module):

    def __init__(self, dim: int = -1) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, X: Tensor) -> Tensor:
        return X.mean(self.dim)


class RemovePadding(torch.nn.Module):
    """Remove padding on the last dimension of X."""

    def __init__(self, padding_size: int) -> None:
        super().__init__()
        self.psize = padding_size

    def forward(self, X: Tensor) -> Tensor:
        return X[..., self.psize : -self.psize]


class PositionalEncoding(torch.nn.Module):
    """Batch-first positional encoding"""

    def __init__(self, emb_size: int, maxlen: int = 5000):
        super(PositionalEncoding, self).__init__()
        den = torch.exp(
            -torch.arange(0, emb_size, 2) * math.log(10000) / emb_size
        )
        pos = torch.arange(0, maxlen).reshape(maxlen, 1)
        pos_embedding = torch.zeros((maxlen, emb_size))
        pos_embedding[:, 0::2] = torch.sin(pos * den)
        pos_embedding[:, 1::2] = torch.cos(pos * den)
        pos_embedding = pos_embedding.unsqueeze(0)  # batch first

        self.register_buffer("pos_embedding", pos_embedding)

    def forward(
        self, token_embedding: Tensor, pos_ind: int = 0
    ) -> torch.Tensor:
        if token_embedding.ndim < 2:
            inds = torch.tensor([pos_ind])
        else:
            inds = torch.arange(pos_ind, pos_ind + token_embedding.size(1))
        return token_embedding + self.pos_embedding[:, inds, :]
