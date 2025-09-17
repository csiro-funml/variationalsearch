"""Utilities for variational search distributions.

Provides small helpers and lightweight neural-network building blocks used
throughout the VSD codebase.
"""

import math
from collections import deque
from itertools import batched
from typing import Iterable, NewType, Optional, Sequence, Deque, Tuple

import numpy as np
import torch
from torch import Tensor
from botorch.utils.multi_objective.pareto import is_non_dominated
from botorch.optim.stopping import StoppingCriterion


SequenceArray = NewType("SequenceArray", Sequence[str])
SequenceTensor = NewType("SequenceTensor", torch.IntTensor)


def batch_indices(
    n: int, batchsize: int, seed: Optional[int] = None
) -> Iterable[np.ndarray]:
    """Yield random mini-batch indices forever.

    Parameters
    ----------
    n : int
        Total number of items to index from ``0`` to ``n-1``.
    batchsize : int
        Size of each batch (last batch per epoch may be smaller if ``n`` is not
        divisible).
    seed : int, optional
        Seed for the internal NumPy RNG; ``None`` uses a random seed.

    Yields
    ------
    numpy.ndarray
        One array of indices per batch. Iterates indefinitely, shuffling each
        epoch.
    """
    rnd = np.random.RandomState(seed)
    while True:
        rinds = rnd.permutation(n)
        for b in batched(rinds, batchsize):
            yield np.array(b)


def batch_indices_val(
    n: int, batchsize: int, seed: Optional[int] = None, val_prop: float = 0.1
) -> Iterable[Tuple[np.ndarray, np.ndarray]]:
    """Yield random mini-batch indices forever with a fixed validation split.

    Returns tuples ``(batch, val)`` indefinitely. A single validation subset
    of size ``round(n * val_prop)`` is sampled once and kept fixed, while the
    remaining training indices are reshuffled into mini-batches every epoch.

    Parameters
    ----------
    n : int
        Total number of items to index from ``0`` to ``n-1``.
    batchsize : int
        Size of each batch (last batch per epoch may be smaller if ``n`` is not
        divisible).
    seed : int, optional
        Seed for the internal NumPy RNG; ``None`` uses a random seed.
    val_prop : float, default: 0.1
        Proportion of the dataset to reserve as a validation set. Must satisfy
        ``0 < val_prop < 1``.

    Yields
    ------
    (numpy.ndarray, numpy.ndarray)
        A tuple per mini-batch. The first array contains training batch indices.
        The second array contains the fixed validation indices used for this run.
    """
    if batchsize < 1:
        raise ValueError("batchsize must be >= 1")
    if not (0 < val_prop < 1):
        raise ValueError("val_prop must be in (0, 1)")
    if n <= batchsize:
        raise ValueError("n must be greater than batchsize")

    # Split train/validation
    n_val = int(round(float(n) * float(val_prop)))
    n_val = max(1, min(n - 1, n_val))
    rnd = np.random.RandomState(seed)
    rinds = rnd.permutation(n)
    val = rinds[:n_val]
    trn = rinds[n_val:]

    epochs = 0
    while True:
        # Shuffle every epoch
        strn = rnd.permutation(trn)

        # Yield mini-batches from remaining indices, paired with the fixed val
        for b in batched(strn, batchsize):
            yield np.array(b), val

        epochs += 1


def inv_softplus(y: Tensor) -> Tensor:
    """Numerically stable inverse of softplus.

    Computes ``x`` such that ``softplus(x) == y``.

    Parameters
    ----------
    y : torch.Tensor
        Target softplus values.

    Returns
    -------
    torch.Tensor
        Inverse-transformed tensor with the same shape as ``y``.
    """
    return y + torch.log(-torch.expm1(-y))


def is_non_dominated_strict(Y: Tensor) -> Tensor:
    """Compute non-dominated mask assuming maximization.

    Parameters
    ----------
    Y : torch.Tensor
        Objective values of shape ``(n, m)`` for ``n`` points and ``m``
        objectives.

    Returns
    -------
    torch.Tensor
        Boolean mask of shape ``(n,)`` where ``True`` marks non-dominated
        points. Duplicates are not removed.
    """
    return is_non_dominated(Y, maximize=True, deduplicate=False)


#
# Various Sequential NN components
#


class Transpose(torch.nn.Module):
    """Transpose two dimensions of a tensor.

    Parameters
    ----------
    dim0 : int, default: -1
        First dimension to swap.
    dim1 : int, default: -2
        Second dimension to swap.
    """

    def __init__(self, dim0=-1, dim1=-2) -> None:
        super().__init__()
        self.dim0 = dim0
        self.dim1 = dim1

    def forward(self, X: Tensor) -> Tensor:
        """Return ``X`` with ``dim0`` and ``dim1`` swapped."""
        return X.transpose(dim0=self.dim0, dim1=self.dim1)


class Max(torch.nn.Module):
    """Reduce by maximum along a dimension.

    Parameters
    ----------
    dim : int, default: -1
        Dimension to reduce.
    """

    def __init__(self, dim: int = -1) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, X: Tensor) -> Tensor:
        """Return ``X.max(dim)`` (values only)."""
        return X.max(self.dim)[0]


class Average(torch.nn.Module):
    """Mean reduction along a dimension.

    Parameters
    ----------
    dim : int, default: -1
        Dimension to reduce.
    """

    def __init__(self, dim: int = -1) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, X: Tensor) -> Tensor:
        """Return ``X.mean(dim)``."""
        return X.mean(self.dim)


class RemovePadding(torch.nn.Module):
    """Slice off symmetric padding from the last dimension.

    Parameters
    ----------
    padding_size : int
        Number of elements to remove from both start and end of the last
        dimension.

    Returns
    -------
    torch.Tensor
        If input is ``X``, output is ``X[..., p:-p]`` with ``p=padding_size``.
    """

    def __init__(self, padding_size: int) -> None:
        super().__init__()
        self.psize = padding_size

    def forward(self, X: Tensor) -> Tensor:
        return X[..., self.psize : -self.psize]


class PositionalEncoding(torch.nn.Module):
    """Sinusoidal positional encoding (batch-first).

    Precomputes a standard transformer-style sinusoidal table and adds it to
    token embeddings. Expects tensors with shape ``(B, L, D)``.

    Parameters
    ----------
    emb_size : int
        Embedding dimension ``D``.
    maxlen : int, default: 5000
        Maximum supported sequence length ``L``.
    """

    def __init__(self, emb_size: int, maxlen: int = 5000):
        super().__init__()
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
        """Add positional encodings to embeddings.

        Parameters
        ----------
        token_embedding : torch.Tensor
            Input embeddings of shape ``(B, L, D)`` or ``(L, D)``.
        pos_ind : int, default: 0
            Starting position index for the sequence window.

        Returns
        -------
        torch.Tensor
            Positional-augmented embeddings with the same shape as input.
        """
        if token_embedding.ndim < 2:
            inds = torch.tensor([pos_ind])
        else:
            inds = torch.arange(pos_ind, pos_ind + token_embedding.size(1))
        return token_embedding + self.pos_embedding[:, inds, :]


class Skip(torch.nn.Module):
    """Residual wrapper that adds input to module output.

    Parameters
    ----------
    nn : torch.nn.Module
        The module to wrap; must map ``x`` to a tensor broadcastable to ``x``.
    """

    def __init__(self, nn: torch.nn.Module):
        super().__init__()
        self.nn = nn

    def forward(self, x: Tensor) -> Tensor:
        return self.nn(x) + x


class CatNorm(torch.nn.Module):
    """LayerNorm inputs and then concatenate them.

    Parameters
    ----------
    normalization_shapes : Sequence[int]
        Feature sizes for per-input ``LayerNorm`` modules.
    """

    def __init__(self, normalization_shapes: Sequence[int]) -> None:
        super().__init__()
        self.norms = torch.nn.ModuleList(
            [torch.nn.LayerNorm(s) for s in normalization_shapes]
        )

    def forward(self, inputs: Sequence[Tensor]) -> Tensor:
        ninputs = [ln(x) for ln, x in zip(self.norms, inputs)]
        return torch.cat(ninputs, dim=-1)


class FuseNorm(torch.nn.Module):
    """LayerNorm inputs and fuse: ``norm(a) + w * norm(b)``.

    Parameters
    ----------
    add_shape : int
        Feature dimension for both inputs.
    alpha0 : float, default: 1.0
        Initial value for the learnable fusion weights ``w``.
    """

    def __init__(self, add_shape: int, alpha0: float = 1.0) -> None:
        super().__init__()
        self.norm_a = torch.nn.LayerNorm(add_shape)
        self.norm_b = torch.nn.LayerNorm(add_shape)
        self.w = torch.nn.Parameter(torch.ones([add_shape]) * alpha0)

    def forward(self, ab: Tuple[Tensor, Tensor]) -> Tensor:
        a, b = ab
        add = self.norm_a(a) + self.w * self.norm_b(b)
        return add


class FuseTwo(torch.nn.Module):
    """Fuse two branches: ``w * a(x) + (1 - w) * b(x)``.

    Parameters
    ----------
    a : torch.nn.Module
        First branch module.
    b : torch.nn.Module
        Second branch module.
    """

    def __init__(self, a: torch.nn.Module, b: torch.nn.Module) -> None:
        super().__init__()
        self.a = a
        self.b = b
        self.w = torch.nn.Parameter(torch.zeros(1))

    def forward(self, X: Tensor) -> Tensor:
        w = torch.sigmoid(self.w)
        fuse = w * self.a(X) + (1 - w) * self.b(X)
        return fuse


class ApplyAndCat(torch.nn.Module):
    """Apply all modules to input and concatenate the results."""

    def __init__(self, input_layers: Sequence[torch.nn.Module]):
        super().__init__()
        self.layers = input_layers

    def forward(self, X: Tensor) -> Tensor:
        res = [l(X) for l in self.layers]
        return torch.cat(res, dim=-1)


#
#  Stopping Criteria
#


class SEPlateauStopping(StoppingCriterion):
    """Variance-aware early stopping for minimization.

    Maintains a sliding window of the last ``n_window`` scalar values. At each
    check (every ``check_every`` calls to :meth:`evaluate` after ``miniter``),
    tests whether the reduction in the window mean exceeds ``k * SE(window)``.
    Stops after ``patience`` consecutive non-improvements or when ``maxiter``
    (if provided) is reached.

    Parameters
    ----------
    n_window : int, default: 300
        Sliding window length. Must be at least 20.
    k : float, default: 0.1
        Required improvement threshold in units of the standard error. Smaller
        is more stringent.
    patience : int, default: 3
        Number of consecutive non-improvements before stopping.
    miniter : int, default: 500
        Minimum number of :meth:`evaluate` calls before applying the check.
    maxiter : int, optional
        Hard cap on total :meth:`evaluate` calls.
    check_every : int, default: 50
        Perform the stopping check every this many :meth:`evaluate` calls.
    """

    def __init__(
        self,
        n_window: int = 300,
        k: float = 0.1,
        patience: int = 3,
        miniter: int = 500,
        maxiter: Optional[int] = 100000,
        check_every: int = 50,
    ) -> None:
        if n_window < 20:
            raise ValueError("`window` should be ≥ 20.")
        self.n_window = n_window
        self.k = k
        self.patience = patience
        self.maxiter = maxiter
        self.min_steps = miniter
        self.check_every = check_every

        self._buf: Deque[float] = deque([], maxlen=n_window)
        self._prev_mean: Optional[float] = None
        self._noimp: int = 0
        self._steps: int = 0

    def reset(self) -> None:
        self._buf.clear()
        self._prev_mean = None
        self._noimp = 0
        self._steps = 0

    @staticmethod
    def _mean_se(buf: Deque[float]) -> tuple[float, float]:
        n = len(buf)
        m = sum(buf) / n
        v = sum((x - m) ** 2 for x in buf) / max(1, n - 1)  # unbiased
        se = math.sqrt(v / n)
        return m, se

    def evaluate(self, fvals: Tensor) -> bool:
        """Evaluate stopping condition for a scalar objective.

        Parameters
        ----------
        fvals : torch.Tensor
            A scalar tensor (minimization) from the latest iteration.

        Returns
        -------
        bool
            ``True`` if training should stop, ``False`` otherwise.
        """
        if fvals.numel() != 1:
            raise ValueError("SEPlateauStopping expects a scalar tensor.")
        self._steps += 1

        # Hard cap on iterations
        if self.maxiter is not None and self._steps >= self.maxiter:
            return True

        v = float(fvals.detach().item())
        self._buf.append(v)

        # Need a full window and respect warmup/cadence before testing
        if (
            self._steps < self.min_steps
            or (self._steps % self.check_every) != 0
            or len(self._buf) < self.n_window
        ):
            return False

        mean, se = self._mean_se(self._buf)
        if self._prev_mean is None:
            self._prev_mean = mean
            return False

        improvement = self._prev_mean - mean  # minimization
        if improvement <= self.k * se:
            self._noimp += 1
        else:
            self._noimp = 0

        self._prev_mean = mean
        return self._noimp >= self.patience
