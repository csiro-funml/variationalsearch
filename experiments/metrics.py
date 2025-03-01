"""Metrics for measuring the quality of discovered sequences."""

from itertools import product
from typing import Optional

import numpy as np
from polyleven import levenshtein
from torch import Tensor

from vsd.utils import SequenceArray


class Performance:
    """Multi-round performance computation."""

    def __init__(
        self,
        S_init: Optional[SequenceArray] = None,
    ):
        self.uniqueS = set() if S_init is None else set(S_init)
        self.cumperf = 0

    def __call__(self, y: np.ndarray | Tensor, S: SequenceArray) -> float:
        # filter non-unique sequences
        yu = y[novel_inds(S, self.uniqueS)]

        # Update set of seen sequences
        self.uniqueS = self.uniqueS.union(set(S))

        # Update score
        self.cumperf += yu.sum()
        return self.cumperf


class FalseDiscovery:
    """Multi-round false discovery calculation."""

    def __init__(
        self,
        best_f: float,
        S_init: Optional[SequenceArray] = None,
    ):
        self.best_f = best_f
        self.uniqueS = set() if S_init is None else set(S_init)
        self.cumreg = 0
        self.count = 0

    def __call__(self, y: np.ndarray | Tensor, S: SequenceArray) -> float:
        # filter non-unique sequences
        iu = novel_inds(S, self.uniqueS)
        yu = y[iu]

        # Update set of seen sequences
        self.uniqueS = self.uniqueS.union(set(S))

        # Update score
        self.cumreg += len(y) - len(yu)  # regret resampling
        self.cumreg += sum(yu < self.best_f)  # regret low scores
        self.count += len(y)
        return self.cumreg


class Precision(FalseDiscovery):
    """Multi-round precision calculation."""

    def __call__(self, y: np.ndarray | Tensor, S: SequenceArray) -> float:
        fd = super().__call__(y=y, S=S)
        return 1.0 - fd / self.count


class Recall(FalseDiscovery):
    """Multi-round recall calculation."""

    def __init__(
        self,
        best_f: float,
        npositives: int,
        S_init: Optional[SequenceArray] = None,
    ):
        super().__init__(best_f=best_f, S_init=S_init)
        self.npositives = npositives

    def __call__(self, y: np.ndarray | Tensor, S: SequenceArray) -> float:
        fd = super().__call__(y=y, S=S)
        return (self.count - fd) / self.npositives


def diversity(S: SequenceArray) -> float:
    num_seqs = len(S)
    if num_seqs <= 1:
        return 0.0
    total_dist = 0
    for si, sj in product(S, S):
        if si == sj:
            continue
        total_dist += levenshtein(si, sj)
    return total_dist / (num_seqs * (num_seqs - 1))


def novelty(
    S: SequenceArray, Strain: SequenceArray, median: bool = True
) -> float:
    num_seqs = len(S)
    all_novelty = np.full(num_seqs, fill_value=1e9)
    for i, si in enumerate(S):
        for sj in Strain:
            dist = levenshtein(si, sj)
            all_novelty[i] = min(dist, all_novelty[i])
    if median:
        return np.median(all_novelty)
    return all_novelty.mean()


def innovation(
    S: SequenceArray,
    Strain: SequenceArray,
    y: Tensor | np.ndarray,
    ytrain: Tensor | np.ndarray,
):
    icand = np.argmax(y)
    itrain = np.argmax(ytrain)
    return gap(S[icand], Strain[itrain])


def performance(y: Tensor, S: SequenceArray, Strain: SequenceArray) -> int:
    """Per round performance computation"""
    return (y[novel_inds(S, Strain)]).sum()


def maxfitness(y: Tensor | np.ndarray) -> float:
    return y.max()


def utility(y: Tensor | np.ndarray, best_f: float) -> int:
    return (y > best_f).sum()


def simple_regret(
    y: Tensor | np.ndarray, y_train: Tensor | np.ndarray, y_max: float
) -> float:
    return y_max - max(y.max(), y_train.max())


def precision(
    y: np.ndarray | Tensor,
    S: SequenceArray,
    Strain: SequenceArray,
    best_f: float,
):
    """Per-round precision computation"""
    return utility(y[novel_inds(S, Strain)], best_f) / len(y)


def fdr(
    y: np.ndarray | Tensor,
    S: SequenceArray,
    Strain: SequenceArray,
    best_f: float,
) -> float:
    return 1 - precision(y, S, Strain, best_f)


def gap(si: str, sj: str) -> int:
    return levenshtein(si, sj)


def novel_inds(S: SequenceArray, Strain: SequenceArray | set) -> np.ndarray:
    Su, iu = np.unique(S, return_index=True)
    iu = [i for i, si in zip(iu, Su) if si not in Strain]
    return np.array(iu, dtype=int)
