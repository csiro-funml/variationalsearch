"""Threshold adaptation strategies."""

from typing import Optional

import numpy as np
from scipy.stats.mstats import mquantiles
from torch import Tensor


class Threshold:

    static = True

    def __init__(self, best_f: float):
        self.best_f = best_f

    def __call__(self, y: Tensor | np.ndarray) -> float:
        return self.best_f

    def labels(self, y: Tensor | np.ndarray) -> Tensor | np.ndarray:
        ind = y >= self.best_f
        if isinstance(ind, Tensor):
            return ind.int()
        return ind.astype(int)


class MaxIncumbent(Threshold):

    static = False

    def __init__(self, eps: Optional[float] = None):
        self.eps = eps if eps is not None else 0.0
        self.best_f = -np.inf

    def __call__(self, y: Tensor | np.ndarray) -> float:
        best_f = max(y.max() + self.eps, self.best_f)
        self.best_f = best_f
        return best_f


class DiscreteMaxIncumbent(Threshold):

    def __init__(self, min_positives=1):
        self.min_positives = min_positives
        self.best_f = -np.inf

    def __call__(self, y: Tensor | np.ndarray) -> float:
        if isinstance(y, Tensor):
            y = y.detach().numpy()
        members, count = np.unique(y, return_counts=True)
        members, count = members[::-1], count[::-1]
        cumcount = np.cumsum(count)
        if cumcount[0] >= self.min_positives:
            best_f = members[0]
        else:
            best_f = members[cumcount >= self.min_positives][0]
        best_f = max(best_f, self.best_f)
        self.best_f = best_f
        return best_f


class QuantileThreshold(Threshold):

    static = False

    def __init__(self, percentile: float, min_positives: int = 1):
        self.percentile = percentile
        self.best_f = -np.inf
        self.min_positives = min_positives

    def __call__(self, y: Tensor | np.ndarray) -> float:
        if isinstance(y, Tensor):
            y = y.detach().numpy()
        best_f = float(mquantiles(y, prob=self.percentile).squeeze())
        best_f = max(best_f, self.best_f)
        # Check we have min_positives
        if sum(y >= best_f) < self.min_positives:
            ysort = np.sort(y)[::-1]
            best_f = ysort[self.min_positives]
        self.best_f = best_f
        return best_f


class AnnealedThreshold(QuantileThreshold):

    def __init__(self, percentile: float, eta: float, min_positives: int = 1):
        super().__init__(percentile=percentile, min_positives=min_positives)
        if eta < 0 or eta > 1:
            raise ValueError("eta must be between 0 and 1!")
        self.eta = eta
        self.first_call = True

    def __call__(self, y: Tensor | np.ndarray) -> float:
        if not self.first_call:
            self.percentile = self.percentile**self.eta
        self.first_call = False
        return super().__call__(y)


class BudgetAnnealedThreshold(AnnealedThreshold):
    """Compute the annealing schedule when we have a known round-budget."""

    def __init__(self, p0: float, pT: float, T: int, min_positives: int = 1):
        eta = 1 if T <= 1 else (np.log(pT) / np.log(p0)) ** (1 / (T - 1))
        super().__init__(percentile=p0, eta=eta, min_positives=min_positives)
