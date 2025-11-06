"""Threshold adaptation strategies."""

from abc import abstractmethod, ABC
from typing import Optional

import numpy as np
import torch
from pymoo.util.dominator import Dominator
from pymoo.util.nds.non_dominated_sorting import rank_from_fronts
from pymoo.util.nds.fast_non_dominated_sort import fast_non_dominated_sort
from scipy.stats.mstats import mquantiles
from torch import Tensor

from vsd.utils import is_non_dominated_strict


class Labeller(ABC):
    """Abstract base class for labelling strategies."""

    static = True

    def __call__(self, y: Tensor | np.ndarray) -> Tensor | np.ndarray:
        """Return labels for ``y`` via :meth:`labels`.

        Parameters
        ----------
        y:
            Objective values to label.

        Returns
        -------
        Tensor or ndarray
            Binary labels specific to the implementation.
        """
        return self.labels(y)

    @abstractmethod
    def labels(self, y: Tensor | np.ndarray) -> Tensor | np.ndarray:
        """Return class labels for the provided objective values."""
        ...


#
# Probability of improvement based labellers -- thresholding.
#


class Threshold(Labeller):
    """Assign positives based on a scalar threshold."""

    static = True

    def __init__(self, best_f: float, update_on_call: bool = True):
        """Initialise a threshold-based labeller.

        Parameters
        ----------
        best_f:
            Initial threshold value that defines the positive class.
        update_on_call:
            If ``True``, :meth:`update` is invoked before creating labels.
        """
        self.best_f = best_f
        self.update_on_call = update_on_call

    def update(self, y: Tensor | np.ndarray) -> float:
        """Return the current threshold without modification.

        Parameters
        ----------
        y:
            Unused. Present for interface compatibility.

        Returns
        -------
        float
            The existing threshold value.
        """
        return self.best_f

    def labels(self, y: Tensor | np.ndarray) -> Tensor | np.ndarray:
        """Label points as positive when they exceed the current threshold.

        Parameters
        ----------
        y:
            Objective values to classify.

        Returns
        -------
        Tensor or ndarray
            Binary labels with value ``1`` for positives and ``0`` otherwise.
        """
        if self.update_on_call:
            self.update(y)
        mask = y >= self.best_f
        if isinstance(mask, Tensor):
            return mask.int()
        return mask.astype(int)


class MaxIncumbent(Threshold):
    """Update the threshold to the best observed value."""

    static = False

    def __init__(
        self, eps: Optional[float] = None, update_on_call: bool = True
    ):
        """Initialise the incumbent tracker.

        Parameters
        ----------
        eps:
            Small margin added to the current maximum before labelling.
        update_on_call:
            If ``True``, :meth:`update` is invoked before creating labels.
        """
        super().__init__(best_f=-np.inf, update_on_call=update_on_call)
        self.eps = eps if eps is not None else 0.0

    def update(self, y: Tensor | np.ndarray) -> float:
        """Track the best value seen so far and update the threshold.

        Parameters
        ----------
        y:
            Candidate objective values.

        Returns
        -------
        float
            Updated threshold corresponding to the incumbent.
        """
        best_f = max(y.max() + self.eps, self.best_f)
        self.best_f = best_f
        return best_f


class DiscreteMaxIncumbent(Threshold):
    """Choose the smallest value with the required number of positives."""

    def __init__(self, min_positives=1, update_on_call: bool = True):
        """Initialise the discrete threshold tracker.

        Parameters
        ----------
        min_positives:
            Minimum number of positive labels to maintain.
        update_on_call:
            If ``True``, :meth:`update` is invoked before creating labels.
        """
        super().__init__(best_f=-np.inf, update_on_call=update_on_call)
        self.min_positives = min_positives

    def update(self, y: Tensor | np.ndarray) -> float:
        """Update the threshold to satisfy the positive-count constraint.

        Parameters
        ----------
        y:
            Discrete objective values.

        Returns
        -------
        float
            Threshold ensuring at least ``min_positives`` positives.
        """
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
    """Set the threshold according to a running quantile of the data."""

    static = False

    def __init__(
        self,
        percentile: float,
        min_positives: int = 1,
        update_on_call: bool = True,
    ):
        """Initialise the quantile-based threshold.

        Parameters
        ----------
        percentile:
            Quantile in ``[0, 1]`` used to determine the threshold.
        min_positives:
            Minimum number of positives to keep when the quantile is too high.
        update_on_call:
            If ``True``, :meth:`update` is invoked before creating labels.
        """
        super().__init__(best_f=-np.inf, update_on_call=update_on_call)
        self.percentile = percentile
        self.min_positives = min_positives

    def update(self, y: Tensor | np.ndarray) -> float:
        """Update the threshold to the desired quantile.

        Ensures the minimum number of positives is preserved.

        Parameters
        ----------
        y:
            Objective values used to estimate the quantile.

        Returns
        -------
        float
            Updated threshold value.
        """
        if isinstance(y, Tensor):
            y = y.detach().cpu().numpy()
        best_f = float(mquantiles(y, prob=self.percentile).squeeze())
        best_f = max(best_f, self.best_f)
        # Check we have min_positives
        if sum(y >= best_f) < self.min_positives:
            ysort = np.sort(y)[::-1]
            best_f = ysort[self.min_positives - 1]
        self.best_f = best_f
        return best_f


class AnnealedThreshold(QuantileThreshold):
    """Progressively tighten the quantile threshold according to an annealing schedule."""

    def __init__(
        self,
        percentile: float,
        eta: float,
        min_positives: int = 1,
        update_on_call: bool = True,
    ):
        """Initialise the annealing scheme.

        Parameters
        ----------
        percentile:
            Starting quantile in ``[0, 1]``.
        eta:
            Exponent used to shrink the percentile after each update.
        min_positives:
            Minimum number of positives to maintain.
        update_on_call:
            If ``True``, :meth:`update` is invoked before creating labels.

        Raises
        ------
        ValueError
            If ``eta`` is not within ``[0, 1]``.
        """
        super().__init__(
            percentile=percentile,
            min_positives=min_positives,
            update_on_call=update_on_call,
        )
        if eta < 0 or eta > 1:
            raise ValueError("eta must be between 0 and 1!")
        self.eta = eta
        self.first_call = True

    def update(self, y: Tensor | np.ndarray) -> float:
        """Shrink the percentile after the first call and update the threshold.

        Parameters
        ----------
        y:
            Objective values passed to :class:`QuantileThreshold`.

        Returns
        -------
        float
            Updated threshold value.
        """
        if not self.first_call:
            self.percentile = self.percentile**self.eta
        self.first_call = False
        return super().update(y)


class BudgetAnnealedThreshold(AnnealedThreshold):
    """Compute the annealing schedule when we have a known round-budget."""

    def __init__(
        self,
        p0: float,
        pT: float,
        T: int,
        min_positives: int = 1,
        update_on_call: bool = True,
    ):
        """Initialise the schedule using a known evaluation budget.

        Parameters
        ----------
        p0:
            Starting percentile in ``[0, 1]``.
        pT:
            Target percentile reached on the final call.
        T:
            Total number of calls expected.
        min_positives:
            Minimum number of positives to maintain.
        update_on_call:
            If ``True``, :meth:`update` is invoked before creating labels.
        """
        eta = 1 if T <= 1 else (np.log(pT) / np.log(p0)) ** (1 / (T - 1))
        super().__init__(
            percentile=p0,
            eta=eta,
            min_positives=min_positives,
            update_on_call=update_on_call,
        )


#
# Probability of hypervolume improvement based labellers -- non dominance
#


class ParetoFront(Labeller):
    """Class labels based on membership to Pareto Set."""

    def __init__(self, jitter: Optional[float] = None) -> None:
        """Initialise the Pareto front labeller.

        Parameters
        ----------
        jitter:
            Optional random noise added to break ties during comparison.
        """
        super().__init__()
        self.jitter = jitter

    def labels(self, y: Tensor | np.ndarray) -> Tensor | np.ndarray:
        """Assign label ``1`` to non-dominated points.

        Parameters
        ----------
        y:
            Objective array to evaluate.

        Returns
        -------
        Tensor or ndarray
            Binary labels with positives corresponding to the Pareto set.

        Raises
        ------
        AssertionError
            If no point is classified as positive.
        """
        makenp = False
        if isinstance(y, np.ndarray):
            makenp = True
            y = torch.tensor(y)
        if self.jitter is not None:
            y = y + torch.rand_like(y) * self.jitter
        z = is_non_dominated_strict(y).int()
        assert z.sum() > 0
        if makenp:
            z = z.numpy()
        return z


class ParetoQuantile(ParetoFront):
    """Class labels according to the quantile of their Pareto rank.

    Parameters
    ----------
    percentile : float
        Percentile in ``[0, 1]`` of Pareto ranks to threshold. ``0`` results in all
        points being labelled positive, ``1`` results in just the Pareto set.
    epsilon : float, optional
        Relaxation added to the non-dominance criterion, ``A >= B + epsilon``.
    """

    static = False

    def __init__(
        self,
        percentile: float,
        jitter: Optional[float] = None,
        epsilon: Optional[float] = None,
    ) -> None:
        """Initialise the Pareto-rank quantile labeller.

        Parameters
        ----------
        percentile:
            Quantile in ``[0, 1]`` used to threshold Pareto ranks.
        jitter:
            Optional random noise added before ranking.
        epsilon:
            Relaxation factor for the dominance comparison.
        """
        super().__init__()
        self.percentile = self._check_percentile(percentile)
        self.jitter = jitter
        self.epsilon = epsilon

    def labels(self, y: Tensor | np.ndarray) -> Tensor | np.ndarray:
        """Label points whose Pareto rank falls within the target quantile.

        Parameters
        ----------
        y:
            Objective matrix where higher values are preferred.

        Returns
        -------
        Tensor or ndarray
            Binary labels with ``1`` for the selected quantile.
        """
        yn = y
        if isinstance(y, Tensor):
            yn = y.cpu().numpy()
        if self.jitter is not None:
            yn = yn + np.random.rand(*yn.shape) * self.jitter
        ranks = non_dominated_sorting(-yn, epsilon=self.epsilon)
        percentiles = 1 - ranks / max(ranks.max(), 1)
        z = (percentiles >= self.percentile).astype(int)
        assert z.sum() > 0
        if isinstance(y, Tensor):
            return torch.tensor(z, device=y.device, dtype=torch.int)
        return z

    @staticmethod
    def _check_percentile(percentile: float) -> float:
        """Validate that ``percentile`` lies in ``[0, 1]``."""
        if percentile > 1 or percentile < 0:
            raise ValueError("percentile must be in [0, 1].")
        return percentile


class ParetoAnnealed(ParetoQuantile):
    """Class labels according to an annealed quantile of their Pareto rank.

    Parameters
    ----------
    percentile : float
        Percentile in ``[0, 1]`` of Pareto ranks to threshold. ``0`` results in all
        points being labelled positive, ``1`` results in just the Pareto set.
    """

    def __init__(
        self,
        percentile: float,
        T: int,
        percentileT: float = 0.99,
        jitter: Optional[float] = None,
        epsilon: Optional[float] = None,
    ) -> None:
        """Initialise the annealing schedule over Pareto ranks.

        Parameters
        ----------
        percentile:
            Starting quantile used to select positives.
        T:
            Total number of labelling rounds.
        percentileT:
            Target quantile on the final call.
        jitter:
            Optional random perturbation added before ranking.
        epsilon:
            Relaxation factor for the dominance comparison.
        """
        super().__init__(percentile=percentile, epsilon=epsilon, jitter=jitter)
        self.t = 0
        self.T = T
        p0 = self.percentile
        pT = self._check_percentile(percentileT)
        self.eta = 1 if T <= 1 else (np.log(pT) / np.log(p0)) ** (1 / (T - 1))

    def labels(self, y: Tensor | np.ndarray) -> Tensor | np.ndarray:
        """Update the percentile according to the schedule and label points.

        Parameters
        ----------
        y:
            Objective matrix where higher values are preferred.

        Returns
        -------
        Tensor or ndarray
            Binary labels with ``1`` for samples within the current quantile.

        Raises
        ------
        ValueError
            If the method is called more than ``T`` times.
        """
        if self.t >= self.T:
            raise ValueError(f"Already called {self.T} times!")
        if self.t > 0:
            self.percentile = self.percentile**self.eta
        self.t += 1
        return super().labels(y)


#
#  Utils
#


class _EpsDominator(Dominator):
    """Wrap pymoo's dominator with an epsilon-aware interface."""

    def __init__(self, epsilon: Optional[float]):
        """Store the dominance tolerance.

        Parameters
        ----------
        epsilon:
            Relaxation factor used in dominance comparisons.
        """
        self.epsilon = 0.0 if epsilon is None else float(epsilon)

    def calc_domination_matrix(self, F, _F=None):
        """Forward to pymoo while injecting the stored epsilon.

        Parameters
        ----------
        F, _F:
            Objective matrices expected by :class:`~pymoo.util.dominator.Dominator`.

        Returns
        -------
        ndarray
            Domination matrix respecting ``epsilon``.
        """
        # call the base implementation but *pass epsilon*
        return super().calc_domination_matrix(F, _F, epsilon=self.epsilon)


def non_dominated_sorting(
    F: np.ndarray,
    epsilon: Optional[float] = None,
    only_non_dominated_front: bool = False,
    n_stop_if_ranked: Optional[int] = None,
    n_fronts: Optional[int] = None,
) -> np.ndarray:
    """Perform epsilon-aware non-dominated sorting.

    Parameters
    ----------
    F:
        Objective matrix where lower values are preferred.
    epsilon:
        Relaxation factor applied to the dominance relation.
    only_non_dominated_front:
        If ``True``, return only the indices of the first front.
    n_stop_if_ranked:
        Early stopping threshold on the number of ranked points.
    n_fronts:
        Optional limit on the number of fronts to construct.

    Returns
    -------
    ndarray
        Either the indices of the first front or per-point ranks.
    """
    F = F.astype(float)

    # if not set just set it to a very large values because the cython
    #   algorithms do not take None
    if n_stop_if_ranked is None:
        n_stop_if_ranked = int(1e8)

    # if only_non_dominated_front is True, we only need 1 front
    if only_non_dominated_front:
        n_fronts = 1
    elif n_fronts is None:
        n_fronts = int(1e8)

    # Run fast_non_dominated_sort
    dominator = _EpsDominator(epsilon=epsilon)
    fronts = fast_non_dominated_sort(F, dominator=dominator)

    # convert to numpy array for each front and filter by n_stop_if_ranked
    _fronts = []
    n_ranked = 0
    for front in fronts:

        _fronts.append(np.array(front, dtype=int))

        # increment the n_ranked solution counter
        n_ranked += len(front)

        # stop if more solutions than n_ranked are ranked
        if n_ranked >= n_stop_if_ranked:
            break

    fronts = _fronts

    if only_non_dominated_front:
        return fronts[0]

    rank = rank_from_fronts(fronts, F.shape[0])
    return rank
