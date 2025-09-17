r"""Acquisition functions for (variational) Bayesian optimisation.

This module provides small, composable wrappers used throughout VSD (Variational
Search Distributions) and A-GPS (Amortized Generation of Pareto Sets).

Key ideas
---------
1) **Class-probability acquisitions**
   - Use a classifier (CPE) that returns `log p(z=1|x)` to realise (log)
     Probability of Improvement (PI). These are black-box friendly, do not
     require surrogate gradients, and are stable under label noise.

2) **Variational search acquisitions**
   - Reparameterise the search objective with a KL regulariser against a prior
     over designs `p(x)` (or a *search distribution*). For a batch of candidate
     designs `x ~ q(x)`, the generic objective is:

       .. math::
           \mathcal{L}(x) = \underbrace{\text{acq}(x)}_{\text{utility}}
                           - \underbrace{\lambda\,\mathrm{KL}\big[q(x)\,\|\,p(x)\big]}_{\text{regularisation}}

     In practice, this is implemented point-wise as

       .. math::
           \text{acq}(x) - \lambda\,(\log q(x) - \log p(x)).

   - The scalar ``kl_weight`` (``λ``) plays the role of *power-VI* temperature:
       * ``kl_weight = 1.0`` recovers the standard VI objective and is
         appropriate when the goal is to learn the *Bayesian posterior* over
         improving designs, e.g. ``q(x) ≈ p(x | z=1)``.
       * ``kl_weight < 1.0`` (e.g. ``0.5``) down-weights the KL, allowing more
         exploitation and empirically improving black-box optimisation progress
         in early rounds at the cost of a looser approximation.

3) **CbAS**
   - Implements the original CbAS reweighting (``kl_weight = 1.0``) with a
     numerically stable exponential via a max-shift.

Shapes & conventions
--------------------
* Inputs ``X`` may arrive with superfluous singleton dimensions; ``_squeeze``
  ensures shape ``(N, D)``. ``logqX`` is a tensor broadcastable to ``(N,)``.
* ``prior_dist`` may be a torch ``Distribution`` or a ``SearchDistribution``
  from this codebase, both defining ``log_prob``.

Notes
-----
* All modules are ``torch.nn.Module`` or BoTorch ``AcquisitionFunction`` so they
  can be used with standard optimisers and reparameterisation tricks.
* See class docstrings below for usage examples.
"""

from abc import abstractmethod

import torch
from torch import nn
from botorch.acquisition import AcquisitionFunction
from botorch.acquisition.analytic import LogProbabilityOfImprovement
from torch import Tensor

from vsd.cpe import ClassProbabilityModel, PreferenceClassProbabilityModel
from vsd.proposals import SearchDistribution


class LogPIClassifierAcquisition(torch.nn.Module):
    """Log Probability of Improvement via a classifier.

    Wraps a :class:`ClassProbabilityModel` that estimates ``log p(z=1|x)`` and
    exposes it with an acquisition-function interface.

    Parameters
    ----------
    model : ClassProbabilityModel
        A classifier returning **log-probabilities** when called on ``X``.

    Forward
    -------
    X : Tensor, shape ``(..., N, D)`` or ``(N, D)``
        Candidate designs. Extra singleton dims are tolerated.

    Returns
    -------
    Tensor, shape ``(N,)``
        ``log p(z=1|x)`` evaluated at each row of ``X``.
    """

    def __init__(self, model: ClassProbabilityModel) -> None:
        super().__init__()
        self.model = model

    def forward(self, X: Tensor) -> Tensor:
        return self.model(_squeeze(X))


class PIClassiferAcquisition(LogPIClassifierAcquisition):
    """Probability of Improvement via a classifier.

    Exponentiates :class:`LogPIClassifierAcquisition` to return ``p(z=1|x)``.
    """

    def forward(self, X: Tensor) -> Tensor:
        # Convert log-PI to PI; safe because upstream uses log-sum-exp
        return torch.exp(super().forward(X))


class MarginalAcquisition(AcquisitionFunction):
    """Acquisitions that marginalise over a proposal ``q(x)``.

    These wrappers take an inner acquisition (e.g. PI/log-PI) and augment it
    with terms that depend on a *generative* or *proposal* distribution used
    during search, typically through ``log q(x)``.
    """

    def __init__(self, acquisition: AcquisitionFunction):
        super().__init__(acquisition.model)
        self.acq = acquisition

    @abstractmethod
    def forward(self, X: Tensor, logqX: Tensor) -> Tensor:
        """Evaluate the marginal acquisition.

        Parameters
        ----------
        X : Tensor, shape ``(N, D)``
            Candidate designs (post-squeeze).
        logqX : Tensor, shape ``(N,)`` or broadcastable
            Log density of the *current* proposal at ``X``.
        """
        ...


class VariationalSearchAcquisition(MarginalAcquisition):
    r"""Variational Search Acquisition (VSA).

    Implements the per-sample objective

    .. math:: \text{acq}(x) - \lambda\,(\log q(x) - \log p(x)),

    where ``acq`` is usually log-PI and ``p(x)`` is a prior or search prior.
    The parameter ``kl_weight`` corresponds to the *power-VI* coefficient.

    Guidance
    --------
    * Use ``kl_weight = 1.0`` when the aim is posterior learning
        (``q(x) ≈ p(x|z=1)``).
    * Use ``kl_weight = 0.5`` (default here) to encourage exploitation in
        black-box optimisation when a faithful posterior is not required.
    """

    def __init__(
        self,
        acquisition: LogProbabilityOfImprovement | LogPIClassifierAcquisition,
        prior_dist: torch.distributions.Distribution | SearchDistribution,
        kl_weight: float = 0.5,
    ) -> None:
        super().__init__(acquisition)
        self.prior = prior_dist
        # Default < 1.0 encourages exploitation; set to 1.0 to target the true
        #  posterior.
        self.kl_weight = kl_weight

    def forward(self, X: Tensor, logqX: Tensor) -> Tensor:
        """Return per-point objective with KL correction; shape ``(N,)``."""
        logpX = self.prior.log_prob(_squeeze(X))
        acq = self.acq(X) - self.kl_weight * (logqX - logpX)
        return acq


class CbASAcquisition(VariationalSearchAcquisition):
    """CbAS acquisition with numerically stable reweighting.

    Fixes ``kl_weight = 1.0`` (original CbAS) and applies a max-shift before the
    exponential to avoid overflow.
    """

    def __init__(
        self,
        acquisition: LogProbabilityOfImprovement | LogPIClassifierAcquisition,
        prior_dist: torch.distributions.Distribution | SearchDistribution,
    ) -> None:
        super().__init__(
            acquisition=acquisition, prior_dist=prior_dist, kl_weight=1.0
        )

    def forward(self, X: Tensor, logqX: Tensor) -> Tensor:
        log_cbas = super().forward(X, logqX)
        # Stable exponentiation: exp(a - max(a)) avoids overflow.
        return torch.exp(log_cbas - log_cbas.max())  # Shift for stability


class PreferenceAcquisition(torch.nn.Module):
    r"""Preference-aware acquisition for subjective multi-objective optimisation.

    Combines two class-probability models: one estimating Pareto non-dominance
    and
    one capturing user preferences ``u``. The forward pass returns

    .. math:: \log p(z=1 \mid x, u) + \log p(\text{pref}=1 \mid x, u).

    This additive form can be interpreted as a product of independent signals in
    log space.
    """

    def __init__(
        self,
        pareto_model: PreferenceClassProbabilityModel,
        pref_model: PreferenceClassProbabilityModel,
    ) -> None:
        super().__init__()
        self.pareto_model = pareto_model
        self.pref_model = pref_model

    def forward(self, X: Tensor, U: Tensor, logqX: Tensor) -> Tensor:
        """Compute combined log-scores for ``X`` under user preferences ``U``.

        Parameters
        ----------
        X : Tensor, shape ``(N, D)`` (after squeeze)
        U : Tensor, shape ``(N, K)`` or broadcastable
            User/context preference embeddings.
        logqX : Tensor
            Unused in this base class; kept for API symmetry with variational
            version.
        """
        X = _squeeze(X)
        logp = self.pareto_model(X, U) + self.pref_model(X, U)
        return logp


class VariationalPreferenceAcquisition(PreferenceAcquisition):
    """Preference-aware variational acquisition.

    Extends :class:`PreferenceAcquisition` with a KL correction against a prior
    ``p(x)`` exactly as in :class:`VariationalSearchAcquisition`.
    Use ``kl_weight = 1.0`` for posterior learning ``q(x) ≈ p(x|z=1, u)``; use
    smaller values (e.g. ``0.5``) for more aggressive exploitation in black-box
    search.
    """

    def __init__(
        self,
        pareto_model: PreferenceClassProbabilityModel,
        pref_model: PreferenceClassProbabilityModel,
        prior_dist: torch.distributions.Distribution | SearchDistribution,
        kl_weight: float = 0.5,
    ):
        super().__init__(pareto_model, pref_model)
        self.prior = prior_dist
        self.kl_weight = kl_weight

    def forward(self, X: Tensor, U: Tensor, logqX: Tensor) -> Tensor:
        """Return combined log-score minus KL term; shape ``(N,)``."""
        X = _squeeze(X)
        logpz = super().forward(X, U, None)
        logpX = self.prior.log_prob(X)
        acq = logpz - self.kl_weight * (logqX - logpX)
        return acq


def _squeeze(X: Tensor) -> Tensor:
    """Ensure ``X`` has at least 2D shape ``(N, D)`` by removing extra
    singleton dims.
    """
    X = torch.atleast_2d(X.squeeze())
    return X
