"""Bayesian optimisation aquisition functions and wrappers."""

from abc import abstractmethod

import torch
from botorch.acquisition import AcquisitionFunction
from botorch.acquisition.analytic import LogProbabilityOfImprovement
from torch import Tensor

from vsd.proposals import SearchDistribution
from vsd.surrogates import ClassProbabilityModel


class LogPIClassiferAcquisition(torch.nn.Module):
    """Log probability of improvement using class probability estimation."""

    def __init__(self, model: ClassProbabilityModel) -> None:
        super().__init__()
        self.model = model

    def forward(self, X: Tensor) -> Tensor:
        return self.model(_squeeze(X))


class PIClassiferAcquisition(LogPIClassiferAcquisition):
    """Probability of improvement using class probability estimation."""

    def forward(self, X: Tensor) -> Tensor:
        return torch.exp(super().forward(X))


class MarginalAcquisition(AcquisitionFunction):
    """Acquisition functions that marginalise over candidates, X.

    These can be used to reparameterize acquisition functions.
    """

    def __init__(self, acquisition: AcquisitionFunction):
        super().__init__(acquisition.model)
        self.acq = acquisition

    @abstractmethod
    def forward(self, X: Tensor, logqX: Tensor) -> Tensor: ...


class VariationalSearchAcquisition(MarginalAcquisition):
    """Variational Search Acquisition

    KL-weight allows for implementation of power-VI.
    """

    def __init__(
        self,
        acquisition: LogProbabilityOfImprovement | LogPIClassiferAcquisition,
        prior_dist: torch.distributions.Distribution | SearchDistribution,
        kl_weight: float = 1.0,
    ) -> None:
        super().__init__(acquisition)
        self.prior = prior_dist
        self.kl_weight = kl_weight

    def forward(self, X: Tensor, logqX: Tensor) -> Tensor:
        logpX = self.prior.log_prob(_squeeze(X))
        acq = self.acq(X) - self.kl_weight * (logqX - logpX)
        return acq


class CbASAcquisition(VariationalSearchAcquisition):

    def __init__(
        self,
        acquisition: LogProbabilityOfImprovement | LogPIClassiferAcquisition,
        prior_dist: torch.distributions.Distribution | SearchDistribution,
        kl_weight: float = 1.0,
    ) -> None:
        super().__init__(
            acquisition=acquisition, prior_dist=prior_dist, kl_weight=kl_weight
        )

    def forward(self, X: Tensor, logqX: Tensor) -> Tensor:
        return torch.exp(super().forward(X, logqX))


def _squeeze(X: Tensor) -> Tensor:
    X = torch.atleast_2d(X.squeeze())
    return X
