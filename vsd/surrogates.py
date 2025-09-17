"""Gaussian process surrogate models and learning routines.

Compact GP models for sequence inputs and helpers to fit or update them.
"""

from typing import Any, Callable, Dict, Optional
import torch
from math import log, sqrt
from torch import Tensor
from botorch.fit import fit_gpytorch_mll_scipy
from botorch.models.gpytorch import GPyTorchModel
from botorch.models.kernels import CategoricalKernel, InfiniteWidthBNNKernel
from botorch.optim.core import OptimizationResult
from gpytorch.constraints import GreaterThan
from gpytorch.distributions import MultivariateNormal
from gpytorch.kernels import IndexKernel, Kernel, ScaleKernel
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.means import ConstantMean
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.models import ExactGP
from gpytorch.priors import LogNormalPrior


#
# GP Surrogates for sequences
#


class CategoricalGP(GPyTorchModel, ExactGP):
    """GP with a categorical kernel for token sequences.

    Parameters
    ----------
    seq_len : int
        Sequence length.
    alpha_len : int
        Alphabet size.
    X, y : Tensor
        Training data.
    ard : bool, default=False
        If True, use per-position lengthscales.
    """

    num_outputs = 1

    def __init__(
        self, seq_len, alpha_len, X: Tensor, y: Tensor, ard: bool = False
    ):
        super().__init__(
            train_inputs=X,
            train_targets=y,
            likelihood=GaussianLikelihood(),
        )
        self.d = seq_len
        self.k = alpha_len
        ard_num_dims = self.d if ard else None
        lmu = sqrt(2) + log(ard_num_dims) * 0.5
        self.mean_module = ConstantMean()
        self.covar_module = ScaleKernel(
            base_kernel=CategoricalKernel(
                ard_num_dims=ard_num_dims,
                lengthscale_prior=LogNormalPrior(loc=lmu, scale=sqrt(3)),
                lengthscale_constraint=GreaterThan(2.5e-2),
            )
        )
        self.to(y)  # make sure we're on the right device/dtype

    def forward(self, X: Tensor) -> Tensor:
        mean_x = self.mean_module(X)
        covar_x = self.covar_module(X)
        return MultivariateNormal(mean_x, covar_x)


class IndexGP(GPyTorchModel, ExactGP):
    """GP with an index kernel over alphabet indices.

    Parameters
    ----------
    seq_len, alpha_len : int
        Sequence length and alphabet size.
    X, y : Tensor
        Training data.
    rank : int, optional
        Rank for the index kernel embedding.
    """

    num_outputs = 1

    def __init__(
        self,
        seq_len,
        alpha_len,
        X: Tensor,
        y: Tensor,
        rank: Optional[int] = None,
    ):
        self.d = seq_len
        self.k = alpha_len
        super().__init__(
            train_inputs=X,
            train_targets=y,
            likelihood=GaussianLikelihood(),
        )
        rank = max(2, alpha_len // 2) if rank is None else rank
        self.mean_module = ConstantMean()
        self.covar_module = ScaleKernel(
            base_kernel=IndexKernel(num_tasks=self.k, rank=rank)
        )
        self.to(y)  # make sure we're on the right device/dtype

    def forward(self, X: Tensor) -> Tensor:
        mean_x = self.mean_module(X)
        covar_x = self.covar_module(X)
        return MultivariateNormal(mean_x, covar_x)


class IBNN(GPyTorchModel, ExactGP):
    """Infinite-width BNN kernel GP for sequences."""

    num_outputs = 1

    def __init__(
        self,
        seq_len,
        alpha_len,
        X: Tensor,
        y: Tensor,
        depth: Optional[int] = 3,
    ):
        self.d = seq_len
        self.k = alpha_len
        super().__init__(
            train_inputs=X,
            train_targets=y,
            likelihood=GaussianLikelihood(),
        )
        ibnn = _onehot_kernel(InfiniteWidthBNNKernel, num_classes=alpha_len)
        self.mean_module = ConstantMean()
        self.covar_module = ScaleKernel(base_kernel=ibnn(depth=depth))
        self.to(y)  # make sure we're on the right device/dtype

    def forward(self, X: Tensor) -> Tensor:
        mean_x = self.mean_module(X)
        covar_x = self.covar_module(X)
        return MultivariateNormal(mean_x, covar_x)


def _onehot_kernel(kernelclass: Kernel, num_classes: int) -> Kernel:

    class _CategoricalInputs(kernelclass):

        def forward(self, x1: Tensor, x2: Tensor, *args, **kwargs) -> Tensor:
            n1, n2 = list(x1.shape[:-1]), list(x2.shape[:-1])
            x1 = torch.nn.functional.one_hot(x1, num_classes).reshape(n1 + [-1])
            x2 = torch.nn.functional.one_hot(x2, num_classes).reshape(n2 + [-1])
            return super().forward(x1.float(), x2.float(), *args, **kwargs)

    return _CategoricalInputs


#
#   Model fitting and updating
#


def fit_gp(
    model: GPyTorchModel,
    optimiser_options: Optional[Dict[str, Any]] = None,
    stop_options: Optional[Dict[str, Any]] = None,
    device: str = "cpu",
    callback: Optional[Callable[[Tensor, OptimizationResult], None]] = None,
):
    """Fit a GP by maximising the exact marginal log-likelihood (scipy).

    Parameters mirror ``botorch.fit_gpytorch_mll_scipy``.
    """
    optimiser_options = {} if optimiser_options is None else optimiser_options
    stop_options = {} if stop_options is None else stop_options

    model.to(device)
    model.train()
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    mll = fit_gpytorch_mll_scipy(
        mll, options=optimiser_options, callback=callback
    )
    model.eval()


def update_gp(
    model: GPyTorchModel,
    X: Tensor,
    y: Tensor,
    device: str = "cpu",
    refit: bool = False,
    optimizer_options: Optional[Dict[str, Any]] = None,
    stop_options: Optional[Dict[str, Any]] = None,
    callback: Optional[Callable[[Tensor, OptimizationResult], None]] = None,
):
    """Update training data for a fitted GP and optionally refit hyperparams."""
    model.set_train_data(
        inputs=X.to(device), targets=y.to(device), strict=False
    )
    if refit:
        fit_gp(
            model=model,
            optimiser_options=optimizer_options,
            stop_options=stop_options,
            device=device,
            callback=callback,
        )
