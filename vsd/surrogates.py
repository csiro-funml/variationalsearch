"""Surrogate model interfaces and estimators."""

from abc import ABC, abstractmethod
from math import log, sqrt
from typing import Any, Callable, Dict, Optional

import torch
import torch.nn as nn
from botorch.fit import fit_gpytorch_mll_scipy
from botorch.models.gpytorch import GPyTorchModel
from botorch.models.kernels import CategoricalKernel, InfiniteWidthBNNKernel
from botorch.optim.core import OptimizationResult
from botorch.optim.stopping import ExpMAStoppingCriterion
from gpytorch.constraints import GreaterThan
from gpytorch.distributions import MultivariateNormal
from gpytorch.kernels import IndexKernel, Kernel, ScaleKernel
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.means import ConstantMean
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.models import ExactGP
from gpytorch.priors import LogNormalPrior
from torch import Tensor
from torch.optim import Optimizer

from vsd.thresholds import Threshold
from vsd.utils import PositionalEncoding, Transpose, batch_indices

#
# Gaussian process surrogates
#


class CategoricalGP(GPyTorchModel, ExactGP):

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
# Class probability estimators
#


class ClassProbabilityModel(ABC, nn.Module):

    @abstractmethod
    def _logits(self, X: Tensor) -> Tensor:
        pass

    def forward(self, X: Tensor, return_logits: bool = False) -> Tensor:
        logits = squeeze_1D(self._logits(X))
        if return_logits:
            return logits
        return nn.functional.logsigmoid(logits)


class EnsembleProbabilityModel(ClassProbabilityModel):

    def __init__(
        self,
        base_class: ClassProbabilityModel,
        init_kwargs: dict,
        ensemble_size: int = 10,
    ):
        super().__init__()
        self.base_class = ClassProbabilityModel
        self.init_kwargs = init_kwargs
        self.ensemble_size = ensemble_size
        self.ensemble = torch.nn.ModuleList(
            [base_class(**init_kwargs) for _ in range(ensemble_size)]
        )

    def _logits(self, X: Tensor) -> Tensor:
        logits = [torch.sigmoid(m._logits(X)) for m in self.ensemble]
        probs = torch.mean(torch.stack(logits), dim=0)
        return torch.logit(probs, eps=1e-5)


class ContinuousCPEModel(ClassProbabilityModel):

    def __init__(self, x_dim: int, latent_dim: int, dropoutp: float = 0):
        super().__init__()
        self.nn = torch.nn.Sequential(
            nn.Linear(in_features=x_dim, out_features=latent_dim),
            nn.Dropout(p=dropoutp),
            nn.LeakyReLU(),
            nn.Linear(in_features=latent_dim, out_features=1),
        )

    def _logits(self, X: Tensor) -> Tensor:
        return self.nn(X)


class SequenceProbabilityModel(ClassProbabilityModel):

    def __init__(self, seq_len, alpha_len) -> None:
        super().__init__()
        self.seq_len = seq_len
        self.alpha_len = alpha_len


class NNClassProbability(SequenceProbabilityModel):

    def __init__(
        self,
        seq_len,
        alpha_len,
        embedding_dim: Optional[int] = None,
        dropoutp: float = 0,
        hlsize: int = 64,
    ) -> None:
        super().__init__(seq_len, alpha_len)
        if embedding_dim is None:
            embedding_dim = max(2, self.alpha_len // 2)
        self.nn = torch.nn.Sequential(
            nn.Embedding(num_embeddings=alpha_len, embedding_dim=embedding_dim),
            nn.Dropout(p=dropoutp),
            nn.Flatten(),
            nn.LeakyReLU(),
            nn.Linear(in_features=embedding_dim * seq_len, out_features=hlsize),
            nn.LeakyReLU(),
            nn.Linear(in_features=hlsize, out_features=1),
        )

    def _logits(self, X: Tensor) -> Tensor:
        return self.nn(X)


class CNNClassProbability(SequenceProbabilityModel):

    def __init__(
        self,
        seq_len,
        alpha_len,
        embedding_dim: Optional[int] = None,
        ckernel: int = 7,
        xkernel: int = 2,
        xstride: int = 2,
        cfilter_size: int = 16,
        linear_size: int = 128,
        dropoutp: float = 0,
        pos_encoding: bool = False,
    ) -> None:
        super().__init__(seq_len, alpha_len)
        if embedding_dim is None:
            embedding_dim = max(2, self.alpha_len // 2)
        self.seq_len = seq_len
        self.alpha_len = alpha_len
        emb = [
            nn.Embedding(num_embeddings=alpha_len, embedding_dim=embedding_dim)
        ]
        if pos_encoding:
            emb.append(PositionalEncoding(emb_size=embedding_dim))
        self.nn = torch.nn.Sequential(
            *emb,
            nn.Dropout(p=dropoutp),
            Transpose(),
            nn.Conv1d(
                in_channels=embedding_dim,
                out_channels=cfilter_size,
                kernel_size=ckernel,
            ),
            nn.LeakyReLU(),
            nn.MaxPool1d(kernel_size=xkernel, stride=xstride),
            nn.Conv1d(
                in_channels=cfilter_size,
                out_channels=cfilter_size,
                kernel_size=ckernel,
            ),
            nn.LeakyReLU(),
            nn.MaxPool1d(kernel_size=xkernel, stride=xstride),
            Transpose(),
            nn.Flatten(),
            nn.LazyLinear(out_features=linear_size),
            nn.LeakyReLU(),
            nn.Linear(in_features=linear_size, out_features=1),
        )

    def _logits(self, X: Tensor) -> Tensor:
        return self.nn(X)


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


def fit_cpe(
    model: ClassProbabilityModel,
    X: Tensor,
    y: Tensor,
    best_f: float | Threshold,
    X_val: Optional[Tensor] = None,
    y_val: Optional[Tensor] = None,
    batch_size: int = 256,
    optimizer: Optimizer = torch.optim.AdamW,
    optimizer_options: Optional[Dict[str, Any]] = None,
    stop_options: Optional[Dict[str, Any]] = None,
    device: str = "cpu",
    callback: Optional[Callable[[int, Tensor], None]] = None,
    seed: Optional[int] = None,
    stop_using_val_loss: bool = False,
):
    """TODO"""
    if (X_val is None) and stop_using_val_loss:
        raise ValueError("Need to specify X_val to stop_using_xval_loss")

    model.to(device)
    optimizer_options = {} if optimizer_options is None else optimizer_options
    stop_options = {} if stop_options is None else stop_options
    optim = optimizer(model.parameters(), **optimizer_options)
    lossfn = torch.nn.BCEWithLogitsLoss()
    stopping_criterion = ExpMAStoppingCriterion(**stop_options)  # type: ignore

    z = _get_labels(y, best_f)
    if X_val is None:
        vloss = torch.zeros([])
    else:
        X_val = X_val.to(device)
        z_val = _get_labels(y_val, best_f).to(device)

    model.train()
    for i, bi in enumerate(batch_indices(len(z), batch_size, seed)):
        Xb = torch.atleast_2d(X[bi].to(device))
        zb = z[bi].to(device)
        loss = lossfn(model(Xb, return_logits=True), zb)
        loss.backward()
        optim.step()
        optim.zero_grad()

        sloss = loss.detach()
        if X_val is not None:
            model.eval()
            with torch.no_grad():
                vloss = lossfn(model(X_val, return_logits=True), z_val).detach()
            if stop_using_val_loss:
                sloss = vloss
            model.train()

        if callback is not None:
            callback(i, loss, vloss)
        if stopping_criterion.evaluate(fvals=sloss):
            break
    model.eval()


def _get_labels(y: Tensor, best_f: float | Threshold) -> Tensor:
    if isinstance(best_f, Threshold):
        return best_f.labels(y).float()
    return (y >= best_f).float()


def squeeze_1D(x: Tensor) -> Tensor:
    return torch.atleast_1d(x.squeeze())
