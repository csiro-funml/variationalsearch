"""Solver interfaces for VSD and other methods for poli compatibility

See https://machinelearninglifescience.github.io/poli-docs/contributing/a_new_solver.html
"""

import logging
import typing as T
from abc import ABC
from copy import deepcopy

import numpy as np
import torch
from numpy import ndarray
from poli.core.abstract_black_box import AbstractBlackBox
from poli_baselines.core.step_by_step_solver import StepByStepSolver
from torch import Tensor

from vsd.acquisition import (
    CbASAcquisition,
    LogPIClassiferAcquisition,
    VariationalSearchAcquisition,
)
from vsd.generation import (
    generate_candidates_eda,
    generate_candidates_reinforce,
)
from vsd.proposals import SequenceSearchDistribution, fit_ml
from vsd.surrogates import ClassProbabilityModel, fit_cpe
from vsd.thresholds import Threshold

LOG = logging.getLogger(name=__name__)


class _VariationalSolver(ABC, StepByStepSolver):

    optim: callable = None
    vacquisition: VariationalSearchAcquisition = None

    def __init__(
        self,
        black_box: AbstractBlackBox,
        x0: ndarray,
        y0: T.Optional[ndarray],
        threshold: Threshold,
        cpe: ClassProbabilityModel,
        vdistribution: SequenceSearchDistribution,
        prior: T.Optional[SequenceSearchDistribution] = None,
        bsize: int = 128,
        device: str | torch.device = "cpu",
        cpe_options: T.Optional[dict] = None,
        prior_options: T.Optional[None] = None,
        vdist_options: T.Optional[None] = None,
        cpe_validation_prop: float = 0,
        prior_validation_prop: float = 0,
        seed: T.Optional[int] = None,
        acq_fn_kwargs: T.Optional[dict] = None,
    ):
        super().__init__(black_box, x0, y0 if y0 is not None else black_box(x0))
        self.threshold = threshold
        self.cpe = cpe
        self.vdistribution = vdistribution.to(device)
        self.bsize = bsize
        self.prior = prior
        self.device = device
        self.cpe_validation_prop = cpe_validation_prop
        self.prior_validation_prop = prior_validation_prop
        self.seed = seed
        self.acq_fn_kwargs = {} if acq_fn_kwargs is None else acq_fn_kwargs

        self.acq = LogPIClassiferAcquisition(model=self.cpe)

        # CPE -- Fit "well"
        self.cpe_options = setdefaults(
            cpe_options,
            dict(
                optimizer_options=dict(lr=1e-3, weight_decay=1e-6),
                stop_options=dict(maxiter=10000, n_window=2000),
                batch_size=32,
            ),
        )
        # Prior -- don't over fit, need some probability mass everywhere
        self.prior_options = setdefaults(
            prior_options,
            dict(
                optimizer_options=dict(lr=1e-3, weight_decay=1e-4),
                stop_options=dict(maxiter=20000, n_window=200),
                batch_size=64,
            ),
        )
        # Posterior -- regularized, take as many iterations as needed
        self.vdist_options = setdefaults(
            vdist_options,
            dict(
                optimizer_options=dict(lr=1e-3),
                stop_options=dict(maxiter=20000, n_window=3000),
                gradient_samples=512,
            ),
        )

        # Tokenize/de-tokenize
        self._s_to_i = {s: i for i, s in enumerate(black_box.alphabet)}
        self._i_to_s = {i: s for i, s in enumerate(black_box.alphabet)}

    def next_candidate(self) -> ndarray:
        # Tokenize and convert to tensors
        x = seq2int(np.concatenate(self.history["x"], axis=0), self._s_to_i)
        y = torch.tensor(np.concatenate(self.history["y"], axis=0)).squeeze()
        maxy = max(y)

        # Updates
        thresh = self.threshold(y)
        numpos = sum(self.threshold.labels(y))
        LOG.info(
            f"Round {self.iteration}, threshold = {thresh:.3f}, "
            f"# pos = {numpos}, max y = {maxy:.3f}."
        )
        if self.prior is None:
            LOG.info("Fitting prior and initial variational distribution ...")
            self._fit_prior(x)

        LOG.info("Fitting CPE ...")
        self._fit_cpe(x, y, self.threshold)

        LOG.info(f"Fitting {self.name} ...")
        vacq = self.vacquisition(self.acq, self.prior, **self.acq_fn_kwargs)
        vacq = vacq.to(self.device)
        xcand, _ = type(self).optim(
            acquisition_function=vacq,
            proposal_distribution=self.vdistribution,
            candidate_samples=self.bsize,
            callback=_grad_callback,
            **self.vdist_options,
        )

        return int2seq(xcand, self._i_to_s)

    def _fit_cpe(self, x: Tensor, y: Tensor, thresh: Threshold):
        # Make a validation set for early stopping
        x_val = None
        y_val = None
        callback = _callback
        if (self.iteration == 0) and (self.cpe_validation_prop > 0):
            nval = max(1, round(len(x) * self.prior_validation_prop))
            indices = torch.randperm(len(x))
            valind, trind = indices[:nval], indices[nval:]
            x_val, y_val = x[valind], y[valind]
            x, y = x[trind], y[trind]
            callback = _val_callback

        fit_cpe(
            self.cpe,
            x,
            y,
            thresh,
            X_val=x_val,
            y_val=y_val,
            device=self.device,
            callback=callback,
            seed=self.seed,
            **self.cpe_options,
        )

    def _fit_prior(self, x: Tensor):
        # Make a validation set for early stopping
        x_val = None
        callback = _callback
        if self.prior_validation_prop > 0:
            nval = max(1, round(len(x) * self.prior_validation_prop))
            indices = torch.randperm(len(x))
            x_val = x[indices[:nval]]
            x = x[indices[nval:]]
            callback = _val_callback

        fit_ml(
            self.vdistribution,
            x,
            X_val=x_val,
            callback=callback,
            device=self.device,
            seed=self.seed,
            **self.prior_options,
        )

        # Make sure we are not learning the prior from now
        prior = deepcopy(self.vdistribution)
        for p in prior.parameters():
            p.requires_grad = False

        self.prior = prior


class VSDSolver(_VariationalSolver):

    name = "VSD"
    optim = generate_candidates_reinforce
    vacquisition = VariationalSearchAcquisition


class CbASSolver(_VariationalSolver):

    name = "CbAS"
    optim = generate_candidates_eda
    vacquisition = CbASAcquisition


def seq2int(S: ndarray, mapping: T.Dict[str, int]) -> Tensor:
    Xi = np.vectorize(mapping.__getitem__)(S)
    return torch.tensor(Xi).long()


def int2seq(X: Tensor, mapping: T.Dict[int, str]) -> ndarray:
    S = np.vectorize(mapping.__getitem__)(X.detach().cpu().numpy())
    return S


def setdefaults(opt: dict | None, defaults: dict) -> None:
    if opt is not None:
        defaults.update(opt)
    return defaults


def _callback(it, loss, *args, log_iters=100):
    if (it % log_iters) == 0:
        LOG.info(f"  It: {it}, Loss = {loss:.3f}")


def _val_callback(it, loss, vloss, log_iters=100):
    if (it % log_iters) == 0:
        LOG.info(f"  It: {it}, Loss = {loss:.3f}, Valid. loss = {vloss:.3f}")


def _grad_callback(it, loss, grad, log_iters=100):
    if (it % log_iters) == 0:
        mgrad = np.mean([g.detach().to("cpu").mean() for g in grad])
        LOG.info(f"  It: {it}, Loss = {loss:.3f}, Mean gradient = {mgrad:.3f}")
