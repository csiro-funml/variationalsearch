"""Functions for composing search methods."""

from copy import deepcopy
from typing import Any, Callable, Tuple

import numpy as np
import torch
from botorch.acquisition import AcquisitionFunction, ProbabilityOfImprovement
from botorch.acquisition.analytic import LogProbabilityOfImprovement
from botorch.models.gpytorch import GPyTorchModel

from vsd import proposals
from vsd.acquisition import (
    CbASAcquisition,
    LogPIClassifierAcquisition,
    PIClassiferAcquisition,
    VariationalSearchAcquisition,
)
from vsd.generation import (
    generate_candidates_adalead,
    generate_candidates_eda,
    generate_candidates_pex,
    generate_candidates_iw,
)
from vsd.proposals import (
    SearchDistribution,
    SequenceUninformativePrior,
    MaskedSearchDistribution,
)
from vsd.cpe import ClassProbabilityModel
from vsd.utils import SequenceTensor


def _load_prior(
    seq_len: int, alpha_len: int, prior_config: dict, device: str
) -> SearchDistribution | SequenceUninformativePrior:
    prior = _load_proposal(seq_len, alpha_len, prior_config, device)
    if prior_config["trainable"]:
        sd = torch.load(
            f=prior_config["save_path"],
            map_location=torch.device(device),
            weights_only=True,
        )
        prior.load_state_dict(sd)
        prior.eval()

    # Make sure we are not learning the prior
    for p in prior.parameters():
        p.requires_grad = False

    return prior


def _load_proposal(
    seq_len: int, alpha_len: int, prop_config: dict, device: str
) -> SearchDistribution | SequenceUninformativePrior:
    pclass = getattr(proposals, prop_config["class"])
    dist = pclass(
        d_features=seq_len, k_categories=alpha_len, **prop_config["parameters"]
    )
    return dist.to(device)


def _copy_prior(prior: SearchDistribution, proposal: SearchDistribution):
    if not isinstance(proposal, type(prior)):
        msg = (
            f"Prior {type(prior)} and proposal {type(proposal)} not compatible!"
        )
        raise TypeError(msg)
    proposal.load_state_dict(deepcopy(prior.state_dict()))

    # Make sure we are learning the proposal
    for p in proposal.parameters():
        p.requires_grad = True


def _initialise_transition_proposal(
    proposal: MaskedSearchDistribution | SearchDistribution,
    X_train: SequenceTensor,
    y_train: torch.Tensor,
    config: dict,
):
    if not isinstance(proposal, MaskedSearchDistribution):
        return
    sind = torch.argsort(y_train, descending=True)[: config["b_cands"]]
    X0 = X_train[sind]
    proposal.set_seeds(X0.to(config["device"]))


def _create_logpi_acquistition(
    surrogate: ClassProbabilityModel | GPyTorchModel,
    best_f: torch.Tensor | float,
) -> LogPIClassifierAcquisition | LogProbabilityOfImprovement:
    """Basic log-PI acquisition"""
    if isinstance(surrogate, GPyTorchModel):
        return LogProbabilityOfImprovement(model=surrogate, best_f=best_f)
    return LogPIClassifierAcquisition(model=surrogate)


def _create_pi_acquistition(
    surrogate: ClassProbabilityModel | GPyTorchModel,
    best_f: torch.Tensor | float,
) -> PIClassiferAcquisition | ProbabilityOfImprovement:
    """Basic PI acquisition"""
    if isinstance(surrogate, GPyTorchModel):
        return ProbabilityOfImprovement(model=surrogate, best_f=best_f)
    return PIClassiferAcquisition(model=surrogate)


def _create_score_acquistition(
    surrogate: ClassProbabilityModel | GPyTorchModel,
) -> PIClassiferAcquisition | Callable:
    """Acquisition for models that don't use an acquisition"""
    if isinstance(surrogate, GPyTorchModel):

        class _GPScoreAdapt(torch.nn.Module):

            def __init__(self):
                super().__init__()
                self.model = surrogate

            def forward(self, X):
                return self.model(X).mean

        return _GPScoreAdapt()

    return PIClassiferAcquisition(surrogate)


def get_vsd_components(
    seq_len: int,
    alpha_len: int,
    surrogate: ClassProbabilityModel | GPyTorchModel,
    best_f: torch.Tensor | float,
    config: dict,
    X_train: SequenceTensor,
    y_train: np.ndarray,
) -> Tuple[AcquisitionFunction, SearchDistribution, callable]:
    """Return components for running VSD."""
    device = config["device"]
    prior = _load_prior(seq_len, alpha_len, config["prior"], device)
    proposal = _load_proposal(seq_len, alpha_len, config["proposal"], device)
    if config["proposal"]["from_prior"]:
        _copy_prior(prior, proposal)
    _initialise_transition_proposal(proposal, X_train, y_train, config)
    acq = _create_logpi_acquistition(surrogate, best_f).to(device)
    vsd = VariationalSearchAcquisition(acq, prior).to(device)
    return vsd, proposal, _wrap_reinforce_generation(config)


def get_cbas_components(
    seq_len: int,
    alpha_len: int,
    surrogate: ClassProbabilityModel,
    best_f: torch.Tensor | float,
    config: dict,
    X_train: SequenceTensor,
    y_train: torch.Tensor,
) -> Tuple[AcquisitionFunction, SearchDistribution, callable]:
    """Return components for running CbAS."""
    device = config["device"]
    prior = _load_prior(seq_len, alpha_len, config["prior"], device)
    proposal = _load_proposal(seq_len, alpha_len, config["proposal"], device)
    if config["proposal"]["from_prior"]:
        _copy_prior(prior, proposal)
    _initialise_transition_proposal(proposal, X_train, y_train, config)
    acq = _create_logpi_acquistition(surrogate, best_f).to(device)
    cbas = CbASAcquisition(acq, prior).to(device)
    return cbas, proposal, _wrap_eda_generation(config)


def get_bore_components(
    seq_len: int,
    alpha_len: int,
    surrogate: ClassProbabilityModel,
    best_f: torch.Tensor | float,
    config: dict,
    X_train: SequenceTensor,
    y_train: torch.Tensor,
) -> Tuple[AcquisitionFunction, SearchDistribution, callable]:
    """Return components for running BORE with reinforce."""
    device = config["device"]
    proposal = _load_proposal(seq_len, alpha_len, config["proposal"], device)
    if config["proposal"]["from_prior"]:
        prior = _load_prior(seq_len, alpha_len, config["prior"], device)
        _copy_prior(prior, proposal)
    _initialise_transition_proposal(proposal, X_train, y_train, config)
    acq = _create_logpi_acquistition(surrogate, best_f).to(device)
    return acq, proposal, _wrap_reinforce_generation(config)


def get_dbas_components(
    seq_len: int,
    alpha_len: int,
    surrogate: ClassProbabilityModel,
    best_f: torch.Tensor | float,
    config: dict,
    X_train: SequenceTensor,
    y_train: torch.Tensor,
) -> Tuple[AcquisitionFunction, SearchDistribution, callable]:
    """Return components for running DbAS."""
    device = config["device"]
    proposal = _load_proposal(seq_len, alpha_len, config["proposal"], device)
    if config["proposal"]["from_prior"]:
        prior = _load_prior(seq_len, alpha_len, config["prior"], device)
        _copy_prior(prior, proposal)
    _initialise_transition_proposal(proposal, X_train, y_train, config)
    acq = _create_pi_acquistition(surrogate, best_f).to(device)
    return acq, proposal, _wrap_eda_generation(config)


def get_rs_components(
    seq_len: int,
    alpha_len: int,
    surrogate: ClassProbabilityModel,
    best_f: torch.Tensor | float,
    config: dict,
    X_train: SequenceTensor,
    y_train: torch.Tensor,
) -> Tuple[AcquisitionFunction, SearchDistribution, callable]:
    """Return components for running a random search with scoring."""
    device = config["device"]
    proposal = _load_prior(seq_len, alpha_len, config["prior"], device)
    acq = _create_score_acquistition(surrogate).to(device)

    def generate_random(acquisition_function, proposal_distribution, callback):
        X_rand, _ = proposal_distribution(samples=config["b_cands"])
        scores = acquisition_function(X_rand)
        callback(0, torch.tensor(0), torch.tensor([0.0]))
        return X_rand, scores

    return acq, proposal, generate_random


def get_adalead_components(
    seq_len: int,
    alpha_len: int,
    surrogate: ClassProbabilityModel,
    best_f: torch.Tensor | float,
    config: dict,
    X_train: SequenceTensor,
    y_train: torch.Tensor,
) -> Tuple[AcquisitionFunction, Any, callable]:
    """Return components for AdaLead."""
    acq = _create_score_acquistition(surrogate)
    proposal = _CandidateContainer(X_train)

    def generate(acquisition_function, proposal_distribution, callback):
        X_init = proposal_distribution.get_candidates()
        acquisition_function.to("cpu")
        X_cand, X_cand_acq = generate_candidates_adalead(
            X_init=X_init,
            surrogate=acquisition_function,
            alphalen=alpha_len,
            batchsize=config["b_cands"],
            **config["proposal"]["adalead_options"],
        )
        callback(0, torch.tensor(0), torch.tensor([0.0]))
        proposal_distribution.update(X_cand)
        return X_cand, X_cand_acq

    return acq, proposal, generate


def get_pex_components(
    seq_len: int,
    alpha_len: int,
    surrogate: ClassProbabilityModel,
    best_f: torch.Tensor | float,
    config: dict,
    X_train: SequenceTensor,
    y_train: torch.Tensor,
) -> Tuple[AcquisitionFunction, Any, callable]:
    """Return components for PEX."""
    acq = _create_score_acquistition(surrogate)

    # Use the fittest/unfittest training sample as the incumbent
    X_incumbent = X_train[torch.argmin(y_train)]
    proposal = _CandidateContainer(X_train)

    def generate(acquisition_function, proposal_distribution, callback):
        X_init = proposal_distribution.get_candidates()
        acquisition_function.to("cpu")
        X_cand, X_cand_acq = generate_candidates_pex(
            X_init=X_init,
            X_incumbent=X_incumbent,
            surrogate=acquisition_function,
            alphalen=alpha_len,
            batchsize=config["b_cands"],
            **config["proposal"]["pex_options"],
        )
        callback(0, torch.tensor(0), torch.tensor([0.0]))
        proposal_distribution.update(torch.vstack((X_cand, X_init)))
        return X_cand, X_cand_acq

    return acq, proposal, generate


class _CandidateContainer:

    def __init__(self, X):
        self.update(X)

    def update(self, X):
        self.X = X

    def get_candidates(self):
        return self.X

    def train(self):
        pass

    def eval(self):
        pass


def _wrap_reinforce_generation(config: dict) -> callable:
    def generate(acquisition_function, proposal_distribution, callback):
        candidate_samples = config["b_cands"]
        X_cand, acq_cand = generate_candidates_iw(
            acquisition_function=acquisition_function,
            proposal_distribution=proposal_distribution,
            stop_options=config["proposal"]["stop"],
            optimizer_options=config["proposal"]["optimisation"],
            gradient_samples=config["proposal"]["samples"],
            candidate_samples=candidate_samples,
            callback=callback,
        )
        if isinstance(proposal_distribution, MaskedSearchDistribution):
            proposal_distribution.set_seeds(X_cand)
        return X_cand, acq_cand

    return generate


def _wrap_eda_generation(config: dict) -> callable:
    def generate(acquisition_function, proposal_distribution, callback):
        candidate_samples = config["b_cands"]
        X_cand, acq_cand = generate_candidates_eda(
            acquisition_function=acquisition_function,
            proposal_distribution=proposal_distribution,
            stop_options=config["proposal"]["stop"],
            optimizer_options=config["proposal"]["optimisation"],
            gradient_samples=config["proposal"]["samples"],
            candidate_samples=candidate_samples,
            callback=callback,
        )
        if isinstance(proposal_distribution, MaskedSearchDistribution):
            proposal_distribution.set_seeds(X_cand)
        return X_cand, acq_cand

    return generate
