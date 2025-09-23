"""Candidate generation routines for variational & preference-aware search.

This module provides BoTorch-compatible generators used with VSD/A-GPS style
acquisitions. It implements three optimisation strategies over proposal
(search) distributions and two heuristic sequence algorithms:

- ``generate_candidates_reinforce``: on-policy REINFORCE with a moving-average
  control variate (baseline) and optional timeout/stopping.
- ``generate_candidates_iw``: off-policy REINFORCE with importance weighting
  and effective-sample-size (ESS) driven resampling.
- ``generate_candidates_eda``: estimation-of-distributions (weighted MLE) with
  optional periodic resampling of the proposal.
- ``generate_candidates_pex``: Proximal Exploration (Ren et al., 2022) for
  sequence design.
- ``generate_candidates_adalead``: AdaLead (Sinai et al., 2020) for sequence
  design with greedy rollouts and recombination.

Notes
-----
Shapes are explicit where helpful, and all routines avoid differentiating
through the acquisition itself unless stated otherwise.
"""

import time
import warnings
from inspect import signature
from itertools import batched
from typing import Callable, Dict, NoReturn, Optional, Tuple, Union, List

import numpy as np
import torch
from botorch.acquisition import AcquisitionFunction
from botorch.logging import _get_logger

# from botorch.optim.stopping import ExpMAStoppingCriterion
from polyleven import levenshtein
from scipy.spatial import cKDTree
from torch import IntTensor, Tensor
from torch.optim import Optimizer

from vsd.acquisition import PreferenceAcquisition, MarginalAcquisition
from vsd.condproposals import PreferenceSearchDistribution
from vsd.proposals import (
    SearchDistribution,
    clip_gradients,
)
from vsd.utils import SequenceTensor, SEPlateauStopping

logger = _get_logger()


def generate_candidates_reinforce(
    acquisition_function: AcquisitionFunction | PreferenceAcquisition,
    proposal_distribution: SearchDistribution | PreferenceSearchDistribution,
    cv_smoothing: float = 0.7,
    optimizer: type[Optimizer] = torch.optim.Adam,
    optimizer_options: Optional[Dict[str, Union[float, str]]] = None,
    stop_options: Optional[Dict[str, Union[float, str]]] = None,
    callback: Optional[
        Callable[[int, Tensor, List[Tensor | None]], NoReturn]
    ] = None,
    timeout_sec: Optional[float] = None,
    gradient_samples: Optional[int] = None,
    candidate_samples: Optional[int] = None,
) -> Tuple[Tensor, Tensor]:
    """Generate candidates using an on-policy REINFORCE reparameterisation.

    Parameters
    ----------
    acquisition_function : AcquisitionFunction | PreferenceAcquisition
        Acquisition / black-box function. May be a plain BoTorch acquisition
        or a preference-aware wrapper.
    proposal_distribution : SearchDistribution | PreferenceSearchDistribution
        Proposal/search distribution to optimise with REINFORCE and sample
        candidates from.
    cv_smoothing : float, default=0.7
        Exponential moving-average coefficient for the control-variate (baseline).
    optimizer : Optimizer, default=torch.optim.Adam
        Optimiser class used for parameter updates.
    optimizer_options : dict, optional
        Keyword arguments passed to ``optimizer`` (e.g., ``lr``, ``weight_decay``).
    stop_options : dict, optional
        Keyword arguments for :class:`SEPlateauStopping` (e.g., ``patience``,
        ``min_delta``).
    callback : callable, optional
        Function ``callback(iteration, loss, gradients)`` invoked each step.
    timeout_sec : float, optional
        Wall-clock timeout for optimisation; returns best found upon expiry.
    gradient_samples : int, optional
        Number of proposal samples per step for gradient estimation.
    candidate_samples : int, optional
        Number of final candidates to draw for return.

    Returns
    -------
    (Tensor, Tensor)
        ``Xcand`` and their acquisition values ``Xcand_acq``.

    Notes
    -----
    The loss implements the classic REINFORCE estimator ``E[(a-b) * log q(x)]``
    with a moving-average baseline ``b``. Gradients do **not** flow through the
    acquisition value.
    """
    acquisition_function = _adapt_acquisition(acquisition_function)
    start_time = time.monotonic()
    optimizer_options = optimizer_options or {"weight_decay": 0.0}
    stop_options = stop_options or {}

    # Set up the optimiser
    clip_gradients(proposal_distribution)
    params = list(proposal_distribution.parameters())
    _optimizer = optimizer(params=params, **optimizer_options)  # type: ignore

    i, b = 0, torch.tensor(0)
    stop = False
    stopping_criterion = SEPlateauStopping(**stop_options)  # type: ignore
    proposal_distribution.train()
    while not stop:
        X, U, logqX = _sample_proposal(
            proposal_distribution, gradient_samples, with_gradients=True
        )

        # Reinforce does not differentiate through acq
        with torch.no_grad():
            nacq = -acquisition_function(X, U, logqX)

        if nacq.ndim != logqX.ndim:
            raise RuntimeError(f"acq. dim:{nacq.ndim} != logp dim:{logqX.ndim}")
        loss = ((nacq - b) * logqX).mean()  # Reinforce gradient loss

        with torch.no_grad():
            lossa = nacq.mean()  # actual loss
            if i == 0:
                b = lossa
            b = (1 - cv_smoothing) * b + cv_smoothing * lossa  # baseline

        loss.backward()
        if callback:
            callback(i, lossa, [p.grad for p in params])

        _optimizer.step()
        _optimizer.zero_grad()

        stop = stopping_criterion.evaluate(fvals=lossa)
        i += 1

        if timeout_sec is not None:
            runtime = time.monotonic() - start_time
            if runtime > timeout_sec:
                stop = True
                logger.info(f"Optimization timed out after {runtime} seconds.")

    # Sample candidates
    with torch.no_grad():
        Xcand, Ucand, logqX = _sample_proposal(
            proposal_distribution, candidate_samples
        )
        Xcand_acq = acquisition_function(Xcand, Ucand, logqX)
    proposal_distribution.eval()
    return Xcand, Xcand_acq


def generate_candidates_iw(
    acquisition_function: AcquisitionFunction,
    proposal_distribution: SearchDistribution | PreferenceSearchDistribution,
    optimizer: type[Optimizer] = torch.optim.Adam,
    optimizer_options: Optional[Dict[str, Union[float, str]]] = None,
    stop_options: Optional[Dict[str, Union[float, str]]] = None,
    callback: Optional[
        Callable[[int, Tensor, List[Tensor | None]], NoReturn]
    ] = None,
    timeout_sec: Optional[float] = None,
    gradient_samples: Optional[int] = None,
    candidate_samples: Optional[int] = None,
    resample_ess_p: float = 0.33,
    resample_iters: int = 101,
) -> Tuple[Tensor, Tensor]:
    """Generate candidates using off-policy importance weighting.

    Parameters
    ----------
    acquisition_function : AcquisitionFunction
        Acquisition or black-box function to evaluate candidate quality.
    proposal_distribution : SearchDistribution | PreferenceSearchDistribution
        Distribution to optimise and from which to sample candidates.
    optimizer : Optimizer, default=torch.optim.Adam
        Optimiser class used for parameter updates.
    optimizer_options : dict, optional
        Keyword arguments passed to ``optimizer`` (e.g., ``lr``, ``weight_decay``).
    stop_options : dict, optional
        Keyword arguments for :class:`SEPlateauStopping`.
    callback : callable, optional
        Function ``callback(iteration, loss, gradients)`` invoked each step.
    timeout_sec : float, optional
        Wall-clock timeout for optimisation; returns best found upon expiry.
    gradient_samples : int, optional
        Number of samples to draw for gradient/weight estimation.
    candidate_samples : int, optional
        Number of final samples to draw for candidate evaluation.
    resample_ess_p : float, default=0.5
        Resample the reference distribution when the effective sample size ratio
        ``ESS / N`` falls below this threshold (``1``≈on-policy, ``0``=never).
    resample_iters : int, default=101
        Force resampling every fixed number of iterations regardless of ESS.

    Returns
    -------
    (Tensor, Tensor)
        ``Xcand`` and their acquisition values ``Xcand_acq``.
    """
    acquisition_function = _adapt_acquisition(acquisition_function)
    start_time = time.monotonic()
    optimizer_options = optimizer_options or {"weight_decay": 0.0}
    stop_options = stop_options or {}

    # Set up the optimiser
    clip_gradients(proposal_distribution)
    params = list(proposal_distribution.parameters())
    _optimizer = optimizer(params=params, **optimizer_options)  # type: ignore

    i, ess_p, stop, low_ess = 0, 1.0, False, False
    stopping_criterion = SEPlateauStopping(**stop_options)  # type: ignore
    proposal_distribution.train()
    while not stop:

        # Resample if effective sample size drops, or specfied iterations
        if low_ess or (i % resample_iters == 0):
            X, U, logpX = _sample_proposal(
                proposal_distribution, gradient_samples
            )
            Xs = (X,) if U is None else (X, U)
            low_ess = False
            N = len(X)

        # Score samples under proposal
        logqX = proposal_distribution.log_prob(*Xs)

        # Off-policy method does not differentiate through acq
        with torch.no_grad():

            # Importance weights (generous clamp to allow good ess_p calc.)
            iw = torch.exp(logqX - logpX).clamp(10 * N)
            ess_p = iw.sum().square() / (N * iw.square().sum())

            # Check effective samples size, ess_p is nan if any iw are inf
            if torch.isnan(ess_p) or (ess_p < resample_ess_p):
                low_ess = True
                _optimizer.zero_grad()  # forget current round
                continue

            nacq = -acquisition_function(X, U, logqX)
            if nacq.ndim != logqX.ndim:
                raise RuntimeError(
                    f"acquisition dim {nacq.ndim} != logp dim {logqX.ndim}"
                )

            # Baseline variance reduction
            b = (iw * nacq).sum() / iw.sum()  # also "raw" loss

        # IW off policy loss
        loss = (iw * (nacq - b) * logqX).sum()

        loss.backward()
        if callback:
            callback(i, b, [p.grad for p in params])
        _optimizer.step()
        _optimizer.zero_grad()

        stop = stopping_criterion.evaluate(fvals=b)
        i += 1

        if timeout_sec is not None:
            runtime = time.monotonic() - start_time
            if runtime > timeout_sec:
                stop = True
                logger.info(f"Optimization timed out after {runtime} seconds.")

    # Sample candidates
    Xcand, Ucand, logqX = _sample_proposal(
        proposal_distribution, candidate_samples
    )
    Xcand_acq = acquisition_function(Xcand, Ucand, logqX).detach()
    proposal_distribution.eval()
    return Xcand, Xcand_acq


def generate_candidates_eda(
    acquisition_function: AcquisitionFunction,
    proposal_distribution: SearchDistribution | PreferenceSearchDistribution,
    optimizer: type[Optimizer] = torch.optim.Adam,
    optimizer_options: Optional[Dict[str, Union[float, str]]] = None,
    stop_options: Optional[Dict[str, Union[float, str]]] = None,
    callback: Optional[
        Callable[[int, Tensor, List[Tensor | None]], NoReturn]
    ] = None,
    timeout_sec: Optional[float] = None,
    gradient_samples: Optional[int] = None,
    candidate_samples: Optional[int] = None,
    resample_iters: Optional[int] = 101,
) -> Tuple[Tensor, Tensor]:
    """Generate candidates using Estimation of Distributions (EDA).

    Parameters
    ----------
    acquisition_function : AcquisitionFunction
        Acquisition / black-box function used to form non-negative weights.
    proposal_distribution : SearchDistribution | PreferenceSearchDistribution
        Proposal to optimise with weighted maximum likelihood (WML).
    optimizer : Optimizer, default=torch.optim.Adam
        Optimiser class used for parameter updates.
    optimizer_options : dict, optional
        Keyword arguments passed to ``optimizer``.
    stop_options : dict, optional
        Keyword arguments for :class:`SEPlateauStopping`.
    callback : callable, optional
        Function ``callback(iteration, loss, gradients)`` invoked each step.
    timeout_sec : float, optional
        Wall-clock timeout for optimisation; returns best found upon expiry.
    gradient_samples : int, optional
        Number of samples to draw from the current proposal for WML.
    candidate_samples : int, optional
        Number of final candidate samples to return.
    resample_iters : int, optional, default=101
        Period between refreshing the sample set for WML (``None`` = online).

    Returns
    -------
    (Tensor, Tensor)
        ``Xcand`` and their acquisition values ``Xcand_acq``.

    Notes
    -----
    Uses weighted MLE with weights ``w = acq(X)`` and objective ``-E_w[log q(X)]``.
    The acquisition is not differentiated through.
    """
    acquisition_function = _adapt_acquisition(acquisition_function)
    start_time = time.monotonic()
    optimizer_options = optimizer_options or {}
    stop_options = stop_options or {}

    # Set up the optimiser
    clip_gradients(proposal_distribution)
    params = list(proposal_distribution.parameters())
    _optimizer = optimizer(params=params, **optimizer_options)  # type: ignore

    # Draw samples once to optimise against
    @torch.no_grad()
    def iw_and_samples(dist):
        X, U, logpX = _sample_proposal(dist, gradient_samples)
        Xs = (X,) if U is None else (X, U)

        # EDA does not differentiate through acq
        wght = acquisition_function(X, U, logpX).detach()
        assert all(wght >= 0)
        return wght, Xs

    i = 0
    stop = False
    stopping_criterion = SEPlateauStopping(**stop_options)  # type: ignore
    wght, Xs = iw_and_samples(proposal_distribution)
    proposal_distribution.train()
    while not stop:

        # Sample the proposal and compute importance weights
        if resample_iters is not None:
            if (i > 0) and (i % resample_iters == 0):
                wght, Xs = iw_and_samples(proposal_distribution)

        logqX = proposal_distribution.log_prob(*Xs)

        if wght.ndim != logqX.ndim:
            raise RuntimeError(
                f"acquisition dim, {wght.ndim} != logp dim, " f"{logqX.ndim}"
            )
        loss = -(wght * logqX).mean()  # EDA weighted maximum likelihood

        loss.backward()
        if callback:
            callback(i, loss.detach(), [p.grad for p in params])
        _optimizer.step()
        _optimizer.zero_grad()

        stop = stopping_criterion.evaluate(fvals=loss.detach())
        i += 1

        if timeout_sec is not None:
            runtime = time.monotonic() - start_time
            if runtime > timeout_sec:
                stop = True
                logger.info(f"Optimization timed out after {runtime} seconds.")

    # Sample candidates
    with torch.no_grad():
        Xcand, Ucand, logqX = _sample_proposal(
            proposal_distribution, candidate_samples
        )
        Xcand_acq = acquisition_function(Xcand, Ucand, logqX)
    proposal_distribution.eval()
    return Xcand, Xcand_acq


def generate_candidates_pex(
    X_init: SequenceTensor,
    X_incumbent: IntTensor,
    surrogate: torch.nn.Module,
    alphalen: int,
    batchsize: int,
    mutation_budget: int = 20,
    max_mutations: int = 10,
    prox_neigh: int = 5,
) -> Tuple[Tensor, Tensor]:
    """Use Proximal Exploration (PEX) for candidate generation.

    Parameters
    ----------
    X_init : SequenceTensor
        Initial pool of sequences.
    X_incumbent : IntTensor
        Current best (incumbent) sequence used to define proximity.
    surrogate : torch.nn.Module
        Surrogate scoring model.
    alphalen : int
        Alphabet size for mutations.
    batchsize : int
        Number of candidates to return.
    mutation_budget : int, default=20
        Rollout attempts per batch element.
    max_mutations : int, default=10
        Maximum single-sequence mutations during rollout.
    prox_neigh : int, default=5
        Number of nearest neighbours to include alongside the proximal frontier.

    Returns
    -------
    (Tensor, Tensor)
        Top-scoring candidates and their surrogate scores.

    References
    ----------
    Ren et al., ICML 2022: *Proximal exploration for model-guided protein
    sequence design*.
    """
    # Generate X_pool by random mutation on the proximal frontier of X_init
    X_prox, _ = _prox(X_incumbent, X_init, surrogate, batchsize, prox_neigh)
    X_pool = []
    npool = mutation_budget * batchsize
    while len(X_pool) < npool:
        for x in X_prox:
            k = torch.randint(low=1, high=max_mutations, size=[])
            X_pool.append(_mutate(x, alphalen, k))
            X_pool = list(torch.unique(torch.vstack(X_pool), dim=0))

    X_query = []
    while len(X_query) < batchsize:
        X_prox, ind_prox = _prox(
            X_incumbent, torch.vstack(X_pool), surrogate, batchsize
        )
        X_pool = [x for i, x in enumerate(X_pool) if i not in ind_prox]
        X_query.extend(list(X_prox))

    X_query = torch.vstack(X_query)
    X_query_score = surrogate(X_query)
    topinds = torch.argsort(X_query_score, descending=True)[:batchsize]
    return X_query[topinds], X_query_score[topinds]


def generate_candidates_adalead(
    X_init: SequenceTensor,
    surrogate: torch.nn.Module,
    alphalen: int,
    batchsize: int,
    mutation_budget: int = 20,
    max_mutations: int = 10,
    kappa_cutoff: float = 0.05,
    recombination_rate: float = 0.2,
    recombination_attempts: int = 10,
) -> Tuple[Tensor, Tensor]:
    """Use AdaLead for sequence candidate generation.

    Parameters
    ----------
    X_init : SequenceTensor
        Initial pool of sequences.
    surrogate : torch.nn.Module
        Surrogate scoring model.
    alphalen : int
        Alphabet size.
    batchsize : int
        Number of candidates to return.
    mutation_budget : int, default=20
        Rollout attempts per batch element.
    max_mutations : int, default=10
        Maximum mutations per rollout chain.
    kappa_cutoff : float, default=0.05
        Fitness threshold relative to the current best to keep seeds.
    recombination_rate : float, default=0.2
        Bernoulli probability for per-position crossover.
    recombination_attempts : int, default=10
        Max attempts without improvement before giving up.

    Returns
    -------
    (Tensor, Tensor)
        Top-scoring candidates and their surrogate scores.

    References
    ----------
    Sinai et al., 2020: *AdaLead: A simple and robust adaptive greedy search
    algorithm for sequence design*.
    """
    if kappa_cutoff > 1 or kappa_cutoff < 0:
        raise ValueError("kappa_cuttoff needs to be within [0, 1]")
    if recombination_rate > 1 or recombination_rate < 0:
        raise ValueError("recombination_rate needs to be within [0, 1]")

    # Score X_init, keep only "fit" sequences
    s = _batch_score(X_init, surrogate, batchsize)
    smax = s.max()
    keepind = s > (smax - abs(smax) * kappa_cutoff)
    if sum(keepind) < 1:
        warnings.warn(f"No X_init are fit, using {batchsize} highest scoring.")
        keepind = torch.argsort(s, descending=True)[:batchsize]
    X = X_init[keepind]

    # Candidate generation loop
    M = []
    bmuts = batchsize * mutation_budget
    attempts = 0
    while (len(M) < bmuts) and (attempts < recombination_attempts):
        lenM = len(M)
        P = X if lenM == 0 else _recombine(X, recombination_rate)
        for x in P:
            M.extend(list(_rollout(x, surrogate, alphalen, max_mutations)))

        # Keep unique only
        if len(M) > 0:
            M = list(torch.unique(torch.vstack(M), dim=0))

        if lenM == len(M):
            attempts += 1
        else:
            attempts = 0

    if len(M) < 1:
        raise RuntimeError("AdaLead failed to find any new candidates.")

    M = torch.vstack(M)
    pM = _batch_score(M, surrogate, batchsize)

    if len(M) <= batchsize:
        return M, pM

    topinds = torch.argsort(pM, descending=True)[:batchsize]
    X_cand = M[topinds]
    return X_cand, pM[topinds]


#
# PEX Routines
#


def _prox(
    X_incumbent: IntTensor,
    X: SequenceTensor,
    surrogate: torch.nn.Module,
    batchsize: int,
    neighbours: int = 0,
) -> SequenceTensor:
    """Compute proximal frontier and (optionally) include nearest neighbours.

    Parameters
    ----------
    X_incumbent : IntTensor
        Incumbent sequence.
    X : SequenceTensor
        Candidate pool.
    surrogate : torch.nn.Module
        Scoring surrogate.
    batchsize : int
        Batch size used during surrogate evaluation.
    neighbours : int, default=0
        Number of nearest neighbours to include around the frontier.

    Returns
    -------
    SequenceTensor
        Subset of ``X`` on/near the proximal frontier.
    """
    s_incumbent = _inttensor2str(X_incumbent)
    d = [levenshtein(s_incumbent, _inttensor2str(x)) for x in X]
    f = _batch_score(X, surrogate, batchsize).detach().numpy()
    dp, fp, indprox = _upper_convex_hull(d, f)

    # Include nearest neighbours of proximal frontier
    if neighbours > 0:
        dt, ft = zip(*[(d[i], f[i]) for i in range(len(d)) if i not in indprox])
        kdt = cKDTree(np.array((dt, ft)).T)
        _, nnind = kdt.query(np.array((dp, fp)).T, k=neighbours)
        indprox = list(indprox) + list(np.unique(nnind.flatten()))

    indprox = torch.tensor(indprox)
    return X[indprox], indprox


def _inttensor2str(x: IntTensor) -> str:
    """Convert an integer-encoded sequence to a concatenated string."""
    s = "".join(np.char.mod("%d", x.detach().numpy()))
    return s


def _upper_convex_hull(d, f):
    r"""Upper convex hull of 2D points with original indices.

    Parameters
    ----------
    d : array-like
        x-coordinates (e.g., distances).
    f : array-like
        y-coordinates (e.g., fitness values).

    Returns
    -------
    tuple
        ``(d_hull, f_hull, ind_hull)`` where each is a tuple/array aligned and
        ``ind_hull`` are indices into the original arrays.

    Notes
    -----
    Implements Andrew's monotone chain algorithm in :math:`O(n \log n)`.
    Based on the implementation at Wikibooks:
    https://en.wikibooks.org/wiki/Algorithm_Implementation/Geometry/Convex_hull/Monotone_chain
    """
    # De-duplicate d, keep only those with highest fitness
    fitunique = {}
    indices = {}
    for i, (di, fi) in enumerate(zip(d, f)):
        if di not in fitunique:
            fitunique[di], indices[di] = fi, i
        elif fitunique[di] <= fi:
            fitunique[di], indices[di] = fi, i

    d, f, ind = zip(*[(k, v, indices[k]) for k, v in fitunique.items()])

    # Boring case: no points or a single point possibly repeated multiple times.
    if len(d) <= 1:
        return d, f, ind

    # sort the points based on distance
    sind = np.argsort(d)
    d, f, ind = np.array(d)[sind], np.array(f)[sind], np.array(ind)[sind]

    # Make the tuple of point, with the original index
    points = list(zip(d, f, ind))

    # 2D cross product of OA and OB vectors, i.e. z-component of their 3D cross
    # product. Returns a positive value, if OAB makes a counter-clockwise turn,
    # negative for clockwise turn, and zero if the points are collinear.
    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    # Build upper hull
    upper = []
    for p in reversed(points):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)

    return tuple(zip(*upper))


#
#  AdaLead routines
#


def _rollout(
    x: IntTensor,
    surrogate: torch.nn.Module,
    alphalen: int,
    max_mutations: int,
    tolerance: float = 0.01,
) -> SequenceTensor:
    """One-step-at-a-time mutation rollout retaining near-neutral moves.

    Parameters
    ----------
    x : IntTensor
        Seed sequence.
    surrogate : torch.nn.Module
        Surrogate scoring model.
    alphalen : int
        Alphabet size.
    max_mutations : int
        Number of sequential mutations to attempt.
    tolerance : float, default=0.01
        Keep mutated sequences whose score is within ``tolerance * |score(x)|``.

    Returns
    -------
    SequenceTensor
        A set of retained mutated sequences.
    """
    M = torch.zeros(size=(max_mutations + 1, x.size(0)), dtype=x.dtype)
    M[0] = x
    for i in range(1, max_mutations + 1):
        M[i] = _mutate(M[i - 1], alphalen, 1)

    # Make this vectorised, since calling surrogates like GPs can be slow.
    scores = surrogate(M)
    keepind = scores >= (scores[0] - tolerance * abs(scores[0]))
    keepind[0] = False  # don't keep input sequence

    return M[keepind, :]


def _mutate(x: IntTensor, alphalen: int, k_mutations: int = 1) -> IntTensor:
    """Randomly mutate ``k_mutations`` positions of a sequence in-place copy."""
    m = len(x)
    xm = x.clone()
    p = torch.randint(m, size=[k_mutations])
    a = torch.randint(alphalen, size=[k_mutations])
    xm[p] = a
    return xm


def _recombine(X: SequenceTensor, recombination_rate: float) -> SequenceTensor:
    """Randomly recombine (crossover) pairs of sequences.

    Parameters
    ----------
    X : SequenceTensor
        Pool of sequences to recombine.
    recombination_rate : float
        Per-position Bernoulli probability for taking the partner's token.

    Returns
    -------
    SequenceTensor
        Unique set of recombined sequences.
    """
    n, m = X.shape
    if n < 2:
        return X

    # Permute indices
    ind = torch.randperm(n=n)

    # Cross
    Xcross = []
    p = torch.ones(m) * recombination_rate
    for i, j in zip(ind[0:-1:2], ind[1::2]):
        positions = torch.bernoulli(p).type(torch.IntTensor)
        xi, xj = X[i].clone(), X[j].clone()
        xi[positions] = X[j][positions]
        xj[positions] = X[i][positions]
        Xcross += [xi, xj]
    return torch.unique(torch.vstack(Xcross), dim=0)


#
# Generic routines
#


def _batch_score(X: Tensor, surrogate: torch.nn.Module, bsize: int) -> Tensor:
    """Batch surrogate scoring with mini-batches for efficiency."""
    n = len(X)
    S = [
        torch.atleast_1d(surrogate(X[list(b), :]))
        for b in batched(range(n), bsize)
    ]
    return torch.concat(S)


def _add_q_batch_dimension(X: Tensor) -> Tensor:
    """Insert a singleton q-batch dimension expected by BoTorch acquisitions."""
    return X.unsqueeze(dim=1)


def _adapt_acquisition(acqfn: Callable) -> Callable:
    """Adapt different acquisition call signatures to a common 3-arg form.

    Returns a wrapper that accepts ``(X, U, logpX)`` and dispatches to:
    (1) plain acquisitions: ``acq(X[q_batch_dim_added])``,
    (2) marginal acquisitions: ``acq(X[q_batch], logpX)``,
    (3) preference acquisitions: ``acq(X[q_batch], U, logpX)``.
    """
    sig = signature(acqfn.forward)
    nparams = len(sig.parameters)
    if nparams == 1:
        return lambda X, _1, _2: acqfn(_add_q_batch_dimension(X))
    elif nparams == 2:
        if not isinstance(acqfn, MarginalAcquisition):
            raise ValueError(
                f"{acqfn} has two args but not MarginalAcquisition"
            )
        return lambda X, _1, logpX: acqfn(_add_q_batch_dimension(X), logpX)
    elif nparams == 3:
        if not isinstance(acqfn, PreferenceAcquisition):
            raise ValueError(
                f"{acqfn} has three args but not PreferenceAcquisition"
            )
        return lambda X, U, logpX: acqfn(_add_q_batch_dimension(X), U, logpX)
    else:
        raise ValueError(f"{acqfn} has an invalid number of input arguments")


def _sample_proposal(
    proposal_distribution: SearchDistribution | PreferenceSearchDistribution,
    samples: Optional[int] = None,
    with_gradients: bool = False,
) -> Tuple[Tensor, Tensor | None, Tensor]:
    """Sample from a proposal distribution, optionally enabling gradients.

    Parameters
    ----------
    proposal_distribution : SearchDistribution | PreferenceSearchDistribution
        Distribution to sample from.
    samples : int, optional
        Number of samples to draw.
    with_gradients : bool, default=False
        If ``True``, keeps gradients through the sampling op (when supported).

    Returns
    -------
    (Tensor, Optional[Tensor], Tensor)
        ``(X, U_or_None, logqX)`` consistent with the proposal type.
    """
    proposal_distribution.eval()
    if with_gradients:
        result = proposal_distribution(samples=samples)
    else:
        with torch.no_grad():
            result = proposal_distribution(samples=samples)
    proposal_distribution.train()
    if len(result) == 2:
        X, logqX = result
        return X, None, logqX
    return result
