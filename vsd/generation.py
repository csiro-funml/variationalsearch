"""BoTorch compatible candidate generation routines."""

import time
import warnings
from copy import deepcopy
from inspect import signature
from itertools import batched
from typing import Callable, Dict, NoReturn, Optional, Tuple, Union

import numpy as np
import torch
from botorch.acquisition import AcquisitionFunction
from botorch.logging import _get_logger
from botorch.optim.stopping import ExpMAStoppingCriterion
from polyleven import levenshtein
from scipy.spatial import cKDTree
from torch import IntTensor, Tensor
from torch.optim import Optimizer

from vsd.proposals import (
    SearchDistribution,
    TransitionSearchDistribution,
    clip_gradients,
)
from vsd.utils import SequenceTensor

logger = _get_logger()


def generate_candidates_reinforce(
    acquisition_function: AcquisitionFunction,
    proposal_distribution: SearchDistribution,
    cv_smoothing: float = 0.7,
    optimizer: Optimizer = torch.optim.Adam,
    optimizer_options: Optional[Dict[str, Union[float, str]]] = None,
    stop_options: Optional[Dict[str, Union[float, str]]] = None,
    callback: Optional[
        Callable[[int, Tensor, Tuple[Tensor, ...]], NoReturn]
    ] = None,
    timeout_sec: Optional[float] = None,
    gradient_samples: Optional[int] = None,
    candidate_samples: Optional[int] = None,
) -> Tuple[Tensor, Tensor]:
    r"""Generate a set of candidates using a REINFORCE reparameterisation.

    Args:
        acquisition_function: Acquisition/black box function to be used.
        proposal_distribution: a SearchPosterior distribution to generate
            candidates from, and optimise using REINFFORCE.
        cv_smoothing: Control variate exponential average smoothing coefficient.
        optimizer (Optimizer): The pytorch optimizer to use to perform
            candidate search.
        opti_options: Options used to control the optimization. Includes
            maxiter: Maximum number of iterations
        stop_options: Options used to control the stopping criterion. Includes
            maxiter: Maximum number of iterations
        callback: A callback function accepting the current iteration, loss,
            and gradients as arguments. This function is executed after
            computing the loss and gradients, but before calling the optimizer.
        timeout_sec: Timeout (in seconds) for optimization. If provided,
            `gen_candidates_torch` will stop after this many seconds and return
            the best solution found so far.
        gradient_samples: Number of samples to draw from the proposal
            distribution for estimating the REINFORCE gradient.
        candidate_samples: Number of final candidate samples to return from the
            proposal distribution.

    Returns:
        2-element tuple containing

        - A set of generated candidates from the proposal distribution.
        - The acquisition value for each candidate.
    """
    acquisition_function = _adapt_acquisition(acquisition_function)
    start_time = time.monotonic()
    optimizer_options = optimizer_options or {"weight_decay": 0.0}
    stop_options = stop_options or {}

    # Set up the optimiser
    clip_gradients(proposal_distribution)
    params = list(proposal_distribution.parameters())
    _optimizer = optimizer(params=params, **optimizer_options)  # type: ignore

    i, cv = 0, 0
    stop = False
    stopping_criterion = ExpMAStoppingCriterion(**stop_options)  # type: ignore
    proposal_distribution.train()
    while not stop:
        X, logqX = proposal_distribution(samples=gradient_samples)

        # Reinforce does not differentiate through acq
        with torch.no_grad():
            acq = acquisition_function(X, logqX)

        if acq.ndim != logqX.ndim:
            raise RuntimeError(f"acq. dim:{acq.ndim} != logp dim:{logqX.ndim}")
        loss = -((acq - cv) * logqX).mean()  # Reinforce gradient loss

        with torch.no_grad():
            lossa = -acq.mean()  # actual loss
            cv = cv_smoothing * lossa + (1 - cv_smoothing) * cv  # baseline

        loss.backward()
        if callback:
            callback(i, lossa, [p.grad for p in params])

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
    proposal_distribution.eval()
    with torch.no_grad():
        Xcand, logqX = proposal_distribution(samples=candidate_samples)
        Xcand_acq = acquisition_function(Xcand, logqX)
    return Xcand, Xcand_acq


def generate_candidates_eda(
    acquisition_function: AcquisitionFunction,
    proposal_distribution: SearchDistribution,
    optimizer: Optimizer = torch.optim.Adam,
    optimizer_options: Optional[Dict[str, Union[float, str]]] = None,
    stop_options: Optional[Dict[str, Union[float, str]]] = None,
    callback: Optional[
        Callable[[int, Tensor, Tuple[Tensor, ...]], NoReturn]
    ] = None,
    timeout_sec: Optional[float] = None,
    gradient_samples: Optional[int] = None,
    candidate_samples: Optional[int] = None,
) -> Tuple[Tensor, Tensor]:
    r"""Generate a set of candidates using Estimation of Distribution (EDA).

    Args:
        acquisition_function: Acquisition/black box function to be used.
        proposal_distribution: a SearchDistribution to optimise using
            REINFORCE, this will be used to generate candidates from.
        optimizer (Optimizer): The pytorch optimizer to use to perform
            candidate search.
        opti_options: Options used to control the optimization. Includes
            maxiter: Maximum number of iterations
        stop_options: Options used to control the stopping criterion. Includes
            maxiter: Maximum number of iterations
        callback: A callback function accepting the current iteration, loss,
            and gradients as arguments. This function is executed after
            computing the loss and gradients, but before calling the optimizer.
        timeout_sec: Timeout (in seconds) for optimization. If provided,
            `gen_candidates_torch` will stop after this many seconds and return
            the best solution found so far.
        gradient_samples: Number of samples to draw from the posterior
            distribution for estimating the maximum likelihood gradient.
        candidate_samples: Number of final candidate samples to return from the
            proposal distribution.

    Returns:
        2-element tuple containing

        - A set of generated candidates from the proposal distribution.
        - The acquisition value for each candidate.
    """
    acquisition_function = _adapt_acquisition(acquisition_function)
    start_time = time.monotonic()
    optimizer_options = optimizer_options or {}
    stop_options = stop_options or {}

    # The posterior is the latest, best proposal
    posterior_distribution = deepcopy(proposal_distribution)

    # Make sure we are just evaluating gradients on the proposal
    for p in posterior_distribution.parameters():
        p.grad = None
        p.requires_grad = False

    # Set up the optimiser
    clip_gradients(proposal_distribution)
    params = list(proposal_distribution.parameters())
    _optimizer = optimizer(params=params, **optimizer_options)  # type: ignore

    # Draw samples once to optimise against
    if isinstance(posterior_distribution, TransitionSearchDistribution):
        X, logpX, X0 = posterior_distribution(
            samples=gradient_samples, return_X0=True
        )
        Xs = (X, X0)
    else:
        X, logpX = posterior_distribution(samples=gradient_samples)
        Xs = (X,)

    # EDA does not differentiate through acq
    with torch.no_grad():
        wght = acquisition_function(X, logpX)

    i = 0
    stop = False
    stopping_criterion = ExpMAStoppingCriterion(**stop_options)  # type: ignore
    proposal_distribution.train()
    while not stop:
        logqX = proposal_distribution.log_prob(*Xs)

        if wght.ndim != logqX.ndim:
            raise RuntimeError(
                f"acquisition dim, {wght.ndim} != logp dim, " f"{logqX.ndim}"
            )
        loss = -(wght * logqX).mean()  # EDA maximum likelihood

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
    proposal_distribution.eval()
    with torch.no_grad():
        Xcand, logqX = proposal_distribution(samples=candidate_samples)
        Xcand_acq = acquisition_function(Xcand, logqX)
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
    """Use the Proximal Exploration (PEX) algorithm for candidate generation.

    See:

    Ren, Z., Li, J., Din, F., Zhou, Y., Ma, J., & Peng, J. (2022, June).
        Proximal exploration for model-guided protein sequence design. In
        International Conference on Machine Learning (pp. 18520-18536). PMLR.
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
    """Use the AdaLead algorithm for candidate generation.

    See:

    Sinai, S., Wang, R., Whatley, A., Slocum, S., Locane, E., & Kelsic, E. D.
        (2020). AdaLead: A simple and robust adaptive greedy search algorithm
        for sequence design. arXiv preprint arXiv:2010.02141.
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
    s = "".join(np.char.mod("%d", x.detach().numpy()))
    return s


def _upper_convex_hull(d, f):
    """Computes the upper convex hull of a set of 2D points.

    Input: iterable sequences of (f, d) pairs representing the points.
    Output: a list of vertices of the convex hull in counter-clockwise order,
      starting from the vertex with the lexicographically smallest coordinates.
      indicies into the original array are also returned.
    Implements Andrew's monotone chain algorithm. O(n log n) complexity.

    From:
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
    m = len(x)
    xm = x.clone()
    p = torch.randint(m, size=[k_mutations])
    a = torch.randint(alphalen, size=[k_mutations])
    xm[p] = a
    return xm


def _recombine(X: SequenceTensor, recombination_rate: float) -> SequenceTensor:
    """Randomly recombine pairs of sequences."""
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
# Generaic routines
#


def _batch_score(X: Tensor, surrogate: torch.nn.Module, bsize: int) -> Tensor:
    n = len(X)
    S = [
        torch.atleast_1d(surrogate(X[list(b), :]))
        for b in batched(range(n), bsize)
    ]
    return torch.concat(S)


def _add_q_batch_dimension(X: Tensor) -> Tensor:
    return X.unsqueeze(dim=1)


def _adapt_acquisition(acqfn: Callable) -> Callable:
    sig = signature(acqfn.forward)
    nparams = len(sig.parameters)
    if nparams == 1:
        return lambda X, _: acqfn(_add_q_batch_dimension(X))
    elif nparams == 2:
        return lambda X, logpX: acqfn(_add_q_batch_dimension(X), logpX)
    else:
        raise ValueError(f"{acqfn} has an invalid number of input arguments")
