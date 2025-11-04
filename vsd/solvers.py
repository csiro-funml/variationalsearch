"""Solver interfaces for VSD and poli compatibility.

See https://machinelearninglifescience.github.io/poli-docs/contributing/a_new_solver.html
"""

import logging
import typing as T
from abc import ABC
from copy import deepcopy

import numpy as np
import torch
import torch.nn.functional as fnn
from botorch.utils.multi_objective.hypervolume import infer_reference_point
from numpy import ndarray
from poli.core.abstract_black_box import AbstractBlackBox
from poli_baselines.core.step_by_step_solver import StepByStepSolver
from poli_baselines.solvers.simple.random_mutation import RandomMutation
from torch import Tensor


from vsd.acquisition import (
    AcquisitionFunction,
    CbASAcquisition,
    LogPIClassifierAcquisition,
    MarginalAcquisition,
    PreferenceAcquisition,
    VariationalPreferenceAcquisition,
    VariationalSearchAcquisition,
)
from vsd.condproposals import (
    ConditionalSearchDistribution,
    CondMutationProposal,
    PreferenceSearchDistribution,
)
from vsd.cpe import (
    ClassProbabilityModel,
    PreferenceClassProbabilityModel,
    fit_cpe_labels,
    make_contrastive_alignment_data,
)
from vsd.generation import (
    generate_candidates_eda,
    generate_candidates_iw,
    generate_candidates_reinforce,
)
from vsd.preferences import EmpiricalPreferences
from vsd.proposals import (
    AutoRegressiveSearchDistribution,
    SearchDistribution,
    SequenceSearchDistribution,
    MaskedSearchDistribution,
    fit_ml,
)
from vsd.labellers import Labeller, Threshold, ParetoFront

LOG = logging.getLogger(name=__name__)


#
# Base Solvers
#


class _VariationalSolver(ABC, StepByStepSolver):
    """Variational solver for single-objective optimisation.

    The high-level loop:

    1. Fit a class-probability estimator (CPE) separating positive examples.
    2. Optionally fit a prior distribution to avoid overfitting.
    3. Fit a variational search distribution approximating the posterior over
       promising designs.
    4. Sample variational search distribution for candidates.

    Parameters
    ----------
    black_box : AbstractBlackBox
        Objective evaluated by the solver.
    alphabet : Sequence
        Alphabet used to tokenise sequences.
    x0 : ndarray
        Initial designs.
    y0 : ndarray or None
        Initial objective evaluations. If ``None`` they are queried from
        ``black_box``.
    labeller : callable or Labeller
        Labelling strategy turning objective values into binary outcomes.
    cpe : ClassProbabilityModel
        Class-probability estimator.
    vdistribution : SequenceSearchDistribution | AutoRegressiveSearchDistribution
        Variational search distribution to optimise.
    prior : SequenceSearchDistribution | AutoRegressiveSearchDistribution | None, optional
        Optional prior distribution to regularise the search.
    bsize : int, default=128
        Batch size used when generating candidates.
    device : str | torch.device, default="cpu"
        Device on which to perform optimisation.
    cpe_options : dict, optional
        Options forwarded to the CPE fitting routine.
    prior_options : dict, optional
        Options for fitting the prior distribution.
    vdist_options : dict, optional
        Options for optimising the variational proposal.
    cpe_validation_prop : float, default=0
        Validation proportion when fitting the CPE.
    prior_val_prop : float, default=0
        Validation proportion when fitting the prior via ML.
    topk_selection : bool, default=False
        If ``True``, select top-K candidates after each iteration.
    seed : int, optional
        Random seed for reproducibility.
    acq_fn_kwargs : dict, optional
        Additional keyword arguments for acquisition construction.
    """

    name: str
    optim: T.Callable
    vacquisition: type[AcquisitionFunction | MarginalAcquisition]

    def __init__(
        self,
        black_box: AbstractBlackBox,
        alphabet: T.Sequence,
        x0: ndarray,
        y0: T.Optional[ndarray],
        labeller: T.Callable[[Tensor], Tensor] | Labeller,
        cpe: ClassProbabilityModel,
        vdistribution: (
            SequenceSearchDistribution | AutoRegressiveSearchDistribution
        ),
        prior: T.Optional[
            SequenceSearchDistribution | AutoRegressiveSearchDistribution
        ] = None,
        bsize: int = 128,
        device: str | torch.device = "cpu",
        cpe_options: T.Optional[T.Dict[str, T.Any]] = None,
        prior_options: T.Optional[T.Dict[str, T.Any]] = None,
        vdist_options: T.Optional[T.Dict[str, T.Any]] = None,
        cpe_validation_prop: float = 0,
        prior_val_prop: float = 0,
        topk_selection: bool = False,
        seed: T.Optional[int] = None,
        acq_fn_kwargs: T.Optional[dict] = None,
    ):
        super().__init__(black_box, x0, y0 if y0 is not None else black_box(x0))
        self.labeller = labeller
        self.cpe = cpe.to(device)
        self.vdistribution = vdistribution.to(device)
        self.prior = prior.to(device) if prior is not None else prior
        self.bsize = bsize
        self.device = device
        self.cpe_validation_prop = cpe_validation_prop
        self.prior_val_prop = prior_val_prop
        self.topk_selection = topk_selection
        self.seed = seed
        self.acq_fn_kwargs = {} if acq_fn_kwargs is None else acq_fn_kwargs

        self.fit = False  # 0th round fitting flag
        self.acq = LogPIClassifierAcquisition(model=self.cpe)

        self.cpe_options = setdefaults(
            cpe_options,
            dict(
                optimizer_options=dict(lr=1e-3, weight_decay=1e-5),
                stop_options=dict(miniter=1000),
                batch_size=32,
            ),
        )
        self.prior_options = setdefaults(
            prior_options,
            dict(
                optimizer_options=dict(lr=1e-3, weight_decay=1e-5),
                batch_size=32,
            ),
        )
        self.vdist_options = setdefaults(
            vdist_options,
            dict(
                optimizer_options=dict(lr=1e-4),
                gradient_samples=256,
            ),
        )

        # Tokenize/de-tokenize
        self._s_to_i = {s: i for i, s in enumerate(alphabet)}
        self._i_to_s = {i: s for i, s in enumerate(alphabet)}

    def next_candidate(self) -> ndarray:
        # Tokenize and convert to tensors
        x = seq2int(self.history["x"], self._s_to_i)
        y = torch.tensor(np.concatenate(self.history["y"], axis=0)).squeeze()

        x = x.to(self.device)
        y = y.float().to(self.device)

        # Updates
        LOG.info(f"Round {self.iteration}.")

        # If prior is not fitted, then fit it, and lock off grads :-)
        if self.prior is None:
            LOG.info("Fitting prior and initial variational distribution ...")
            self.prior = fit_prior_and_model(
                model=self.vdistribution,
                x=x,
                seed=self.seed,
                prior_options=self.prior_options,
                val_prop=self.prior_val_prop,
            )

        LOG.info("Fitting CPE ...")
        z = self.labeller(y).float()
        LOG.info(
            f"Label count = {z.sum()}, "
            f"proportion = {z.mean().detach().cpu().numpy():.3f}."
        )
        if not self.fit and self.cpe_validation_prop > 0:
            LOG.info("Using hold out validation for the 0th round ...")
            tidx, vidx = train_val_split(z, self.cpe_validation_prop)
            x_cpe, z_cpe = x[tidx], z[tidx]
            x_val, z_val = x[vidx], z[vidx]
            callback = _val_callback
            stop_using_vloss = True
        else:
            callback = _callback
            x_cpe, z_cpe = x, z
            x_val, z_val = None, None
            stop_using_vloss = False

        fit_cpe_labels(
            self.cpe,
            X=x_cpe,
            z=z_cpe,
            X_val=x_val,
            z_val=z_val,
            device=self.device,
            callback=callback,
            seed=self.seed,
            stop_using_val_loss=stop_using_vloss,
            **self.cpe_options,
        )

        if isinstance(self.labeller, Threshold):
            maxy = y.max()
            thresh = self.labeller.best_f
            LOG.info(f"Threshold = {thresh:.3f}, max y = {maxy:.3f}.")

        # Update best sequences to mutate
        if isinstance(self.vdistribution, MaskedSearchDistribution):
            LOG.info("Update seed sequences in proposal ...")
            xb = x[z == 1]
            self.vdistribution.set_seeds(xb)
            if isinstance(self.prior, MaskedSearchDistribution):
                LOG.info("Update seed sequences in prior ...")
                self.prior.set_seeds(xb)

        LOG.info(f"Optimizing {self.name} acquisition function ...")
        vacq = self.vacquisition(self.acq, self.prior, **self.acq_fn_kwargs)
        vacq = vacq.to(self.device)
        xcand, _ = type(self).optim(
            acquisition_function=vacq,
            proposal_distribution=self.vdistribution,
            candidate_samples=self.bsize,
            callback=_grad_callback,
            **self.vdist_options,
        )
        self.fit = True

        if self.topk_selection:
            samples = self.vdist_options.get("gradient_samples", 512)
            Xs = self.vdistribution.sample(torch.Size([samples]))
            asort = torch.argsort(self.cpe(Xs), descending=True)
            xcand = Xs[asort[: self.bsize]]

        assert len(xcand) == self.bsize, "Wrong candidate size"
        return int2seq(xcand, self._i_to_s)


class _MooVariationalSolver(ABC, StepByStepSolver):
    """Variational solver for multi-objective optimisation with preferences.

    The loop:

    1. Fit a Pareto-front CPE to identify non-dominated points.
    2. Fit a preference CPE that scores candidate/context pairs.
    3. Fit the preference distribution using the normalised targets.
    4. Fit a variational search distribution approximating the posterior over
       promising designs.
    5. Sample preference distribution and variational search distribution for
       candidates.

    Parameters
    ----------
    black_box : AbstractBlackBox
        Objective evaluated by the solver.
    alphabet : Sequence
        Alphabet used to tokenise sequences.
    x0 : ndarray
        Initial designs.
    y0 : ndarray or None
        Initial objective evaluations. If ``None`` they are queried from
        ``black_box``.
    labeller : callable | ParetoFront
        Labelling strategy for Pareto membership.
    pareto_cpe : PreferenceClassProbabilityModel
        CPE predicting non-dominance.
    preference_cpe : PreferenceClassProbabilityModel
        CPE modelling user/stochastic preferences.
    vdistribution : PreferenceSearchDistribution
        Conditional search distribution ``q(X | U)``.
    prior : SearchDistribution | None, optional
        Optional prior distribution.
    ref : Tensor, optional
        Reference point for hypervolume estimation. Defaults to inferred.
    bsize : int, default=128
        Batch size used when generating candidates.
    device : str | torch.device, default="cpu"
        Device on which to perform optimisation.
    par_cpe_options, pre_cpe_options, prior_options, vdist_options, pref_options : dict, optional
        Keyword arguments forwarded to the respective fitting routines.
    fit_only_pareto_directions : bool, default=True
        If ``True``, update the preference distribution using Pareto-positive
        directions only.
    cpe_validation_prop : float, default=0
        Validation proportion when fitting CPEs in the first round.
    prior_val_prop : float, default=0
        Validation proportion when fitting the conditional prior.
    topk_selection : bool, default=False
        If ``True``, select top-K candidates after each iteration.
    seed : int, optional
        Random seed for reproducibility.
    acq_fn_kwargs : dict, optional
        Additional keyword arguments for acquisition construction.
    """

    name: str
    optim: T.Callable
    vacquisition: type[PreferenceAcquisition]

    def __init__(
        self,
        black_box: AbstractBlackBox,
        alphabet: T.Sequence,
        x0: ndarray,
        y0: T.Optional[ndarray],
        labeller: T.Callable[[Tensor], Tensor] | ParetoFront,
        pareto_cpe: PreferenceClassProbabilityModel,
        preference_cpe: PreferenceClassProbabilityModel,
        vdistribution: PreferenceSearchDistribution,
        prior: T.Optional[SearchDistribution] = None,
        ref: T.Optional[Tensor] = None,
        bsize: int = 128,
        device: str | torch.device = "cpu",
        par_cpe_options: T.Optional[T.Dict[str, T.Any]] = None,
        pre_cpe_options: T.Optional[T.Dict[str, T.Any]] = None,
        prior_options: T.Optional[T.Dict[str, T.Any]] = None,
        vdist_options: T.Optional[T.Dict[str, T.Any]] = None,
        pref_options: T.Optional[T.Dict[str, T.Any]] = None,
        fit_only_pareto_directions: bool = True,
        cpe_validation_prop: float = 0,
        prior_val_prop: float = 0,
        topk_selection: bool = False,
        seed: T.Optional[int] = None,
        acq_fn_kwargs: T.Optional[T.Dict] = None,
    ):
        super().__init__(black_box, x0, y0 if y0 is not None else black_box(x0))
        self.device = device
        self.seed = seed
        self.bsize = bsize
        self.vdistribution = vdistribution.to(device)
        self.pareto_cpe = pareto_cpe.to(device)
        self.preference_cpe = preference_cpe.to(device)
        self.prior = prior.to(device) if prior is not None else prior
        self.labeller = labeller if labeller is not None else ParetoFront()
        self.ref = ref
        self.fit_only_pareto_directions = fit_only_pareto_directions
        self.topk_selection = topk_selection
        self.cpe_validation_prop = cpe_validation_prop
        self.prior_val_prop = prior_val_prop
        self.acq_fn_kwargs = {} if acq_fn_kwargs is None else acq_fn_kwargs

        self.fit = False
        self.eps = 1e-6
        self._s_to_i = {s: i for i, s in enumerate(alphabet)}
        self._i_to_s = {i: s for i, s in enumerate(alphabet)}

        cpe_defaults = dict(
            optimizer_options=dict(lr=1e-3, weight_decay=1e-5),
            stop_options=dict(miniter=1000),
            batch_size=32,
        )
        self.par_cpe_options = setdefaults(par_cpe_options, cpe_defaults)
        self.pre_cpe_options = setdefaults(pre_cpe_options, cpe_defaults)

        self.pref_options = setdefaults(
            pref_options,
            dict(
                optimizer_options=dict(lr=1e-3, weight_decay=1e-8),
                stop_options=dict(k=0.5),
                batch_size=32,
            ),
        )

        self.prior_options = setdefaults(
            prior_options,
            dict(
                optimizer_options=dict(lr=1e-3, weight_decay=1e-5),
                batch_size=32,
            ),
        )

        self.vdist_options = setdefaults(
            vdist_options,
            dict(
                optimizer_options=dict(lr=1e-4),
                gradient_samples=256,
            ),
        )

    def next_candidate(self) -> ndarray:
        x = seq2int(self.history["x"], self._s_to_i)
        y = torch.tensor(np.concatenate(self.history["y"], axis=0))

        x = x.to(self.device)
        y = y.float().to(self.device)

        LOG.info(f"Round {self.iteration}.")

        # If prior is not fitted, then fit it, and lock off grads :-)
        if self.prior is None:
            LOG.info("Fitting prior and initial variational distribution ...")
            self.prior = fit_prior_and_model(
                model=self.vdistribution.cproposal,
                x=x,
                seed=self.seed,
                prior_options=self.prior_options,
                val_prop=self.prior_val_prop,
            )

        LOG.info("Computing training preference vectors ...")
        if self.ref is None:
            self.ref = infer_reference_point(y)
        self.ref = self.ref.to(y.device)
        u = fnn.normalize(y - self.ref, p=2, dim=1, eps=self.eps)

        LOG.info("Fitting Pareto CPE ...")
        z = self.labeller(y).float()
        LOG.info(
            f"Label count = {z.sum()}, "
            f"proportion = {z.mean().detach().cpu().numpy():.3f}"
        )
        self._fit_cpe(self.pareto_cpe, x, z, u)

        LOG.info("Fitting Preference CPE ...")
        xa, ua, za = make_contrastive_alignment_data(x, u)
        self._fit_cpe(self.preference_cpe, xa, za, ua)

        LOG.info("Updating preference distribution ...")
        Uz = (
            u[z == 1, :]
            if (self.fit_only_pareto_directions and self.fit)
            else u
        )
        if isinstance(self.vdistribution.preference, EmpiricalPreferences):
            self.vdistribution.preference.set_preferences(Uz)
        else:
            fit_ml(
                self.vdistribution.preference,
                Uz,
                callback=_callback,
                **self.pref_options,
            )

        # Update best sequences to mutate
        if isinstance(self.vdistribution.cproposal, CondMutationProposal):
            LOG.info("Update seed sequences in proposal ...")
            xb, ub = x[z == 1], u[z == 1]
            self.vdistribution.cproposal.set_seeds(xb, ub)
            if isinstance(self.prior, CondMutationProposal):
                LOG.info("Update seed sequences in prior ...")
                self.prior.set_seeds(xb, ub)
            if isinstance(self.prior, MaskedSearchDistribution):
                LOG.info("Update seed sequences in prior ...")
                self.prior.set_seeds(xb)

        LOG.info(f"Optimizing {self.name} acquisition function ...")
        vacq = self.vacquisition(
            pareto_model=self.pareto_cpe,
            pref_model=self.preference_cpe,
            prior_dist=self.prior,
            **self.acq_fn_kwargs,
        )
        vacq = vacq.to(self.device)
        xcand, _ = type(self).optim(
            acquisition_function=vacq,
            proposal_distribution=self.vdistribution,
            candidate_samples=self.bsize,
            callback=_grad_callback,
            **self.vdist_options,
        )
        self.fit = True

        if self.topk_selection:
            samples = self.vdist_options.get("gradient_samples", 512)
            Xs, Us = self.vdistribution.sample(torch.Size([samples]))
            asort = torch.argsort(self.pareto_cpe(Xs, Us), descending=True)
            xcand = Xs[asort[: self.bsize]]

        assert len(xcand) == self.bsize, "Wrong candidate size"
        return int2seq(xcand, self._i_to_s)

    def _fit_cpe(
        self,
        cpe: PreferenceClassProbabilityModel,
        x: Tensor,
        z: Tensor,
        u: Tensor,
    ):
        if not self.fit and self.cpe_validation_prop > 0:
            LOG.info("Using hold out validation for the 0th round ...")
            tidx, vidx = train_val_split(z, self.cpe_validation_prop)
            x_cpe, u_cpe, z_cpe = x[tidx], u[tidx], z[tidx]
            x_val, u_val, z_val = x[vidx], u[vidx], z[vidx]
            callback = _val_callback
            stop_using_vloss = True
        else:
            callback = _callback
            x_cpe, u_cpe, z_cpe = x, u, z
            x_val, u_val, z_val = None, None, None
            stop_using_vloss = False

        fit_cpe_labels(
            cpe,
            X=x_cpe,
            z=z_cpe,
            U=u_cpe,
            X_val=x_val,
            z_val=z_val,
            U_val=u_val,
            device=self.device,
            seed=self.seed,
            callback=callback,
            stop_using_val_loss=stop_using_vloss,
            **self.par_cpe_options,
        )


#
# Solver Interfaces
#


class VSDSolver(_VariationalSolver):

    name = "VSD"
    optim = generate_candidates_reinforce
    vacquisition = VariationalSearchAcquisition


class VSDSolverIW(_VariationalSolver):

    name = "VSD"
    optim = generate_candidates_iw
    vacquisition = VariationalSearchAcquisition


class CbASSolver(_VariationalSolver):

    name = "CbAS"
    optim = generate_candidates_eda
    vacquisition = CbASAcquisition


class AGPSSolver(_MooVariationalSolver):

    name = "AGPS"
    optim = generate_candidates_reinforce
    vacquisition = VariationalPreferenceAcquisition


class AGPSSolverIW(_MooVariationalSolver):

    name = "AGPS"
    optim = generate_candidates_iw
    vacquisition = VariationalPreferenceAcquisition


#
# Other solvers
#


class RandomPadMutation(RandomMutation):
    """Random mutation respecting immutable pad tokens."""

    def __init__(
        self,
        black_box: AbstractBlackBox,
        x0: np.ndarray,
        y0: np.ndarray,
        n_mutations: int = 1,
        top_k: int = 1,
        batch_size: int = 1,
        greedy: bool = True,
        alphabet: list[str] | None = None,
        tokenizer: T.Callable[[str], list[str]] | None = None,
        pad_char: str = "-",
    ):
        super().__init__(
            black_box=black_box,
            x0=x0,
            y0=y0,
            n_mutations=n_mutations,
            top_k=top_k,
            batch_size=batch_size,
            greedy=greedy,
            alphabet=alphabet,
            tokenizer=tokenizer,
        )
        self.pad_char = pad_char
        if pad_char not in self.alphabet_without_empty:
            self.alphabet_without_empty += [self.pad_char]
        self.pad_token = self.alphabet_without_empty.index(self.pad_char)
        self.ok_tokens, self.ok_alphabet = zip(
            *[
                (i, a)
                for i, a in enumerate(self.alphabet_without_empty)
                if a != self.pad_char
            ]
        )

    def _next_candidate(self) -> np.ndarray:
        """
        Returns the next candidate solution
        after checking the history.

        In this case, the RandomMutation solver
        simply returns a random mutation of the
        best performing solution so far.
        """
        if self.greedy:
            # Get the best performing solution(s) so far
            best_xs = self.get_best_solution(top_k=self.top_k)

            # Get a random sample from the top-k
            best_x = best_xs[np.random.choice(len(best_xs))]
        else:
            xs, _ = self.get_history_as_arrays()
            random_index = np.random.choice(len(xs))
            best_x = xs[random_index]

        # Perform a random mutation
        # TODO: this assumes that x has shape [1, L],
        # what happens with batches? So far, POLi is
        # implemented without batching in mind.
        next_x = best_x.copy().reshape(1, -1)

        if next_x.dtype.kind in ("i", "f"):
            okpos = np.nonzero(next_x.flatten() != self.pad_token)[0]
        elif next_x.dtype.kind in ("U", "S"):
            okpos = np.nonzero(next_x.flatten() != self.pad_char)[0]
        else:
            raise ValueError(
                f"Unknown dtype for the input: {next_x.dtype}. "
                "Only integer, float and unicode dtypes are supported."
            )

        for _ in range(self.n_mutations):
            pos = np.random.choice(okpos)
            while next_x[0][pos] == "":
                pos = np.random.choice(okpos)

            if next_x.dtype.kind in ("i", "f"):
                mutant = np.random.choice(self.ok_tokens)
            else:
                mutant = np.random.choice(self.ok_alphabet)

            next_x[0][pos] = mutant

        return next_x


#
# Useful methods
#


def seq2int(S: T.List[T.Sequence[str]], mapping: T.Dict[str, int]) -> Tensor:
    S = [to_char_list(s) for s in S]
    Xi = np.vectorize(mapping.__getitem__)(np.asarray(S).squeeze())
    return torch.tensor(Xi).long()


def int2seq(X: Tensor, mapping: T.Dict[int, str]) -> ndarray:
    S = np.vectorize(mapping.__getitem__)(X.detach().cpu().numpy())
    return S


def seq2int_padded(
    S: T.List[T.Sequence[str]],
    mapping: T.Dict[str, int],
    pad_token: str = "-",
) -> Tensor:
    """Tokenise sequences and pad them to a common length.

    Parameters
    ----------
    S : list[Sequence[str]]
        Input sequences.
    mapping : dict[str, int]
        Token-to-index mapping.
    pad_token : str, default="-"
        Token used for padding.

    Returns
    -------
    Tensor
        Long tensor of shape ``(batch, max_len)`` containing padded indices.
    """
    # ensure pad_token in mapping
    if pad_token not in mapping:
        mapping[pad_token] = max(mapping.values()) + 1
    pad_idx = mapping[pad_token]

    # convert each sequence to a tensor of indices
    tensors = []
    for seq in S:
        idxs = [mapping[s] for s in to_char_list(seq)]
        tensors.append(torch.tensor(idxs, dtype=torch.long))

    # pad into a rectangular tensor
    padded = torch.nn.utils.rnn.pad_sequence(
        tensors, batch_first=True, padding_value=pad_idx
    )
    return padded


def to_char_list(seq: T.Union[np.ndarray, T.Sequence[str], str]) -> T.List[str]:
    """Normalise an input sequence to a list of single-character tokens.

    Supports strings, iterables of strings, and NumPy arrays of either form.

    Parameters
    ----------
    seq : array-like or str
        Sequence to normalise.

    Returns
    -------
    list[str]
        Flattened list of single-character tokens.
    """
    arr = np.atleast_1d(seq)  # 1-D or higher
    arr = np.squeeze(arr)  # remove size-1 dims
    # If it’s now a 1-D array of chars, return that
    if (
        isinstance(arr, np.ndarray)
        and arr.ndim == 1
        and all(len(x) == 1 for x in arr.tolist())
    ):
        return arr.tolist()
    # Otherwise convert to one string and split
    s = arr.item() if isinstance(arr, np.ndarray) else str(arr)
    return list(s)


def int2seq_unpad(
    X: Tensor, mapping: T.Dict[int, str], pad_token: str = "-"
) -> np.ndarray:
    """Convert padded integer sequences back to tokens and strip padding.

    Parameters
    ----------
    X : Tensor
        Padded integer tensor.
    mapping : dict[int, str]
        Index-to-token mapping.
    pad_token : str, default="-"
        Padding token to remove.

    Returns
    -------
    numpy.ndarray
        Array of unpadded token sequences.
    """
    # ensure pad_token in mapping
    if pad_token not in mapping.values():
        mapping[max(mapping.keys()) + 1] = pad_token

    str_arr = np.vectorize(mapping.__getitem__)(X.detach().cpu().numpy())

    # For each row, find where pad_token first appears
    result = []
    for row in str_arr:
        # find pad positions
        pad_positions = np.where(row == pad_token)[0]
        end = pad_positions[0] if pad_positions.size > 0 else len(row)
        result.append(row[:end])

    # Filter out length 0 results
    result = [seq for seq in result if len(seq) > 0]

    # Return rectangular or ragged array
    if all([len(s) == len(result[0]) for s in result]):
        return np.array(result)
    else:
        return np.array([np.array(["".join(r)]) for r in result])


def setdefaults(
    opt: T.Dict[str, T.Any] | None, defaults: T.Dict[str, T.Any]
) -> T.Dict[str, T.Any]:
    if opt is not None:
        defaults = deepcopy(defaults)  # In case this is re-used
        defaults.update(opt)
    return defaults


def fit_prior_and_model(
    model: SearchDistribution | ConditionalSearchDistribution,
    x: Tensor,
    seed: T.Optional[int],
    prior_options: dict,
    val_prop: float,
) -> SearchDistribution | torch.distributions.Distribution:

    # Check and deal with cases where the model cannot be used as its own prior
    if not model.prior_same_class:
        prior = model.get_compatible_prior()  # New instance
        if isinstance(prior, torch.distributions.Distribution):
            # Ensure posterior dropout is disabled even on early return
            if hasattr(model, "set_dropout_p"):
                LOG.info("Disabling posterior dropout ...")
                model.set_dropout_p(p=0.0)
            return prior
    else:
        prior = model  # Pointer not copy

    fit_ml(
        prior,
        x,
        callback=_val_callback if val_prop > 0 else _callback,
        device=x.device,
        seed=seed,
        val_proportion=val_prop,
        **prior_options,
    )

    # Now copy prior if required
    if model.prior_same_class:
        LOG.info("Copying prior model ...")
        prior = deepcopy(prior)

    # Or copy prior parameters to initialise model
    elif hasattr(model, "load"):
        LOG.info("Copying prior weights to posterior ...")
        model.load(prior)

    # Fix prior
    LOG.info("Fixing prior weights")
    for p in prior.parameters():
        p.requires_grad = False

    # Test model still learnable
    for p in model.parameters():
        assert p.requires_grad

    # Make sure we are not using dropout in subsequent training
    if hasattr(prior, "set_dropout_p"):
        LOG.info("Disabling prior dropout ...")
        prior.set_dropout_p(p=0.0)

    if hasattr(model, "set_dropout_p"):
        LOG.info("Disabling posterior dropout ...")
        model.set_dropout_p(p=0.0)

    # Prior should be used in eval/inference mode downstream
    if hasattr(prior, "eval"):
        prior.eval()

    return prior


def train_val_split(
    z: Tensor, val_prop: float, overlap_if_class_size_lt: int = 20
) -> T.Tuple[Tensor, Tensor]:
    """Binary-label stratified train/val index split.

    - Treats any positive value as label 1, else 0.
    - Keeps class proportions in the validation set.
    - If a class has fewer than ``overlap_if_class_size_lt`` members, allow
      reuse across splits (i.e., the small class can appear in both train and val).
    """
    z = z.flatten()
    device = z.device

    if val_prop <= 0:
        all_idx = torch.arange(z.numel(), device=device)
        return all_idx, torch.empty(0, dtype=torch.long, device=device)

    # Binary labels: cast to boolean then split
    zb = z.to(torch.bool)
    pos_idx = torch.nonzero(zb, as_tuple=False).squeeze(-1)
    neg_idx = torch.nonzero(~zb, as_tuple=False).squeeze(-1)

    def split_class(idx: Tensor) -> tuple[Tensor, Tensor]:
        m = idx.numel()
        if m == 0:
            return idx, idx
        n_val = int(round(float(m) * float(val_prop)))
        n_val = max(1, min(n_val, m))
        perm = idx[torch.randperm(m, device=device)]
        val_c = perm[:n_val]
        if m < overlap_if_class_size_lt:
            train_c = idx  # allow overlap for small classes
        else:
            train_c = perm[n_val:]
        return train_c, val_c

    train_pos, val_pos = split_class(pos_idx)
    train_neg, val_neg = split_class(neg_idx)

    train = torch.cat([t for t in (train_pos, train_neg) if t.numel() > 0])
    val = torch.cat([v for v in (val_pos, val_neg) if v.numel() > 0])

    if train.numel() > 0:
        train = train[torch.randperm(train.numel(), device=device)]
    if val.numel() > 0:
        val = val[torch.randperm(val.numel(), device=device)]

    return train, val


def _callback(it, loss, *args, log_iters=100):
    if (it % log_iters) == 0:
        LOG.info(f"  It: {it}, Loss = {loss:.3f}")


def _val_callback(it, loss, vloss, log_iters=100):
    if (it % log_iters) == 0:
        LOG.info(f"  It: {it}, Loss = {loss:.3f}, Valid. loss = {vloss:.3f}")


def _grad_callback(it, loss, grad, log_iters=100):
    if (it % log_iters) == 0:
        gmean = sum([g.detach().mean() for g in grad if g is not None])
        gmean = gmean / len(grad)
        LOG.info(f"  It: {it}, Loss = {loss:.3e}, Mean gradient = {gmean:.3e}")
