"""MOBO using the Ehrlich function and a sequence naturalness score.

This experiment is based off:
https://github.com/MachineLearningLifeScience/poli-baselines/blob/main/examples/07_running_lambo2_on_ehrlich/run.py
"""

import logging
import random
import sys
import time
from copy import deepcopy
from pathlib import Path

import click
import matplotlib as mpl
import numpy as np

mpl.use("Agg")
np.set_printoptions(suppress=True, precision=2)

import matplotlib.pyplot as plt
import torch
from botorch.utils.multi_objective.hypervolume import (
    Hypervolume,
    infer_reference_point,
)
from poli.core.black_box_information import BlackBoxInformation
from poli.core.util.abstract_observer import AbstractObserver
from poli.objective_repository import (
    EhrlichHoloProblemFactory,
    EhrlichProblemFactory,
)
from poli_baselines.solvers.bayesian_optimization.lambo2 import LaMBO2
from poli_baselines.solvers.simple.random_mutation import RandomMutation

from experiments.blackboxes import (
    CombinedBlackBoxFactory,
    ProtBertNaturalnessFactory,
)

from vsd.condproposals import (
    CondDTransformerProposal,
    CondLSTMProposal,
    PreferenceSearchDistribution,
    CondTransformerMutationProposal,
)
from vsd.cpe import CNNClassProbability, PreferenceCNNClassProbability
from vsd.preferences import MixtureUnitNormal
from vsd.proposals import (
    DTransformerProposal,
    TransformerMutationProposal,
    LSTMProposal,
)
from vsd.solvers import (
    AGPSSolver,
    AGPSSolverIW,
    CbASSolver,
    VSDSolver,
    VSDSolverIW,
    int2seq_unpad,
)
from vsd.labellers import ParetoAnnealed
from vsd.utils import is_non_dominated_strict

SEQLEN_SETTING = {
    15: dict(motif_length=3, n_motifs=2, quantization=3),
    32: dict(motif_length=4, n_motifs=3, quantization=4),
    64: dict(motif_length=4, n_motifs=4, quantization=4),
}
EMBEDDING_DIM = 64
LNETWORKS = 64
LLAYERS = 3
DNETWORKS = 128
NHEADS = 4
DLAYERS = 2
NUM_MUTATIONS = {15: 1, 32: 3, 64: 5}
PRIOR_DROPOUT = {15: 0.5, 32: 0.4, 64: 0.2}
START_PERCENTILE = 0.75
TOPK_SELECTION = False
PRIOR_VAL_PROB = 0
PRIOR_MAXITER = 1001
CPE_PARAMS = dict(
    ckernel=5,
    xkernel=3,
    xstride=2,
    cfilter_size=64,
    linear_size=128,
    embedding_dim=16,
    dropoutp=0.2,
    pos_encoding=True,
)

plt.rcParams.update(
    {
        "font.size": 12,
        "axes.labelsize": 14,
        "axes.titlesize": 16,
        "legend.fontsize": 10,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
    }
)


# Code from https://github.com/MachineLearningLifeScience/poli-baselines/blob/
#   main/examples/07_running_lambo2_on_ehrlich/simple_observer.py
class SimpleObserver(AbstractObserver):
    def __init__(self, logger: logging.Logger | None) -> None:
        self.x_s = []
        self.y_s = []
        self.logger = logger
        super().__init__()

    def initialize_observer(
        self,
        problem_setup_info: BlackBoxInformation,
        caller_info: object,
        seed: int,
    ) -> object: ...

    def observe(self, x: np.ndarray, y: np.ndarray, context=None) -> None:
        self.x_s.append(x)
        self.y_s.append(y)
        z = (
            is_non_dominated_strict(torch.Tensor(np.vstack(self.y_s)))
            .detach()
            .numpy()
            .squeeze()
        )
        if np.ndim(z) > 0:
            zb = z[-len(y) :].squeeze()
        else:
            zb = z
        if self.logger is not None:
            if np.ndim(zb) == 0:
                self.logger.info(f"Sequence: {''.join(x.squeeze())}")
                self.logger.info(f" fitness: {y}, is not dominated: {zb}")
                return
            for s, ys, zs in zip(x, y, zb):
                self.logger.info(f"Sequence: {''.join(s)}")
                self.logger.info(f" fitness: {ys}, is not dominated: {zs}")


def plot_hypervolume(
    obs: SimpleObserver, ref: torch.Tensor | None, ax: plt.Axes, tsize: int = 0
):
    y = torch.tensor(np.vstack(obs.y_s)).float()
    n = len(y)
    ref = infer_reference_point(y) if ref is None else ref
    hv = Hypervolume(ref_point=ref)
    hvres = []
    z = np.zeros(n)
    for i in range(tsize, n):
        z[:i] = is_non_dominated_strict(y[:i, :])
        hvres.append(hv.compute(y[z == 1, :]))
    ax.plot(range(len(hvres)), hvres, "b")
    ax.set_xlabel("Number of evaluations")
    ax.set_ylabel("Pareto Hypervolume")


def plot_pareto(obs: SimpleObserver, ax: plt.Axes, start_from: int = 0):
    y = torch.tensor(np.vstack(obs.y_s)).float()
    z = is_non_dominated_strict(y)
    ax.plot(*y.T, ".", label="all points")
    ax.plot(*y[z == 1, :].T, "xr", label="Pareto front")
    ax.set_xlabel("Ehrlich")
    ax.set_ylabel("Naturalness")
    ax.legend()


def plot_cond_prefences(
    obs: SimpleObserver,
    optim: AGPSSolverIW,
    ref: torch.Tensor | None,
    device: torch.device | str,
    ax: plt.Axes,
    samples=10,
):
    y = torch.tensor(np.vstack(obs.y_s)).float()
    ref = infer_reference_point(y) if ref is None else ref
    yplt = y - ref
    uf1 = torch.tensor([yplt[:, 0].max(), yplt[:, 1].min()])
    uf2 = torch.tensor([yplt[:, 0].median(), yplt[:, 1].median()])
    uf3 = torch.tensor([yplt[:, 0].min(), yplt[:, 1].max()])

    ysamps = []
    cols = plt.cm.inferno([0.1, 0.5, 0.9])
    start = (ref[0].item(), ref[1].item())
    ax.plot(*start, "ks", label="Reference point")
    ufns = []
    for uf, marker, col in zip([uf1, uf2, uf3], ["x", "+", "."], cols):
        ufn = torch.nn.functional.normalize(uf, p=2, dim=-1)
        ufns.append(ufn.cpu().numpy())
        ufs = torch.tile(ufn, (samples, 1)).to(device)
        xcand, _ = optim.vdistribution.cproposal(ufs)
        ycand = optim.black_box(int2seq_unpad(xcand, optim._i_to_s))
        ysamps.append(ycand)
        ax.plot(
            *ycand.T,
            marker,
            label=f"u = {ufn.numpy()}",
            c=col,
            markersize=10,
            alpha=0.7,
            mew=2,
        )
        dx, dy = uf[0].item() / 2, uf[1].item() / 2
        end = (start[0] + dx, start[1] + dy)
        ax.annotate(
            "",  # no text
            xy=end,  # arrow head at 'end'
            xytext=start,  # arrow tail at 'start'
            arrowprops={
                "arrowstyle": "->",
                "color": col,
                "lw": 2,
                "shrinkA": 0,
                "shrinkB": 0,
            },
        )
    ax.legend()
    ax.grid()
    ax.set_xlabel("$f_1$")
    ax.set_ylabel("$f_2$")
    return ufns, ysamps


@click.command()
@click.option(
    "--solver",
    type=click.Choice(
        [
            "agps-lstm",
            "agps-tfm",
            "agps-mtfm",
            "agps-lstm-rf",
            "agps-tfm-rf",
            "agps-mtfm-rf",
            "vsd-lstm",
            "vsd-tfm",
            "vsd-mtfm",
            "vsd-lstm-rf",
            "vsd-tfm-rf",
            "vsd-mtfm-rf",
            "cbas-lstm",
            "cbas-tfm",
            "cbas-mtfm",
            "lambo2",
            "rand",
        ]
    ),
    default="vsd-lstm",
    help="solver to run.",
)
@click.option(
    "--sequence-length",
    type=click.Choice([str(k) for k in SEQLEN_SETTING.keys()]),
    default="32",
    help="sequence length for the Ehrlich function.",
)
@click.option(
    "--max-iter", type=int, default=40, help="Maximum iterations to run."
)
@click.option(
    "--bsize",
    type=int,
    default=32,
    help="Batch size for black box evaluation.",
)
@click.option(
    "--tsize",
    type=int,
    default=128,
    help="Training dataset sized.",
)
@click.option(
    "--logdir",
    type=click.Path(file_okay=False),
    default="ehrlich_nat",
    help="log and results directory.",
)
@click.option(
    "--device", type=str, default="cpu", help="device to use for solver."
)
@click.option("--seed", type=int, default=42, help="random seed.")
@click.option(
    "--holo",
    is_flag=True,
    help="use the original holo ehrlich function implementation.",
)
@click.option(
    "--ref",
    type=(float, float),
    default=(-1, 0),
    help="Reference point for hypervolume computation",
)
@click.option(
    "--gsamples",
    type=int,
    default=256,
    help="Samples for gradient estimation.",
)
@click.option(
    "--startp",
    type=float,
    default=START_PERCENTILE,
    help="Start percentile for Pareto rank thresholding",
)
@click.option(
    "--priordropout",
    type=float,
    default=None,
    help="Prior dropout regularisation for the transformer models",
)
def main(
    solver,
    sequence_length,
    max_iter,
    bsize,
    tsize,
    logdir,
    device,
    seed,
    holo,
    ref,
    gsamples,
    startp,
    priordropout,
):
    # Setup logging
    solver_name = f"{solver}_{seed}"
    logdir = Path(logdir) / sequence_length
    logdir.mkdir(exist_ok=True, parents=True)
    logfile = logdir / Path(f"{solver_name}.log")
    logging.basicConfig(
        level=logging.INFO,
        handlers=[
            logging.FileHandler(logfile, mode="w"),
            logging.StreamHandler(),
        ],
    )
    log = logging.getLogger(name=__name__)

    # Log params
    log.info(
        f"solver: {solver}, sequence_length: {sequence_length},"
        f" max_iter: {max_iter}, bsize: {bsize}, tsize: {tsize},"
        f" logdir: {logdir}, device: {device}, seed: {seed}, holo: {holo}."
    )

    # Fix seeds
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if device != "cpu":
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # Setup problem
    sequence_length = int(sequence_length)
    log.info(
        f"Creating Ehrlich function: length {sequence_length} with"
        f" {SEQLEN_SETTING[sequence_length]}"
    )
    if holo:
        log.info("Using Holo bench Ehrlich implementation.")
        ehrlbb, _, x0 = _erhlich_holo(sequence_length, seed, tsize)
    else:
        log.info("Using built-in poli Ehrlich implementation.")
        ehrlbb, _, x0 = _erhlich_poli(sequence_length, seed, tsize)

    protbb = ProtBertNaturalnessFactory().create(device=device)
    black_box, y0 = CombinedBlackBoxFactory(
        blackboxes=[ehrlbb, protbb],
    ).create(x0)

    # Set up a history logger
    observer = SimpleObserver(logger=log)
    observer.observe(x0, y0)
    black_box.set_observer(observer)
    L = y0.shape[1]
    if ref is not None:
        ref = torch.tensor(ref)

    if solver not in ("lambo2", "rand"):
        alpha_len = len(black_box.alphabet)
        if priordropout is None:
            priordropout = PRIOR_DROPOUT[sequence_length]
        log.info(f"Setting prior dropout p={priordropout:.3f}")

        # Variational distributions (or priors for AGPS)
        if "-lstm" in solver:
            if "agps" in solver:
                vdistribution = CondLSTMProposal(
                    d_features=sequence_length,
                    k_categories=alpha_len,
                    embedding_dim=EMBEDDING_DIM,
                    u_dims=L,
                    num_layers=LLAYERS,
                    dropout=priordropout,
                    hidden_size=LNETWORKS,
                    clip_gradients=1.0,
                )
            else:
                vdistribution = LSTMProposal(
                    d_features=sequence_length,
                    k_categories=alpha_len,
                    embedding_dim=EMBEDDING_DIM,
                    num_layers=LLAYERS,
                    hidden_size=LNETWORKS,
                    dropout=priordropout,
                    clip_gradients=1.0,
                )
        elif "-tfm" in solver:
            if "agps" in solver:
                vdistribution = CondDTransformerProposal(
                    d_features=sequence_length,
                    k_categories=alpha_len,
                    u_dims=L,
                    embedding_dim=EMBEDDING_DIM,
                    nhead=NHEADS,
                    num_layers=DLAYERS,
                    dim_feedforward=DNETWORKS,
                    dropout=priordropout,
                    clip_gradients=1.0,
                )
            else:
                vdistribution = DTransformerProposal(
                    d_features=sequence_length,
                    k_categories=alpha_len,
                    embedding_dim=EMBEDDING_DIM,
                    nhead=NHEADS,
                    num_layers=DLAYERS,
                    dim_feedforward=DNETWORKS,
                    dropout=priordropout,
                    clip_gradients=1.0,
                )
        elif "-mtfm" in solver:
            if "agps" in solver:
                vdistribution = CondTransformerMutationProposal(
                    d_features=sequence_length,
                    k_categories=alpha_len,
                    u_dims=L,
                    embedding_dim=EMBEDDING_DIM,
                    nhead=NHEADS,
                    num_layers=DLAYERS,
                    dim_feedforward=DNETWORKS,
                    num_mutations=NUM_MUTATIONS[sequence_length],
                    replacement=False,
                    dropout=priordropout,
                    clip_gradients=1.0,
                )
            else:
                vdistribution = TransformerMutationProposal(
                    d_features=sequence_length,
                    k_categories=alpha_len,
                    embedding_dim=EMBEDDING_DIM,
                    nhead=NHEADS,
                    num_layers=DLAYERS,
                    dim_feedforward=DNETWORKS,
                    num_mutations=NUM_MUTATIONS[sequence_length],
                    replacement=False,
                    dropout=priordropout,
                    clip_gradients=1.0,
                )

        vdist_options = dict(
            optimizer_options=dict(lr=1e-4, weight_decay=0),
            stop_options=dict(miniter=2000),
            gradient_samples=gsamples,
        )
        if "mtfm" in solver and "-rf" not in solver:
            vdist_options |= dict(resample_iters=33)
        prior_options = dict(stop_options=dict(maxiter=PRIOR_MAXITER))

    if "vsd" in solver or "cbas" in solver:
        cpe = CNNClassProbability(
            seq_len=sequence_length, alpha_len=alpha_len, **CPE_PARAMS
        )

        if "cbas" in solver:
            solverclass = CbASSolver
        elif "-rf" in solver:
            solverclass = VSDSolver
        else:
            solverclass = VSDSolverIW

        optim = solverclass(
            black_box=black_box,
            x0=x0,
            y0=y0,
            alphabet=black_box.alphabet,
            labeller=ParetoAnnealed(percentile=startp, T=max_iter),
            cpe=cpe,
            vdistribution=vdistribution,
            device=device,
            seed=seed,
            vdist_options=vdist_options,
            prior_options=prior_options,
            bsize=bsize,
            prior_val_prop=PRIOR_VAL_PROB,
            topk_selection=TOPK_SELECTION,
        )
        solver_args = dict(seed=seed)

    elif "agps" in solver:
        pareto_cpe = PreferenceCNNClassProbability(
            seq_len=sequence_length, alpha_len=alpha_len, u_dims=L, **CPE_PARAMS
        )
        pref_cpe = deepcopy(pareto_cpe)

        locs = torch.randn(size=torch.Size([5, L]))
        pvdistribution = PreferenceSearchDistribution(
            cproposal=vdistribution,
            preference=MixtureUnitNormal(locs=locs),
        )

        if "-rf" in solver:
            solverclass = AGPSSolver
        else:
            solverclass = AGPSSolverIW

        optim = solverclass(
            black_box=black_box,
            x0=x0,
            y0=y0,
            alphabet=black_box.alphabet,
            vdistribution=pvdistribution,
            pareto_cpe=pareto_cpe,
            preference_cpe=pref_cpe,
            ref=ref,
            labeller=ParetoAnnealed(percentile=startp, T=max_iter),
            bsize=bsize,
            device=device,
            seed=seed,
            vdist_options=vdist_options,
            prior_options=prior_options,
            prior_val_prop=PRIOR_VAL_PROB,
            topk_selection=TOPK_SELECTION,
        )
        solver_args = dict(seed=seed)

    elif solver == "lambo2":
        optim = LaMBO2(
            black_box=black_box,
            x0=x0,
            y0=y0,
            overrides=[
                "max_epochs=2",
                f"kernel_size={5}",
                f"batch_size={bsize}",
                f"accelerator={device}",
                f"num_mutations_per_step={NUM_MUTATIONS[sequence_length]}",
            ],
            seed=seed,
            max_epochs_for_retraining=8,
        )
        solver_args = dict()

    elif solver == "rand":
        optim = RandomMutation(
            black_box=black_box,
            x0=x0,
            y0=y0,
            n_mutations=NUM_MUTATIONS[sequence_length],
            batch_size=bsize,
        )
        solver_args = dict(seed=seed)

    else:
        sys.exit(f"Unknown solver {solver}")

    # Solve
    start = time.time()
    optim.solve(max_iter=max_iter, **solver_args)
    end = time.time()
    log.info(f"Elapsed time: {(end - start) / 60:.3f} minutes.")

    fig, (ax1, ax2) = plt.subplots(1, 2, dpi=150, figsize=(12, 5))
    plot_hypervolume(observer, ref, ax1, tsize=tsize)
    plot_pareto(observer, ax2)
    fig.tight_layout()
    figurefile = logdir / f"{solver_name}.png"
    plt.savefig(figurefile)
    plt.close()

    # Save results
    try:
        if solver == "lambo2":
            observer.x_s = [
                np.array([list(str(x.squeeze()))]) if x.shape[1] == 1 else x
                for x in observer.x_s
            ]
        resultsfile = logdir / f"{solver_name}.npz"
        np.savez(
            resultsfile, x=np.vstack(observer.x_s), y=np.vstack(observer.y_s)
        )
    except Exception as e:
        log.error(f"Issue saving results: {e}")

    # Plot and save some preference conditioned samples
    if "agps" in solver:
        fig, ax = plt.subplots(1, 1, dpi=150, figsize=(8, 8))
        (u1, u2, u3), (ys1, ys2, ys3) = plot_cond_prefences(
            observer, optim, ref, device, ax
        )
        figurefile = logdir / f"{solver_name}_samples.png"
        plt.savefig(figurefile)
        plt.close()
        resultsfile = logdir / f"{solver_name}_samples.npz"
        np.savez(resultsfile, u1=u1, u2=u2, u3=u3, ys1=ys1, ys2=ys2, ys3=ys3)

    black_box.terminate()


def _erhlich_poli(sequence_length, seed, bsize):
    problem = EhrlichProblemFactory().create(
        sequence_length=sequence_length,
        return_value_on_unfeasible=-1,
        seed=seed,
        **SEQLEN_SETTING[sequence_length],
    )
    black_box = problem.black_box
    rs = np.random.RandomState(seed=seed)
    x0 = np.array(
        [
            list(black_box._sample_random_sequence(random_state=rs))
            for _ in range(bsize)
        ]
    )

    # Optimum is 1 for Ehrlich functions -- but just in case
    xstar = black_box.construct_optimal_solution()
    ystar = black_box(xstar)

    return black_box, ystar, x0


def _erhlich_holo(sequence_length, seed, bsize):
    problem = EhrlichHoloProblemFactory().create(
        sequence_length=sequence_length,
        return_value_on_unfeasible=-1,
        seed=seed,
        **SEQLEN_SETTING[sequence_length],
    )
    black_box = problem.black_box
    x0 = black_box.initial_solution(n_samples=bsize)
    x0 = np.array([list(x) for x in x0])

    # Optimum is 1 for Ehrlich functions -- but just in case
    xstar = black_box.optimal_solution()
    ystar = black_box(xstar)

    return black_box, ystar, x0
