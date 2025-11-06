"""Sequence Multi-Objective Optimization Experiments.

This uses the data from LaMBO:
https://github.com/samuelstanton/lambo/tree/main/lambo/assets/foldx

And also requires FoldX to be installed, see:
https://machinelearninglifescience.github.io/poli-docs/understanding_foldx/00-installing-foldx.html
"""

import glob
import logging
import random
import sys
import time
from copy import deepcopy
from multiprocessing import cpu_count
from pathlib import Path

import click
import matplotlib as mpl
import numpy as np
import pandas as pd

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
    FoldXStabilityAndSASABlackBox,
    FoldXStabilityAndSASAProblemFactory,
)
from poli_baselines.solvers.bayesian_optimization.lambo2 import LaMBO2

from vsd.condproposals import (
    CondTransformerMutationProposal,
    PreferenceSearchDistribution,
)
from vsd.cpe import CNNClassProbability, PreferenceCNNClassProbability
from vsd.preferences import MixtureUnitNormal
from vsd.proposals import (
    TransformerMutationProposal,
)
from vsd.solvers import (
    AGPSSolverIW,
    CbASSolver,
    VSDSolverIW,
    RandomPadMutation,
    int2seq_unpad,
)
from vsd.utils import is_non_dominated_strict
from vsd.labellers import ParetoAnnealed

NUM_MUTATIONS = 1
START_PERCENTILE = 0.90
EMBEDDING_DIM = 64
NHEAD = 4
NUM_LAYERS = 2
DIM_FEEDFORWARD = 256
PRIOR_DROPOUT = 0.1
TOPK_SELECTION = False
PRIOR_VAL_PROP = 0
PRIOR_MAXITER = 2501
SAMPLES = 256
CPE_PARAMS = dict(
    ckernel=7,
    xkernel=5,
    xstride=4,
    cfilter_size=96,
    linear_size=192,
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
        if self.logger is not None:
            self.logger.info(f"candidate values: {y.squeeze()}")
        self.x_s.append(x)
        self.y_s.append(y)


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
    ax.set_xlabel("$f_0$")
    ax.set_ylabel("$f_1$")
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


def pad_sequences(
    sequences: np.ndarray | list, min_len: int | None = None
) -> np.ndarray:

    # Make sure we remove excess dimensions
    fsequences = [str(np.squeeze(seq)) for seq in sequences]

    # Determine the maximum length
    max_len = max(len(seq) for seq in fsequences)
    if min_len is not None:
        max_len = max(min_len, max_len)

    # Create an array of shape (n_sequences, max_len) filled with '-'
    padded = np.full((len(fsequences), max_len), "-", dtype="<U1")

    # Fill in each sequence
    for i, seq in enumerate(fsequences):
        padded[i, : len(seq)] = list(seq)
    return padded


def unpad_sequences(padded: np.ndarray) -> np.ndarray:
    sequences = ["".join(r).rstrip("-") for r in padded]
    return np.array(sequences)


FoldXStabilityAndSASABlackBox._original_call = (
    FoldXStabilityAndSASABlackBox.__call__
)


def patched_call(self, x):
    x = unpad_sequences(x)
    return FoldXStabilityAndSASABlackBox._original_call(self, x)


FoldXStabilityAndSASABlackBox.__call__ = patched_call


@click.command()
@click.option(
    "--solver",
    type=click.Choice(
        [
            "agps-mtfm",
            "vsd-mtfm",
            "cbas-mtfm",
            "lambo2",
            "rand",
        ]
    ),
    default="vsd-mtfm",
    help="solver to run.",
)
@click.option(
    "--max-iter", type=int, default=64, help="Maximum iterations to run."
)
@click.option(
    "--bsize",
    type=int,
    default=16,
    help="Batch size for black box evaluation.",
)
@click.option(
    "--tsize",
    type=int,
    default=512,
    help="Training dataset size.",
)
@click.option(
    "--datapath",
    type=click.Path(exists=True, file_okay=False),
    help="Path to the directory that holds the PDB files.",
    default="data/rfp_pdbs",
)
@click.option(
    "--logdir",
    type=click.Path(file_okay=False),
    default="foldx_ss",
    help="log and results directory.",
)
@click.option(
    "--device", type=str, default="cpu", help="device to use for solver."
)
@click.option("--seed", type=int, default=42, help="random seed.")
@click.option(
    "--ref",
    type=(float, float),
    default=None,
    help="Reference point for hypervolume computation",
)
def main(
    solver,
    max_iter,
    bsize,
    tsize,
    datapath,
    logdir,
    device,
    seed,
    ref,
):
    # Setup logging
    logdir = Path(logdir)
    logdir.mkdir(exist_ok=True, parents=True)
    logfile = logdir / Path(f"{solver}_{seed}.log")
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
        f"solver: {solver}, max_iter: {max_iter}, bsize: {bsize}, "
        f"tsize: {tsize}, logdir: {logdir}, device: {device}, seed: {seed}."
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
    log.info("Setting up foldx problem ...")
    log.info("Loading PDBs ...")
    ALL_PDBS = [
        Path(p)
        for p in glob.glob(f"{datapath}/**/*_Repair.pdb", recursive=True)
    ]
    log.info(f"Found {len(ALL_PDBS)} initial sequences ...")
    problem = FoldXStabilityAndSASAProblemFactory().create(
        wildtype_pdb_path=ALL_PDBS,
        parallelize=True,
        batch_size=bsize,
        num_workers=cpu_count(),
        verbose=False,
    )
    black_box = problem.black_box
    observer = SimpleObserver(logger=log)
    black_box.set_observer(observer)

    log.info("Loading Training data ...")
    cachefile = Path(datapath) / "rfp_training_cache.csv"
    # Rebuild the cache -- target values are different for old foldx version
    if not Path(cachefile).exists():
        log.info("Rebuilding cache ...")
        df = pd.read_csv(Path(datapath) / "proxy_rfp_seed_data.csv")
        x = df["foldx_seq"].values
        y = black_box(x)
        df[["stability", "SASA"]] = y
        df.to_csv(cachefile, index=False)
        del x, y, df
        log.info("Done rebuilding cache ...")

    df = pd.read_csv(cachefile).sample(n=tsize, random_state=seed)
    x0 = df["foldx_seq"].values
    y0 = df[["stability", "SASA"]].values
    log.info(f"  loaded {len(x0)} training values.")

    if solver != "lambo2":
        x0 = pad_sequences(x0)
        alphabet = list(set(x0.flatten()))
    else:
        alphabet = list(set("".join(x0)).union({"-"}))

    observer.x_s.append(x0)
    observer.y_s.append(y0)
    L = y0.shape[1]
    sequence_length = len(x0[0])
    alpha_len = len(alphabet)
    if ref is not None:
        ref = torch.tensor(ref)

    log.info("Setting up solver ...")
    if "-mtfm" in solver:
        pad_token = alphabet.index("-")
        vdist_options = dict(
            optimizer_options=dict(lr=1e-4, weight_decay=0),
            stop_options=dict(miniter=3000),
            gradient_samples=SAMPLES,
            resample_iters=33,
        )
        prior_options = dict(stop_options=dict(maxiter=PRIOR_MAXITER))

    if solver in ("vsd-mtfm", "cbas-mtfm"):
        vdistribution = TransformerMutationProposal(
            d_features=sequence_length,
            k_categories=alpha_len,
            embedding_dim=EMBEDDING_DIM,
            num_mutations=NUM_MUTATIONS,
            nhead=NHEAD,
            num_layers=NUM_LAYERS,
            dim_feedforward=DIM_FEEDFORWARD,
            pad_token=pad_token,
            replacement=False,
            dropout=PRIOR_DROPOUT,
            clip_gradients=1.0,
        )
        cpe = CNNClassProbability(
            seq_len=sequence_length,
            alpha_len=alpha_len,
            **CPE_PARAMS,
        )

        solverclass = CbASSolver if "cbas" in solver else VSDSolverIW
        optim = solverclass(
            black_box=black_box,
            x0=x0,
            y0=y0,
            alphabet=alphabet,
            labeller=ParetoAnnealed(percentile=START_PERCENTILE, T=max_iter),
            cpe=cpe,
            vdistribution=vdistribution,
            device=device,
            seed=seed,
            vdist_options=vdist_options,
            prior_options=prior_options,
            bsize=bsize,
            prior_val_prop=PRIOR_VAL_PROP,
            topk_selection=TOPK_SELECTION,
        )
        solver_args = dict(seed=seed)

    elif solver == "agps-mtfm":
        vdistribution = CondTransformerMutationProposal(
            d_features=sequence_length,
            k_categories=alpha_len,
            u_dims=L,
            embedding_dim=EMBEDDING_DIM,
            num_mutations=NUM_MUTATIONS,
            nhead=NHEAD,
            num_layers=NUM_LAYERS,
            dim_feedforward=DIM_FEEDFORWARD,
            pad_token=pad_token,
            replacement=False,
            dropout=PRIOR_DROPOUT,
            clip_gradients=1.0,
        )
        pareto_cpe = PreferenceCNNClassProbability(
            seq_len=sequence_length,
            alpha_len=alpha_len,
            u_dims=L,
            **CPE_PARAMS,
        )
        pref_cpe = deepcopy(pareto_cpe)

        locs = torch.randn(size=[5, L])
        pvdistribution = PreferenceSearchDistribution(
            cproposal=vdistribution,
            preference=MixtureUnitNormal(locs=locs),
        )

        optim = AGPSSolverIW(
            black_box=black_box,
            x0=x0,
            y0=y0,
            alphabet=alphabet,
            vdistribution=pvdistribution,
            pareto_cpe=pareto_cpe,
            preference_cpe=pref_cpe,
            labeller=ParetoAnnealed(percentile=START_PERCENTILE, T=max_iter),
            ref=ref,
            bsize=bsize,
            device=device,
            seed=seed,
            vdist_options=vdist_options,
            prior_options=prior_options,
            prior_val_prop=PRIOR_VAL_PROP,
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
                f"kernel_size={7}",  # Same as us, but default is 3, 9 in paper
                f"batch_size={bsize}",
                f"accelerator={device}",
                f"num_mutations_per_step={NUM_MUTATIONS}",
            ],
            seed=seed,
            max_epochs_for_retraining=8,
        )
        solver_args = dict()

    elif solver == "rand":
        optim = RandomPadMutation(
            black_box=black_box,
            x0=x0,
            y0=y0,
            n_mutations=NUM_MUTATIONS,
            batch_size=bsize,
            top_k=bsize,
            pad_char="-",
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
    figurefile = logdir / f"{solver}_{seed}.png"
    plt.savefig(figurefile)
    plt.close()

    # Save results
    try:
        resultsfile = logdir / f"{solver}_{seed}.npz"
        if solver == "lambo2":
            xsave = np.vstack(
                (
                    pad_sequences(observer.x_s[0], min_len=sequence_length),
                    pad_sequences(observer.x_s[1:], min_len=sequence_length),
                )
            )
        else:
            xsave = np.vstack(
                (
                    observer.x_s[0],
                    pad_sequences(observer.x_s[1:], min_len=sequence_length),
                )
            )
        np.savez(resultsfile, x=xsave, y=np.vstack(observer.y_s))
    except Exception as e:
        log.error(f"Issue saving results: {e}")

    # Plot and save some preference conditioned samples
    if "agps" in solver:
        fig, ax = plt.subplots(1, 1, dpi=150, figsize=(8, 8))
        (u1, u2, u3), (ys1, ys2, ys3) = plot_cond_prefences(
            observer, optim, ref, device, ax
        )
        figurefile = logdir / f"{solver}_{seed}_samples.png"
        plt.savefig(figurefile)
        plt.close()
        resultsfile = logdir / f"{solver}_{seed}_samples.npz"
        np.savez(resultsfile, u1=u1, u2=u2, u3=u3, ys1=ys1, ys2=ys2, ys3=ys3)

    black_box.terminate()
