"""BBO using the Ehrlich function.

This experiment is based off:
https://github.com/MachineLearningLifeScience/poli-baselines/blob/main/examples/07_running_lambo2_on_ehrlich/run.py
"""

import logging
import sys
from pathlib import Path

import click
import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from poli.core.black_box_information import BlackBoxInformation
from poli.core.util.abstract_observer import AbstractObserver
from poli.objective_repository import (
    EhrlichHoloProblemFactory,
    EhrlichProblemFactory,
)
from poli_baselines.solvers.bayesian_optimization.lambo2 import LaMBO2
from poli_baselines.solvers.simple.genetic_algorithm import (
    FixedLengthGeneticAlgorithm,
)

from vsd.augmentation import TransitionAugmenter
from vsd.proposals import (
    DTransformerProposal,
    LSTMProposal,
    MultiCategoricalProposal,
)
from vsd.solvers import CbASSolver, VSDSolver
from vsd.surrogates import CNNClassProbability, EnsembleProbabilityModel
from vsd.thresholds import BudgetAnnealedThreshold

SEQLEN_SETTING = {
    15: dict(motif_length=4, n_motifs=2, quantization=4),
    32: dict(motif_length=4, n_motifs=2, quantization=4),
    64: dict(motif_length=4, n_motifs=8, quantization=4),
    256: dict(motif_length=10, n_motifs=16, quantization=10),
}
KERNEL_WIDTHS = {15: 4, 32: 7, 64: 7, 256: 13}  # For the CNNs
LNETWORKS = {15: 32, 32: 32, 64: 64, 256: 128}
DNETWORKS = {15: 32, 32: 64, 64: 128, 256: 256}
NHEADS = {15: 1, 32: 2, 64: 4, 256: 6}


# Code from https://github.com/MachineLearningLifeScience/poli-baselines/blob/
#   main/examples/07_running_lambo2_on_ehrlich/simple_observer.py
class SimpleObserver(AbstractObserver):
    def __init__(self) -> None:
        self.x_s = []
        self.y_s = []
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


def plot_best_y(obs: SimpleObserver, ax: plt.Axes, start_from: int = 0):
    best_y = np.maximum.accumulate(np.vstack(obs.y_s).flatten())
    ax.plot(best_y.flatten()[start_from:])
    ax.set_xlabel("Number of evaluations")
    ax.set_ylabel("Best value found")


def plot_regret(
    ystar: float, obs: SimpleObserver, ax: plt.Axes, start_from: int = 0
):
    diff = ystar - np.vstack(obs.y_s)
    regret = np.minimum.accumulate(diff.flatten())
    ax.plot(regret[start_from:])
    ax.set_xlabel("Number of evaluations")
    ax.set_ylabel("Simple regret")


def plot_all_y(obs: SimpleObserver, ax: plt.Axes, start_from: int = 0):
    ax.plot(np.vstack(obs.y_s).flatten()[start_from:], ".")
    ax.set_xlabel("Number of evaluations")
    ax.set_ylabel("Values found")


@click.command()
@click.option(
    "--solver",
    type=click.Choice(
        [
            "vsd-lstm",
            "vsd-tfm",
            "vsd-mf",
            "cbas-lstm",
            "cbas-tfm",
            "cbas-mf",
            "lambo2",
            "ga",
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
    "--max-iter", type=int, default=5, help="Maximum iterations to run."
)
@click.option(
    "--bsize",
    type=int,
    default=128,
    help="Batch size for black box evaluation.",
)
@click.option(
    "--logdir",
    type=click.Path(file_okay=False),
    default="ehrlich",
    help="log and results directory.",
)
@click.option(
    "--device", type=str, default="cpu", help="device to use for solver."
)
@click.option("--seed", type=int, default=42, help="random seed.")
@click.option(
    "--poli",
    is_flag=True,
    help="use the poli-inbuilt ehrlich function implementation.",
)
def ehrlich(
    solver, sequence_length, max_iter, bsize, logdir, device, seed, poli
):
    # Setup logging
    logdir = Path(logdir) / sequence_length
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

    # Fix seeds
    torch.manual_seed(seed)

    # Setup problem
    sequence_length = int(sequence_length)
    log.info(
        f"Creating Ehrlich function: length {sequence_length} with"
        f" {SEQLEN_SETTING[sequence_length]}"
    )
    if poli:
        log.info("Using built-in poli Ehrlich implementation.")
        black_box, ystar, x0 = _erhlich_poli(sequence_length, seed, bsize)
    else:
        log.info("Using Holo bench Ehrlich implementation.")
        black_box, ystar, x0 = _erhlich_holo(sequence_length, seed, bsize)
    y0 = black_box(x0)

    # Set up a history logger
    observer = SimpleObserver()
    black_box.set_observer(observer)
    observer.x_s.append(x0)
    observer.y_s.append(y0)

    if solver in (
        "vsd-lstm",
        "vsd-tfm",
        "vsd-mf",
        "cbas-lstm",
        "cbas-mf",
        "cbas-tfm",
    ):
        alpha_len = len(black_box.alphabet)
        threshold = BudgetAnnealedThreshold(p0=0.5, pT=0.99, T=max_iter)
        cpe = EnsembleProbabilityModel(
            base_class=CNNClassProbability,
            init_kwargs=dict(
                seq_len=sequence_length,
                alpha_len=alpha_len,
                ckernel=KERNEL_WIDTHS[sequence_length],
                xkernel=2,
                xstride=2,
                cfilter_size=16,
                linear_size=128,
                embedding_dim=10,
                dropoutp=0.2,
                pos_encoding=True,
            ),
            ensemble_size=10,
        )

        # Data augmentation for prior fitting
        augmenter = TransitionAugmenter(max_mutations=5)
        prior_options = dict(augmenter=augmenter, augmentation_p=0.2)

        if "lstm" in solver:
            vdistribution = LSTMProposal(
                d_features=sequence_length,
                k_categories=alpha_len,
                num_layers=3,
                hidden_size=LNETWORKS[sequence_length],
                clip_gradients=1.0,
            )
        elif "tfm" in solver:
            vdistribution = DTransformerProposal(
                d_features=sequence_length,
                k_categories=alpha_len,
                nhead=NHEADS[sequence_length],
                num_layers=2,
                dim_feedforward=DNETWORKS[sequence_length],
                clip_gradients=1.0,
            )
        else:
            vdistribution = MultiCategoricalProposal(
                d_features=sequence_length,
                k_categories=alpha_len,
            )

        if "cbas" in solver:
            solverclass = CbASSolver
        else:
            solverclass = VSDSolver

        # lstm or tfm with score function gradients require a lower lr
        vdist_options = None
        if solver in ("vsd-tfm", "vsd-lstm"):
            vdist_options = dict(optimizer_options=dict(lr=1e-4))

        optim = solverclass(
            black_box=black_box,
            x0=x0,
            y0=y0,
            threshold=threshold,
            cpe=cpe,
            vdistribution=vdistribution,
            device=device,
            seed=seed,
            vdist_options=vdist_options,
            prior_options=prior_options,
        )
        solver_args = dict(seed=seed)

    elif solver == "lambo2":
        optim = LaMBO2(
            black_box=black_box,
            x0=x0,
            y0=y0,
            overrides=[
                "max_epochs=2",
                f"kernel_size={KERNEL_WIDTHS[sequence_length]}",
                f"batch_size={bsize}",
                f"accelerator={device}",
            ],
            seed=seed,
            max_epochs_for_retraining=8,
        )
        solver_args = dict()

    elif solver == "ga":
        optim = FixedLengthGeneticAlgorithm(
            black_box=black_box,
            x0=x0,
            y0=y0,
            population_size=bsize,
            prob_of_mutation=0.2,
            initialize_with_x0=False,
        )
        solver_args = dict(seed=seed)

    else:
        sys.exit(f"Unknown solver {solver}")

    # Solve
    optim.solve(max_iter=max_iter, **solver_args)

    fig, (ax1, ax2) = plt.subplots(1, 2, dpi=150, figsize=(12, 5))
    plot_regret(ystar, observer, ax1)
    plot_all_y(observer, ax2)
    for ax in (ax1, ax2):
        ax.axvline(len(x0), color="red", label="training cuttoff")
        ax.legend()
    fig.tight_layout()
    figurefile = logdir / f"{solver}_{seed}.png"
    plt.savefig(figurefile)
    plt.close()

    # Save results
    try:
        if solver == "lambo2":
            observer.x_s[1:] = [
                np.array(list(x[0, 0])) for x in observer.x_s[1:]
            ]
        resultsfile = logdir / f"{solver}_{seed}.npz"
        np.savez(
            resultsfile, x=np.vstack(observer.x_s), y=np.vstack(observer.y_s)
        )
    except Exception as e:
        log.error(f"Issue saving results: {e}")

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
