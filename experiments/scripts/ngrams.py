import random
import logging
import sys
import time
from pathlib import Path
from copy import deepcopy

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import click

from experiments.ngrams.ngram_utils import ResidueTokenizer
from experiments.ngrams.ngram_regx import RegexTask
from experiments.blackboxes import NgramProblemFactory

from poli.core.util.abstract_observer import AbstractObserver
from poli.core.black_box_information import BlackBoxInformation
from poli_baselines.solvers.simple.random_mutation import RandomMutation

from poli_baselines.solvers.bayesian_optimization.lambo2 import LaMBO2
from vsd.cpe import CNNClassProbability, PreferenceCNNClassProbability
from vsd.proposals import (
    DTransformerProposal,
    TransformerMutationProposal,
)
from vsd.condproposals import (
    CondDTransformerProposal,
    CondTransformerMutationProposal,
    PreferenceSearchDistribution,
)
from vsd.preferences import MixtureUnitNormal
from vsd.solvers import (
    AGPSSolverIW,
    AGPSSolver,
    VSDSolverIW,
    VSDSolver,
    CbASSolver,
)
from vsd.labellers import ParetoAnnealed
from vsd.utils import is_non_dominated_strict

from experiments.ngrams.observer_utils import plot_sum_y

import warnings

warnings.filterwarnings("ignore")
MUTATION_BUDGET = 1  # For those methods that accept this
START_PERCENTILE = 0.5
EPSILON = None
JITTER = None
TEMBEDDING_DIM = 64
NHEAD = 4
TNUM_LAYERS = 2
TDIM_FEEDFORWARD = 128
PRIOR_DROPOUT = 0.2
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
    ) -> object:
        pass

    def observe(self, x: np.ndarray, y: np.ndarray, context=None) -> None:
        self.x_s.append(x)
        self.y_s.append(y)
        z = (
            is_non_dominated_strict(torch.Tensor(self.y_s).reshape(-1, 3))
            .detach()
            .numpy()
            .squeeze()
        )
        if np.ndim(z) > 0:
            zb = z[-len(y) :].squeeze()
        else:
            zb = z
        if self.logger is not None:
            self.logger.info(f"Sequence: {''.join(x[0])}")
            self.logger.info(f" fitness: {y}, is not dominated: {zb}")


@click.command()
@click.option(
    "--solver",
    type=click.Choice(
        [
            "agps-tfm",
            "agps-mtfm",
            "agps-tfm-rf",
            "agps-mtfm-rf",
            "vsd-tfm",
            "vsd-mtfm",
            "vsd-tfm-rf",
            "vsd-mtfm-rf",
            "cbas-tfm",
            "cbas-mtfm",
            "lambo2",
            "rand",
        ]
    ),
    default="vsd-tfm",
    help="Solver to run.",
)
@click.option("--seed", type=int, default=42, help="Random seed.")
@click.option(
    "--max-iter", type=int, default=64, help="Number of optimization steps."
)
@click.option(
    "--logdir",
    type=click.Path(file_okay=False),
    default="ngrammoo",
    help="log and results directory.",
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
    help="Training dataset sized.",
)
@click.option(
    "--device", type=str, default="cpu", help="Device to use for solver."
)
def main(solver, seed, max_iter, logdir, bsize, tsize, device):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

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
        f"solver: {solver},"
        f" max_iter: {max_iter},"
        f" logdir: {logdir}, device: {device}, seed: {seed}."
    )

    tokenizer = ResidueTokenizer()
    task = RegexTask(
        tokenizer=tokenizer,
        regex_list=["(?=AV)", "(?=VC)", "(?=CA)"],
        obj_dim=3,
        log_prefix="regex",
        min_len=32,
        max_len=34,
        num_start_examples=tsize,
        batch_size=bsize,
        max_num_edits=None,
        max_ngram_size=1,
        allow_len_change=True,
        max_score_per_dim=18,
        eval_pref=[0.3, 0.5, 0.2],
        seed=seed,
    )

    x0, _ = task.task_setup()
    sequence_length = max(len(seq) for seq in x0)

    problem = NgramProblemFactory().create(
        tokenizer=tokenizer,
        regex_list=["(?=AV)", "(?=VC)", "(?=CA)"],
        min_len=32,
        max_len=sequence_length,
        eval_pref=[0.3, 0.5, 0.2],
        seed=seed,
    )

    black_box = problem.black_box
    observer = SimpleObserver(logger=log)
    black_box.set_observer(observer)
    ALPHA_LEN = len(black_box.alphabet)

    y0 = black_box(x0)
    L = y0.shape[1]
    log.info(f"# positive examples: {y0.sum(axis=0)}")

    labeller = ParetoAnnealed(
        percentile=START_PERCENTILE, T=max_iter, jitter=JITTER, epsilon=EPSILON
    )
    if "-tfm" in solver:
        lr = 1e-4
        if "agps" in solver:
            vdistribution = CondDTransformerProposal(
                d_features=sequence_length,
                k_categories=ALPHA_LEN,
                u_dims=L,
                embedding_dim=TEMBEDDING_DIM,
                nhead=NHEAD,
                num_layers=TNUM_LAYERS,
                dim_feedforward=TDIM_FEEDFORWARD,
                dropout=PRIOR_DROPOUT,
                clip_gradients=1.0,
            )
        else:
            vdistribution = DTransformerProposal(
                d_features=sequence_length,
                k_categories=ALPHA_LEN,
                embedding_dim=TEMBEDDING_DIM,
                nhead=NHEAD,
                num_layers=TNUM_LAYERS,
                dim_feedforward=TDIM_FEEDFORWARD,
                dropout=PRIOR_DROPOUT,
                clip_gradients=1.0,
            )
    elif "-mtfm" in solver:
        lr = 1e-4
        if "agps" in solver:
            vdistribution = CondTransformerMutationProposal(
                d_features=sequence_length,
                k_categories=ALPHA_LEN,
                u_dims=L,
                embedding_dim=TEMBEDDING_DIM,
                nhead=NHEAD,
                num_layers=TNUM_LAYERS,
                dim_feedforward=TDIM_FEEDFORWARD,
                dropout=PRIOR_DROPOUT / 4,
                num_mutations=MUTATION_BUDGET,
                replacement=False,
                clip_gradients=1.0,
            )
        else:
            vdistribution = TransformerMutationProposal(
                d_features=sequence_length,
                k_categories=ALPHA_LEN,
                embedding_dim=TEMBEDDING_DIM,
                nhead=NHEAD,
                num_layers=TNUM_LAYERS,
                dim_feedforward=TDIM_FEEDFORWARD,
                num_mutations=MUTATION_BUDGET,
                dropout=PRIOR_DROPOUT / 4,
                replacement=False,
                clip_gradients=1.0,
            )

    if any(s in solver for s in ["agps", "vsd", "cbas"]):
        vdist_options = dict(
            optimizer_options=dict(lr=lr, weight_decay=0),
            stop_options=dict(miniter=2000),
            gradient_samples=256,
        )
        prior_options = dict(stop_options=dict(maxiter=PRIOR_MAXITER))
        if "mtfm" in solver and "-rf" not in solver:
            vdist_options |= dict(resample_iters=33)

    if solver in (
        "vsd-tfm",
        "vsd-mtfm",
        "vsd-tfm-rf",
        "vsd-mtfm-rf",
        "cbas-tfm",
        "cbas-mtfm",
    ):
        cpe = CNNClassProbability(
            seq_len=sequence_length, alpha_len=ALPHA_LEN, **CPE_PARAMS
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
            labeller=labeller,
            cpe=cpe,
            vdistribution=vdistribution,
            device=device,
            seed=seed,
            vdist_options=vdist_options,
            prior_options=prior_options,
            bsize=bsize,
            alphabet=black_box.alphabet,
            prior_val_prop=PRIOR_VAL_PROB,
            topk_selection=TOPK_SELECTION,
        )
        solver_args = dict(seed=seed)
    elif solver in ("agps-tfm", "agps-tfm-rf", "agps-mtfm", "agps-mtfm-rf"):
        pareto_cpe = PreferenceCNNClassProbability(
            seq_len=sequence_length, alpha_len=ALPHA_LEN, u_dims=L, **CPE_PARAMS
        )
        preference_cpe = deepcopy(pareto_cpe)

        locs = torch.randn(size=[5, L])
        vpdistribution = PreferenceSearchDistribution(
            cproposal=vdistribution,
            preference=MixtureUnitNormal(locs=locs),
        )
        solverclass = AGPSSolver if "-rf" in solver else AGPSSolverIW
        optim = solverclass(
            black_box=black_box,
            x0=x0,
            y0=y0,
            vdistribution=vpdistribution,
            pareto_cpe=pareto_cpe,
            preference_cpe=preference_cpe,
            ref=torch.zeros(L),
            labeller=labeller,
            bsize=bsize,
            device=device,
            seed=seed,
            vdist_options=vdist_options,
            prior_options=prior_options,
            alphabet=black_box.alphabet,
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
                f"kernel_size={5}",  # Default and also used by us
                f"batch_size={bsize}",
                f"accelerator={device}",
                f"num_mutations_per_step={MUTATION_BUDGET}",
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
            n_mutations=MUTATION_BUDGET,
            batch_size=bsize,
            top_k=bsize,
        )
        solver_args = dict(seed=seed)
    else:
        sys.exit(f"Unknown solver {solver}")

    start = time.time()
    optim.solve(max_iter=max_iter, **solver_args)
    end = time.time()
    log.info(f"Elapsed time: {(end - start) / 60:.3f} minutes.")

    fig, (ax1, ax2) = plt.subplots(1, 2)
    plot_sum_y(observer, ax1)
    plot_sum_y(observer, ax2, start_from=x0.shape[0])
    ax1.axvline(x0.shape[0], color="red")
    fig.tight_layout()
    figurefile = logdir / f"{solver}_{seed}.png"
    plt.savefig(figurefile)
    plt.close()

    # Save results
    try:
        if solver == "lambo2":
            observer.x_s = [
                np.array([list(str(x.squeeze()))]) if x.shape[1] == 1 else x
                for x in observer.x_s
            ]
        resultsfile = logdir / f"{solver}_{seed}.npz"
        np.savez(
            resultsfile, x=np.vstack(observer.x_s), y=np.vstack(observer.y_s)
        )
    except Exception as e:
        log.error(f"Issue saving results: {e}")

    black_box.terminate()
