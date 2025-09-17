"""Experiment simulating multiple rounds with batch selection."""

import json
import logging
from pathlib import Path
from typing import Tuple

import click
import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import Tensor

from experiments.dataclasses import DATASETS, intarr2seq, seq2intarr
from experiments.methods import (
    get_adalead_components,
    get_bore_components,
    get_cbas_components,
    get_dbas_components,
    get_pex_components,
    get_rs_components,
    get_vsd_components,
)
from experiments.metrics import (
    Performance,
    Precision,
    Recall,
    diversity,
    gap,
    innovation,
    maxfitness,
    novel_inds,
    novelty,
    performance,
    precision,
    simple_regret,
)
from vsd import cpe, labellers, surrogates
from vsd.acquisition import VariationalSearchAcquisition
from vsd.cpe import fit_cpe
from vsd.surrogates import update_gp
from vsd.utils import SequenceArray

METHODS = {
    "VSD": get_vsd_components,
    "BORE": get_bore_components,
    "DbAS": get_dbas_components,
    "CbAS": get_cbas_components,
    "Random": get_rs_components,
    "AdaLead": get_adalead_components,
    "PEX": get_pex_components,
}


@click.command()
@click.option(
    "--config",
    type=click.Path(dir_okay=False),
    default=None,
    help="config file to use.",
)
@click.option(
    "--dataset",
    default="DHFR",
    help="dataset for the experiment",
    type=click.Choice(DATASETS.keys(), case_sensitive=False),
)
@click.option(
    "--method",
    default="VSD",
    help="search method to use.",
    type=click.Choice(METHODS.keys(), case_sensitive=True),
)
@click.option(
    "--resultsdir",
    type=click.Path(file_okay=False),
    default="results",
    help="experiment results directory.",
)
@click.option("--seed", type=int, default=None, help="custom seed.")
@click.option(
    "--device", type=str, default=None, help="Override device config setting."
)
@click.option(
    "--suffix",
    default=None,
    help="extra suffix to add to file names",
    type=click.STRING,
)
@click.option("-gp", is_flag=True, help="use a GP surrogate model.")
def run_experiment(
    config, dataset, method, resultsdir, seed, device, suffix, gp
):
    if config is None:
        config = DATASETS[dataset]["config"]
    else:
        config = json.load(Path(config))
        dataset = config["data"]

    if seed is None:
        seed = config["seed"]
    else:
        config["seed"] = seed
    resultsdir = Path(resultsdir) / dataset / f"seed{seed}"

    if device is None:
        device = config["device"]
    else:
        config["device"] = device

    methodname = method
    if suffix is not None:
        methodname += f"_{suffix}"

    # Setup logging
    resultsdir.mkdir(exist_ok=True, parents=True)
    logfile = resultsdir / f"{methodname}_batch_rounds.log"
    logging.basicConfig(
        level=logging.INFO,
        handlers=[
            logging.FileHandler(logfile, mode="w"),
            logging.StreamHandler(),
        ],
    )
    log = logging.getLogger(name=__name__)
    log.info(f"Running {dataset} experiment with {methodname} method.")

    # Load full dataset
    torch.manual_seed(seed)
    log.info(f"Loading {dataset} data ...")
    data = DATASETS[dataset]["data"](device=device)

    # Loading training data -- needed for evaluation too
    log.info("Loading training data ...")
    S, y = data.load_data(config["training_data"]["save_path"])
    seqlen = data.get_seq_len()
    alphalen = data.get_alpha_len()
    ymax, Smax = data.get_max_fitness()
    y = torch.tensor(y, dtype=torch.float32)
    X = seq2intarr(S, is_amino=alphalen > 4)  # tokenizing training data
    dataind = -np.ones(len(y), dtype=int)  # keep track of rounds and training
    ytrainmax, Strainmax = get_fittest(y, S)
    log.info(f"Max fitness in dataset = {ymax:.3f}")
    log.info(f"Max fitness training data = {ytrainmax:.3f}")
    log.info(f"Training data fitness gap = {gap(Smax, Strainmax)}")

    # Load trained surrogate model
    log.info("Loading surrogate ...")
    params = dict(seq_len=seqlen, alpha_len=alphalen)
    if gp:
        soption = "gp"
        params |= dict(X=X, y=y)
        surrogate_class = getattr(surrogates, config[soption]["class"])
    else:
        soption = "cpe"
        surrogate_class = getattr(cpe, config[soption]["class"])
    surrogate = surrogate_class(**(params | config[soption]["parameters"]))
    surrogate.load_state_dict(
        torch.load(
            config[soption]["path"],
            map_location=torch.device(device),
            weights_only=True,
        )
    )
    surrogate.eval()
    log.info(f"Using {surrogate_class} as the predictor.")

    # Get fitness threshold
    thresh_class = getattr(labellers, config["threshold"]["class"])
    thresh = thresh_class(**config["threshold"]["args"], update_on_call=False)
    best_f = thresh.update(y)

    # Call signature mapping for metrics, options:
    #   Sc: sequence candidates, SequenceArray
    #   S: all seen sequences, SequenceArray
    #   yc: sequence candidate fitness values, Tensor
    #   y: all seen sequences fitness values, Tensor
    #   best_f: current fitness threshold, float
    #   yx: global maximum fitness value: float
    t_rounds = config["t_rounds"]
    b_size = config["b_cands"]
    perf = Performance(S_init=S)
    METRICS = {
        "Diversity": (diversity, ["Sc"]),
        "Novelty": (novelty, ["Sc", "S"]),
        "Performance (batch)": (performance, ["yc", "Sc", "S"]),
        "Max Fitness": (maxfitness, ["yc"]),
        "Precision (batch)": (precision, ["yc", "Sc", "S", "best_f"]),
        "Innovation": (innovation, ["Sc", "S", "yc", "y"]),
        "Fitness threshold": (lambda x: x, ["best_f"]),
        "Simple regret": (simple_regret, ["yc", "y", "yx"]),
        "Performance": (perf, ["yc", "Sc"]),
    }

    # Compute precision and recall if the fitness threshold is static
    if thresh.static:
        allpos = sum(data.y > best_f)
        BT = t_rounds * b_size
        prec = Precision(best_f=best_f, S_init=S)
        rec = Recall(best_f=best_f, npositives=allpos, S_init=S)
        recb = Recall(best_f=best_f, npositives=BT, S_init=S)
        METRICS.update(
            {
                "Precision": (prec, ["yc", "Sc"]),
                "Recall": (rec, ["yc", "Sc"]),
                "Recall (budget)": (recb, ["yc", "Sc"]),
            }
        )

    # Set up results file
    resultsfilename = resultsdir / f"{methodname}_results.csv"
    columns = list(METRICS.keys())
    results = pd.DataFrame(columns=columns, index=range(t_rounds))

    # Record labels
    z = thresh(y)

    # load search model components
    log.info(f"Setting up acquisition model ...")
    acq, proposal, gen = METHODS[method](
        seqlen, alphalen, surrogate, best_f, config, X, y
    )

    # Optimisation callbacks
    it, acquis, meangrad = [], [], []

    def callback(i: int, loss: Tensor, grad: Tuple[Tensor]):
        """For logging."""
        it.append(i)
        acquis.append(-loss.detach().to("cpu"))
        meangrad.append(np.mean([g.detach().to("cpu").mean() for g in grad]))

    log.info("Running experimental rounds ...")
    for t in range(t_rounds):
        log.info(f"Round {t}")

        log.info(f"Generating {config["b_cands"]} ...")
        proposal.train()
        it, acquis, meangrad = [], [], []
        Xcand, _ = gen(
            acquisition_function=acq,
            proposal_distribution=proposal,
            callback=callback,
        )
        proposal.eval()
        if len(Xcand) > b_size:
            raise RuntimeError(
                f"Number of candidates {len(Xcand)} > batch size {b_size}"
            )
        Xcand = Xcand.to("cpu").long()
        assert len(Xcand) <= b_size
        plot_optimisation(it, acquis, meangrad)
        figurefile = resultsdir / f"{methodname}_loss_{t}.png"
        plt.savefig(figurefile)
        plt.close()

        log.info(f"Evaluating and saving candidates ...")
        Scand = intarr2seq(Xcand, is_amino=alphalen > 4)
        ycand = torch.tensor(data.get_fitness(Scand), dtype=torch.float32)
        roundres = compute_metrics(METRICS, Scand, S, ycand, y, best_f, ymax)
        for metric, res in roundres.items():
            log.info(f" - {metric}: {res:.3f}")
        results.loc[t] = roundres
        results.to_csv(resultsfilename)

        # Augment data -- novel candidates only to avoid biasing the CPE
        ninds = novel_inds(Scand, S)
        S += [Scand[i] for i in ninds]
        X = torch.vstack((X, Xcand[ninds]))
        y = torch.concat((y, ycand[ninds]))
        z = torch.concat((z, ycand[ninds] > best_f))
        dataind = np.concatenate((dataind, np.ones(len(ninds), dtype=int) * t))

        # Don't bother training the model after the last round
        if t >= (t_rounds - 1):
            break

        log.info(f"Updating fitness threshold ...")
        best_f = thresh.update(y)
        log.info(f" new threshold: {best_f:.3f}")

        log.info(f"Updating surrogate model ...")
        if gp:
            update_acq_threshold(acq, best_f)
            update_gp(
                surrogate,
                X,
                y,
                refit=True,
                optimizer_options=config["gp"]["optimisation"],
                stop_options=config["gp"]["stop"],
                device=device,
            )
        else:
            fit_cpe(
                surrogate,
                X,
                y,
                thresh,
                batch_size=config["cpe"]["batchsize"],
                stop_options=config["cpe"]["stop"],
                optimizer_options=config["cpe"]["optimisation"],
                device=device,
                seed=seed,
            )

    # Save candidates, results and config
    candsfile = resultsdir / f"{methodname}_cands_{t}.csv"
    data.save_data(candsfile, S, y, {"round": dataind, "label": z})
    with open(resultsdir / f"{methodname}_config.json", "w") as f:
        json.dump(config, f)


def adapter(metricf: callable, args: list) -> callable:
    """Allow all of the metrics to be called with the same arguments"""

    def _metric(**kwargs):
        return float(metricf(*[kwargs[n] for n in args]))

    return _metric


def compute_metrics(metrics, Sc, S, yc, y, best_f, yx):
    roundres = pd.Series()
    for name, (f, args) in metrics.items():
        fun = adapter(f, args)
        roundres[name] = fun(Sc=Sc, S=S, yc=yc, y=y, best_f=best_f, yx=yx)
    return roundres


def get_fittest(y: np.ndarray | Tensor, S: SequenceArray) -> Tuple[float, str]:
    imax = np.argmax(y)
    ymax = y[imax]
    Smax = S[imax]
    return ymax, Smax


def plot_optimisation(iter, acquis, meangrad, ax=None):
    if ax is None:
        _, ax = plt.subplots(dpi=150)
    ax.plot(iter, meangrad)
    ax.set_xlabel("Iterations")
    ax.set_ylabel("Gradient mean")
    ax.grid()
    ax2 = ax.twinx()
    ax2.set_ylabel("Acquisition", color="tab:red")
    ax2.plot(iter, acquis, color="tab:red")
    ax2.tick_params(axis="y", labelcolor="tab:red")
    return ax


def update_acq_threshold(acq, best_f):
    if isinstance(acq, VariationalSearchAcquisition):
        acq.acq.best_f = torch.as_tensor(best_f)
    else:
        acq.best_f = torch.as_tensor(best_f)
