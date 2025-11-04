"""Train a class Gaussian process surrogate model for the experiments."""

import json
import logging
import sys
from pathlib import Path
from typing import Sequence

import click
import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from botorch.models.gpytorch import GPyTorchModel
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold

from experiments.dataclasses import DATASETS, seq2intarr
from experiments.vis import YYplot
from vsd import surrogates


def to_probability(logp: torch.Tensor) -> np.ndarray:
    return np.exp(logp.detach().to("cpu").numpy().squeeze())


def plot_trace(its: Sequence, losses: Sequence, title: str, path: Path):
    plt.plot(its, losses)
    plt.title(title)
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.grid()
    plt.savefig(path, dpi=200)
    plt.close()


@click.command()
@click.option(
    "--config",
    type=click.Path(dir_okay=False),
    default=None,
    help="config file to use.",
)
@click.option(
    "--logdir",
    type=click.Path(file_okay=False),
    default="train",
    help="training log directory.",
)
@click.option(
    "--dataset",
    default="GFP",
    help="dataset for the experiment",
    type=click.Choice(DATASETS.keys(), case_sensitive=False),
)
@click.option(
    "--device", type=str, default=None, help="Override device config setting."
)
def train_surrogate(config, logdir, dataset, device):
    if config is None:
        config = DATASETS[dataset]["config"]
    else:
        config = json.load(Path(config))

    if device is None:
        device = config["device"]
    else:
        config["device"] = device

    # Setup logging
    logdir = Path(logdir + f"_{dataset}")
    logdir.mkdir(exist_ok=True)
    logfile = logdir / Path("gp.log")
    logging.basicConfig(
        level=logging.INFO,
        handlers=[
            logging.FileHandler(logfile, mode="w"),
            logging.StreamHandler(),
        ],
    )
    log = logging.getLogger(name=__name__)

    # Making training data -- needed for evaluation too, so also saving
    log.info("Making training data ...")
    torch.manual_seed(config["seed"])
    data = DATASETS[dataset]["data"]()
    S, y = data.make_training_data(
        seed=config["seed"], **config["training_data"]
    )
    log.info(f"Maximum training data fitness = {y.max():.3f}")
    log.info(f"Minimum training data fitness = {y.min():.3f}")

    log.info("Tokenizing sequences ...")
    slen = data.get_seq_len()
    cats = data.get_alpha_len()
    X = seq2intarr(S, is_amino=cats > 4)
    y = torch.tensor(y, dtype=torch.float32)

    log.info("Training surrogate model ...")
    cv = KFold(n_splits=5, shuffle=True, random_state=config["seed"])
    yyplot = YYplot()
    r2s = []
    device = config["device"]
    params = config["gp"]["parameters"]
    GP = getattr(surrogates, config["gp"]["class"])
    if not issubclass(GP, GPyTorchModel):
        log.error(f"{GP} is not a subclass of GPyTorchModel")
        sys.exit(-1)

    for r, (tind, sind) in enumerate(cv.split(X)):
        log.info(f"KFold round {r}:")
        Xt, Xs, yt, ys = X[tind], X[sind], y[tind], y[sind]
        gp = GP(seq_len=slen, alpha_len=cats, X=Xt, y=yt, **params)
        surrogates.fit_gp(
            gp,
            optimiser_options=config["gp"]["optimisation"],
            device=device,
        )
        eys = gp(Xs.to(device)).mean.detach().numpy()
        eyt = gp(Xt.to(device)).mean.detach().numpy()
        yyplot.update(yt, ys, eyt, eys)
        r2 = r2_score(ys, eys)
        log.info(f" - r2 = {r2:.3f}")
        r2s.append(r2)

    log.info(f"R2 score: {np.mean(r2s):.3f} ({np.std(r2s):.3f}).")
    yyplot.plot()
    plt.savefig(logdir / Path("gp_probest.png"), dpi=200)
    plt.close()

    log.info("Training surrogate on all data ...")
    its, losses = [], []

    def callback(params, optres):
        its.append(optres.step)
        losses.append(optres.fval)

    gp = GP(seq_len=slen, alpha_len=cats, X=X, y=y, **params)
    surrogates.fit_gp(
        gp,
        optimiser_options=config["gp"]["optimisation"],
        callback=callback,
        device=device,
    )
    torch.save(gp.state_dict(), Path(config["gp"]["path"]))
    plot_trace(
        its,
        losses,
        title="Final surrogate training",
        path=logdir / "gp_surrogate_trace.png",
    )

    log.info("Done, saving config.")
    with open(logdir / "gp_config_train.json", "w") as f:
        json.dump(config, f)
