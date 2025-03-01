"""Train a prior proposal distribution using ML on data directly."""

import json
import logging
import sys
from pathlib import Path
from typing import Sequence

import click
import matplotlib as mpl
import numpy as np

mpl.use("Agg")
import matplotlib.pyplot as plt
import torch

from experiments.dataclasses import DATASETS, seq2intarr
from vsd import proposals, thresholds


def plot_trace(
    its: Sequence,
    losses: Sequence,
    validation: Sequence,
    title: str,
    path: Path,
):
    plt.plot(its, losses, label="training loss")
    plt.plot(its, validation, label="validation loss")
    plt.title(title)
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.grid()
    plt.legend()
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
def train_prior(config, logdir, dataset, device):
    """Train a prior proposal distribution using maximum likelihood."""
    if config is None:
        config = DATASETS[dataset]["config"]
    else:
        config = json.load(Path(config))

    if not config["prior"]["trainable"]:
        print("Prior not marked as trainable, exiting.")
        sys.exit()

    if device is None:
        device = config["device"]
    else:
        config["device"] = device

    # Setup logging
    logdir = Path(logdir + f"_{dataset}")
    logdir.mkdir(exist_ok=True)
    logfile = logdir / Path("prior.log")
    logging.basicConfig(
        level=logging.INFO,
        handlers=[
            logging.FileHandler(logfile, mode="w"),
            logging.StreamHandler(),
        ],
    )
    log = logging.getLogger(name=__name__)

    # Making training data
    log.info("Making training data ...")
    torch.manual_seed(config["seed"])
    data = DATASETS[dataset]["data"]()
    S, y = data.load_data(config["training_data"]["save_path"])
    if config["prior"]["use_threshold"]:
        log.info("Thresholding prior training data...")
        thresh_class = getattr(thresholds, config["threshold"]["class"])
        thresh = thresh_class(**config["threshold"]["args"])
        best_f = thresh(y)
        pmask = y > best_f
        S = [S[i] for i, p in enumerate(pmask) if p]

    log.info("Making validation data ...")
    nval = config["training_data"]["size"]
    val_ind = np.random.randint(len(data.S), size=nval)
    S_val = [data.S[i] for i in val_ind]

    log.info("Tokenizing sequences ...")
    slen = data.get_seq_len()
    cats = data.get_alpha_len()
    X = seq2intarr(S, is_amino=cats > 4)
    X_val = seq2intarr(S_val, is_amino=cats > 4)

    log.info("Training prior ...")
    its, losses, validation = [], [], []

    def callback(it, loss, vloss):
        its.append(it)
        losses.append(loss.detach().to("cpu").numpy())
        validation.append(vloss.to("cpu").numpy())
        if it % 100 == 0:
            log.info(
                f" - iter: {it}, "
                f"train loss: {losses[-1]:.3f}, "
                f"validation loss: {validation[-1]:.3f}"
            )

    prior = getattr(proposals, config["prior"]["class"])
    prior = prior(slen, cats, **config["prior"]["parameters"])
    its, losses = [], []
    proposals.fit_ml(
        prior,
        X,
        X_val,
        batch_size=config["prior"]["batchsize"],
        optimizer_options=config["prior"]["optimisation"],
        stop_options=config["prior"]["stop"],
        device=config["device"],
        callback=callback,
        seed=config["seed"],
    )
    torch.save(prior.state_dict(), Path(config["prior"]["save_path"]))
    plot_trace(
        its,
        losses,
        validation,
        title="Prior training",
        path=logdir / "prior_trace.png",
    )

    log.info("Done, saving config.")
    with open(logdir / "config_prior.json", "w") as f:
        json.dump(config, f)
