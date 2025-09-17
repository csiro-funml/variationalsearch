"""Train a class probability model for the experiments."""

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
from sklearn.metrics import balanced_accuracy_score, log_loss
from sklearn.model_selection import StratifiedKFold

from experiments.dataclasses import DATASETS, seq2intarr
from experiments.vis import CPplot
from vsd import cpe, labellers


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
    logfile = logdir / Path("cpe.log")
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

    # Get the first threshold
    thresh_class = getattr(labellers, config["threshold"]["class"])
    thresh = thresh_class(**config["threshold"]["args"])
    best_f = thresh.update(y)

    # Make the classifier data
    z = thresh(y)
    y = torch.tensor(y, dtype=torch.float32)
    nnzero = sum(z)
    log.info(f"Non-zero labels: #{nnzero}, proportion = {nnzero/len(z):.3f}")
    log.info(f"Maximum training data fitness = {y.max():.3f}")
    log.info(f"Threshold training data fitness = {best_f:.3f}")
    log.info(f"Minimum training data fitness = {y.min():.3f}")

    log.info("Tokenizing sequences ...")
    slen = data.get_seq_len()
    cats = data.get_alpha_len()
    X = seq2intarr(S, is_amino=cats > 4)

    log.info("Training CPE model ...")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=config["seed"])
    cpeplot = CPplot()
    baccs, lls = [], []
    device = config["device"]
    params = config["cpe"]["parameters"]
    CPE = getattr(cpe, config["cpe"]["class"])
    if not issubclass(CPE, cpe.ClassProbabilityModel):
        log.error(f"{CPE} is not a subclass of ClassProbabilityModel")
        sys.exit(-1)
    for r, (tind, sind) in enumerate(cv.split(S, z)):
        log.info(f"KFold round {r}:")
        Xt, Xs, yt = X[tind], X[sind], y[tind]
        zt, zs = z[tind], z[sind]
        clf = CPE(seq_len=slen, alpha_len=cats, **params).to(device)
        cpe.fit_cpe(
            clf,
            Xt,
            yt,
            labeller=best_f,
            batch_size=config["cpe"]["batchsize"],
            stop_options=config["cpe"]["stop"],
            optimizer_options=config["cpe"]["optimisation"],
            device=device,
        )
        pt = to_probability(clf(Xt.to(device)))
        ps = to_probability(clf(Xs.to(device)))
        cpeplot.update(zt, zs, pt, ps)
        ll = log_loss(zs, np.vstack((1 - ps, ps)).T)
        bacc = balanced_accuracy_score(zs, ps > 0.5)
        log.info(f" - bacc = {bacc:.3f}, ll = {ll:.3f}")
        lls.append(ll)
        baccs.append(bacc)

    log.info(f"Balanced accuracy: {np.mean(baccs):.3f} ({np.std(baccs):.3f}).")
    log.info(f"Log loss: {np.mean(lls):.3f} ({np.std(lls):.3f}).")
    cpeplot.plot(scoring=log_loss)
    plt.savefig(logdir / Path("cpe_probest.png"), dpi=200)
    plt.close()

    log.info("Training surrogate on all data ...")
    its, losses = [], []

    def callback(it, loss, _):
        its.append(it)
        losses.append(loss.detach().to("cpu").numpy())

    clf = CPE(seq_len=slen, alpha_len=cats, **params).to(device)
    cpe.fit_cpe(
        clf,
        X,
        y,
        labeller=best_f,
        batch_size=config["cpe"]["batchsize"],
        stop_options=config["cpe"]["stop"],
        optimizer_options=config["cpe"]["optimisation"],
        callback=callback,
        device=device,
        seed=config["seed"],
    )
    torch.save(clf.state_dict(), Path(config["cpe"]["path"]))
    plot_trace(
        its,
        losses,
        title="Final surrogate training",
        path=logdir / "cpe_surrogate_trace.png",
    )

    log.info("Done, saving config.")
    with open(logdir / "cpe_config_train.json", "w") as f:
        json.dump(config, f)
