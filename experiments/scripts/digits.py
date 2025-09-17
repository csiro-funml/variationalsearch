"""Handwritten digits generation experiment."""

import logging
from copy import deepcopy
from itertools import cycle
from pathlib import Path
from typing import Any, Callable, Optional, Tuple

import click
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from sklearn.metrics import balanced_accuracy_score
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST

from vsd.acquisition import (
    LogPIClassifierAcquisition,
    VariationalSearchAcquisition,
)
from vsd.generation import generate_candidates_reinforce
from vsd.proposals import (
    DTransformerProposal,
    LSTMProposal,
    SearchDistribution,
    clip_gradients,
)
from vsd.cpe import CNNClassProbability

mpl.use("Agg")

# Settings & Constants
ALPHA = 8
LABELS = [3, 5]
IMSIZE = 14
SEQLEN = IMSIZE**2
CPE_LR = 1e-3
PRIOR_LR = 1e-3
PRIOR_REG = 1e-3
VSD_LR = 1e-4
CPE_REG = 0.0
BATCHSIZE = 512
TSBATCHSIZE = 512
LOG_ITS = 1000
CPE_MAXITER = 10000
# PRIOR_MAXITER = 70000
PRIOR_MAXITER = 35000
VSD_MAXITER = 30000
VSD_NWINDOW = 5000
NUMWORKERS = 0
VSD_SAMPLES = 512
VSD_LSTM_REG = 0.0

CPE_PARAMS = dict(
    ckernel=7,
    xkernel=2,
    xstride=2,
    cfilter_size=16,
    linear_size=128,
    dropoutp=0.2,
)

DTFM_PARAMS = dict(
    nhead=8, num_layers=4, dim_feedforward=256, clip_gradients=1.0
)

LSTM_PARAMS = dict(hidden_size=128, num_layers=5, clip_gradients=1.0)


# Image flattening indices ("unrolling")
FINDS = np.arange(SEQLEN).reshape(IMSIZE, IMSIZE)
FINDS[1::2] = FINDS[1::2, ::-1]
FINDS = FINDS.flatten()


@click.command
@click.option("--seed", type=int, default=42, help="custom seed.")
@click.option(
    "--device", type=str, default="cpu", help="device to use for experiment."
)
@click.option(
    "--resultsdir",
    type=click.Path(file_okay=False),
    default="results",
    help="experiment results directory.",
)
@click.option(
    "--cachedir",
    type=click.Path(file_okay=False),
    default="models",
    help="classifier and prior model cache directory.",
)
@click.option(
    "--cachedir",
    type=click.Path(file_okay=False),
    default="models",
    help="classifier and prior model cache directory.",
)
@click.option(
    "-lstm", is_flag=True, help="use an LSTM instead of a transformer."
)
@click.option(
    "-relearn", is_flag=True, help="ignore cached models, re-learn them."
)
def digits(seed, device, resultsdir, cachedir, relearn, lstm):
    torch.manual_seed(seed)

    # Model cache
    cpe_cache = Path(cachedir) / "digits_cpe.pt"
    if lstm:
        prior_name = "LSTM"
        prior_cache = Path(cachedir) / "digits_prior_lstm.pt"
    else:
        prior_name = "DTransformer"
        prior_cache = Path(cachedir) / "digits_prior_dtfm.pt"

    # Setup logging
    resultsdir = Path(resultsdir)
    resultsdir.mkdir(exist_ok=True, parents=True)
    logfile = resultsdir / f"digits_{prior_name}.log"
    logging.basicConfig(
        level=logging.INFO,
        handlers=[
            logging.FileHandler(logfile, mode="w"),
            logging.StreamHandler(),
        ],
    )
    log = logging.getLogger(name=__name__)

    if (not cpe_cache.exists()) or (not prior_cache.exists()) or relearn:
        log.info("Loading digits data")
        trainloader, testloader, prior_trainloader, prior_testloader = (
            digits_data()
        )

    log.info("Making CPE")
    clf = CNNClassProbability(seq_len=SEQLEN, alpha_len=ALPHA, **CPE_PARAMS).to(
        device
    )

    if (not cpe_cache.exists()) or relearn:
        log.info("Training CPE")
        train_cpe(
            clf,
            trainloader,
            testloader,
            device,
            dict(lr=CPE_LR, weight_decay=CPE_REG),
            CPE_MAXITER,
            log,
            LOG_ITS,
        )
        torch.save(clf.state_dict(), cpe_cache)
    else:
        clf.load_state_dict(
            torch.load(
                cpe_cache, map_location=torch.device(device), weights_only=True
            )
        )
        clf.eval()

    log.info(f"Making {prior_name} prior")
    if lstm:
        prior = LSTMProposal(
            d_features=SEQLEN, k_categories=ALPHA, **LSTM_PARAMS
        ).to(device)
    else:
        prior = DTransformerProposal(
            d_features=SEQLEN, k_categories=ALPHA, **DTFM_PARAMS
        ).to(device)

    if (not prior_cache.exists()) or relearn:
        log.info(f"Fitting prior {prior_name} to data")
        train_proposal(
            prior,
            prior_trainloader,
            prior_testloader,
            device,
            dict(lr=PRIOR_LR, weight_decay=PRIOR_REG),
            PRIOR_MAXITER,
            log,
            LOG_ITS,
        )
        torch.save(prior.state_dict(), prior_cache)
    else:
        prior.load_state_dict(
            torch.load(
                prior_cache,
                map_location=torch.device(device),
                weights_only=True,
            )
        )
        prior.eval()

    log.info("Visualising samples from prior")
    with torch.no_grad():
        Xp = prior.sample(torch.Size([64]))
        scores = clf(Xp)
    savefile = resultsdir / f"prior_digit_samples_{prior_name}"
    save_digits(Xp, savefile.with_suffix(".npz"), reroll)
    plot_64_digits(Xp, reroll, scores)
    plt.savefig(savefile.with_suffix(".png"))
    plt.close()

    log.info("Training proposal function")
    prop = deepcopy(prior).to(device)

    # Make sure we turn off prior training
    for p in prior.parameters():
        p.grad = None
        p.requires_grad = False

    acq = LogPIClassifierAcquisition(model=clf).to(device)
    pracq = VariationalSearchAcquisition(acq, prior).to(device)

    def callback(i: int, loss: torch.Tensor, grad: Tuple[torch.Tensor]):
        if i % 100 == 0:
            gmean = np.mean([g.detach().to("cpu").mean() for g in grad])
            nloss = loss.detach().to("cpu").numpy()
            log.info(f" - {i}: loss = {nloss:.3f}, mean grad = {gmean:.3f}")

    reg = VSD_LSTM_REG if lstm else 0.0
    Xc, _ = generate_candidates_reinforce(
        proposal_distribution=prop,
        acquisition_function=pracq,
        callback=callback,
        stop_options=dict(maxiter=VSD_MAXITER, n_window=VSD_NWINDOW),
        optimizer_options=dict(lr=VSD_LR, weight_decay=reg),
        cv_smoothing=0.7,
        gradient_samples=VSD_SAMPLES,
        candidate_samples=64,
    )

    log.info("Visualising samples from variational distribution")
    savefile = resultsdir / f"vsd_digit_samples_{prior_name}"
    save_digits(Xc, savefile.with_suffix(".npz"), reroll)
    plot_64_digits(Xc, reroll, clf(Xc))
    plt.savefig(savefile.with_suffix(".png"))
    plt.close()

    log.info("Finished.")


class DigitsOnly(MNIST):

    def __init__(
        self,
        root: str | Path,
        train: bool = True,
        transform: Callable[..., Any] | None = None,
        download: bool = False,
    ) -> None:
        super().__init__(root, train, transform, None, download)

    def __getitem__(self, index: int) -> Tuple[Any]:
        x, _ = super().__getitem__(index)
        return x


def digits_data() -> Tuple[DataLoader, DataLoader, DataLoader, DataLoader]:
    data_tr = MNIST(
        root="data/",
        download=True,
        train=True,
        transform=image_transform,
        target_transform=target_transform,
    )
    data_ts = MNIST(
        root="data/",
        download=True,
        train=False,
        transform=image_transform,
        target_transform=target_transform,
    )
    trainloader = DataLoader(
        data_tr, batch_size=BATCHSIZE, shuffle=True, num_workers=NUMWORKERS
    )
    testloader = DataLoader(
        data_ts, batch_size=BATCHSIZE, num_workers=NUMWORKERS
    )
    data_prior = DigitsOnly(
        root="data/",
        download=True,
        train=True,
        transform=image_transform,
    )
    data_ptest = DigitsOnly(
        root="data/",
        download=True,
        train=False,
        transform=image_transform,
    )
    # Subset data
    # ptr_ind = [i for i, (_, y) in enumerate(data_tr) if y == 0]
    # pts_ind = [i for i, (_, y) in enumerate(data_ts) if y == 0]
    # data_prior = Subset(data_prior, ptr_ind),
    # data_ptest = Subset(data_ptest, pts_ind),
    prior_trainloader = DataLoader(
        data_prior,
        batch_size=TSBATCHSIZE,
        shuffle=True,
        num_workers=NUMWORKERS,
    )
    prior_testloader = DataLoader(
        data_ptest,
        batch_size=TSBATCHSIZE,
        num_workers=NUMWORKERS,
    )
    return trainloader, testloader, prior_trainloader, prior_testloader


def unroll(x: np.ndarray) -> np.ndarray:
    x = x.flatten()
    return x[FINDS]


def reroll(u: np.ndarray) -> np.ndarray:
    return u[FINDS].reshape(IMSIZE, IMSIZE)


def image_transform(I: Image) -> torch.Tensor:
    x = unroll(np.array(I.quantize(colors=ALPHA).resize((IMSIZE, IMSIZE))))
    return torch.tensor(x, dtype=torch.int)


def target_transform(y: int) -> int:
    return torch.tensor(y in LABELS, dtype=torch.float32)


def train_cpe(
    clf: torch.nn.Module,
    trainloader: DataLoader,
    testloader: DataLoader,
    device: str,
    optim_options: dict,
    maxiter: int,
    log: logging.Logger,
    test_it: int = 100,
):
    optim = torch.optim.Adam(clf.parameters(), **optim_options)
    lossfn = torch.nn.BCELoss()

    clf.train()
    for i, (X, y) in enumerate(cycle(trainloader)):
        X, y = X.to(device), y.to(device)
        loss = lossfn(torch.exp(clf(X)), y)
        loss.backward()
        optim.step()
        optim.zero_grad()

        if i % test_it == 0:
            clf.eval()
            with torch.no_grad():
                trloss = loss.detach()
                ys, pys, tsloss = [], [], 0
                for b, (Xs, y) in enumerate(testloader):
                    Xs, y = Xs.to(device), y.to(device)
                    ys.append(y)
                    pys.append(torch.exp(clf(Xs)))
                    tsloss += lossfn(pys[b], ys[b])
                tsloss /= b + 1
                ys, pys = torch.concat(ys), torch.concat(pys)
                bacc = balanced_accuracy_score(
                    ys.to("cpu"), pys.to("cpu") > 0.5
                )
                log.info(
                    f" - {i}: train loss = {trloss:.3f}, "
                    f"test loss = {tsloss:.3f}, "
                    f"balanced accuracy = {bacc:.3f}"
                )
            clf.train()

        if i > maxiter:
            break
    clf.eval()


def train_proposal(
    prop: SearchDistribution,
    trainloader: DataLoader,
    testloader: Optional[DataLoader],
    device: str,
    optim_options: dict,
    maxiter: int,
    log: logging.Logger,
    test_it: int = 100,
):
    clip_gradients(prop)
    optim = torch.optim.Adam(prop.parameters(), **optim_options)
    prop.train()
    for i, X in enumerate(cycle(trainloader)):
        X = X.to(device)
        loss = -prop.log_prob(X).mean()
        loss.backward()
        optim.step()
        optim.zero_grad()

        if i % test_it == 0:
            prop.eval()
            with torch.no_grad():
                trloss = loss.detach()
                lossstr = f" - {i}: train loss = {trloss:.3f}"
                if testloader is not None:
                    tsloss = 0
                    for b, Xs in enumerate(testloader):
                        Xs = Xs.to(device)
                        tsloss -= prop.log_prob(Xs).mean()
                    tsloss /= b + 1
                    lossstr = lossstr + f", test loss = {tsloss:.3f}"
                log.info(lossstr)
            prop.train()

        if i > maxiter:
            break
    prop.eval()


def to_probability(logp: torch.Tensor) -> np.ndarray:
    return np.exp(logp.detach().to("cpu").numpy().squeeze())


def plot_64_digits(D, reroll, scores=None):
    fig, axs = plt.subplots(8, 8, dpi=150, figsize=(8, 8))
    for xc, ax in zip(D, axs.flatten()):
        xc = xc.detach().to("cpu").numpy()
        I = reroll(xc)
        ax.imshow(I, cmap=plt.cm.gray, interpolation="nearest")
        ax.set_axis_off()
    if scores is not None:
        meanscore = np.mean(np.exp(scores.to("cpu").detach().numpy()))
        fig.suptitle(f"Sample mean CPE p={meanscore:.3f}")
    fig.tight_layout()


def save_digits(digits: torch.Tensor, path: Path, tfm=None):
    d = digits.detach().to("cpu").numpy()
    if tfm is not None:
        d = np.vstack([tfm(i) for i in d])
    np.savez(path, digits=d)
