"""Plot results of the Ehrlich BBO experiment."""

import sys
from glob import glob
from itertools import cycle
from pathlib import Path

import click
import matplotlib as mpl
import numpy as np

mpl.use("Agg")
import matplotlib.pyplot as plt

from experiments.metrics import diversity

LINECYCLE = cycle(
    [
        "-",
        "--",
        ":",
        "-.",
        (0, (3, 1, 1, 1)),
        (0, (5, 1)),
        (0, (3, 1, 1, 1, 1, 1)),
        (0, (1, 1)),
    ]
)

STR_REPS = {
    "vsd": "VSD",
    "cbas": "CbAS",
    "lambo2": "LaMBO-2",
    "ga": "GA",
    "tfm": "TFM",
    "lstm": "LSTM",
    "mf": "MF",
}


@click.command()
@click.option(
    "--resultsdir",
    type=click.Path(exists=True, file_okay=False),
    default="poli",
    help="Base poli results directory.",
)
@click.option(
    "--ystar",
    type=float,
    default=1.0,
    help="Ehrlich function maximum.",
)
@click.option(
    "--trainsize",
    type=int,
    default=128,
    help="Training data size.",
)
@click.option(
    "--batchsize",
    type=int,
    default=128,
    help="batch size for aggregating results.",
)
@click.option(
    "--fileprefix",
    type=str,
    default="",
    help="prefix for the output file names.",
)
def plot_results(resultsdir, ystar, trainsize, batchsize, fileprefix):
    basedir = Path(resultsdir)
    if not basedir.exists():
        print(f"Error: cannot find path {basedir}")
        sys.exit(-1)

    # Get all unique methods
    resfiles = glob((basedir / "*.npz").as_posix())
    methods = set([Path(f).stem.split("_")[0] for f in resfiles])
    method_files = {m: [f for f in resfiles if m in f] for m in methods}

    cycler = plt.cycler(
        color=plt.cm.viridis_r(np.linspace(0.05, 0.95, len(methods)))
    ) + plt.cycler(linestyle=[next(LINECYCLE) for _ in range(len(methods))])
    plt.rc("axes", prop_cycle=cycler)
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

    # Load results by method
    cumregrets = {}
    bdiversity = {}
    for m, files in method_files.items():
        if len(files) < 1:
            continue
        cumregrets[m] = []
        bdiversity[m] = []
        for f in files:
            d = np.load(f)
            regret = ystar - d["y"].flatten()
            cumregret = np.minimum.accumulate(regret)
            cumregrets[m].append(cumregret[(batchsize - 1) :: batchsize])
            X = np.array(["".join(x) for x in d["x"]])
            n = len(X)
            bdiversity[m].append(
                np.array(
                    [
                        diversity(X[i:j])
                        for i, j in zip(
                            range(0, n + 1 - batchsize, batchsize),
                            range(batchsize, n + 1, batchsize),
                        )
                    ]
                )
            )

    # Sort by performance
    means, dmeans = {}, {}
    stds, dstds = {}, {}
    regrets = []
    methods = []
    for m, ys in cumregrets.items():
        means[m] = np.mean(ys, axis=0)
        dmeans[m] = np.mean(bdiversity[m], axis=0)
        stds[m] = np.std(ys, axis=0)
        dstds[m] = np.std(bdiversity[m], axis=0)
        methods.append(m)
        regrets.append(means[m][-1])
    methods = np.array(methods)[np.argsort(regrets)[::-1]]

    # Plot by measure
    for p, f in (("Simple regret", "regret"), ("Diversity", "diversity")):
        fig, ax = plt.subplots(dpi=200)
        for m in methods:
            if f == "regret":
                mean, std = means[m], stds[m]
            else:
                mean, std = dmeans[m], dstds[m]
            x = np.arange(len(mean))
            ax.fill_between(x, mean - std, mean + std, alpha=0.1)
            for old, new in STR_REPS.items():
                m = m.replace(old, new)
            ax.plot(x, mean, label=m, linewidth=2)
            ax.set_xlabel("Round", fontsize=14)
        ax.set_ylabel(p, fontsize=14)
        ax.grid(True, linestyle="--", alpha=0.7)
        ax.legend(loc="best", frameon=True, framealpha=0.5, ncols=2)
        fig.tight_layout()
        fname = f"{f}.png"
        if fileprefix != "":
            fname = f"{fileprefix}-{fname}"
        plt.savefig(basedir / fname)
        plt.close()
