"""Plot results of the N-Grams MOBO experiment."""

import re
import sys
from glob import glob
from itertools import cycle
from pathlib import Path

import click
import matplotlib as mpl
import numpy as np
import pandas as pd
import torch

mpl.use("Agg")
import matplotlib.pyplot as plt

from botorch.utils.multi_objective.hypervolume import Hypervolume
from experiments.metrics import diversity
from vsd.utils import is_non_dominated_strict

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
    "agps": "A-GPS",
    "vsd": "VSD",
    "cbas": "CbAS",
    "lambo2": "LaMBO-2",
    "ga": "GA",
    "tfm": "TFM",
    "lstm": "LSTM",
    "mf": "MF",
    "rand": "Random (greedy)",
}


@click.command()
@click.option(
    "--resultsdir",
    type=click.Path(exists=True, file_okay=False),
    default="ngrammoo",
    help="Base ngrams results directory.",
)
@click.option(
    "--trainsize",
    type=int,
    default=512,
    help="Training data size.",
)
@click.option(
    "--batchsize",
    type=int,
    default=16,
    help="batch size for aggregating results.",
)
@click.option(
    "--fileprefix",
    type=str,
    default="",
    help="prefix for the output file names.",
)
@click.option(
    "--ref",
    type=(float, float, float),
    default=(0.0, 0.0, 0.0),
    help="Reference point for hypervolume computation",
)
@click.option(
    "--absvol",
    is_flag=True,
    help="Use absolute instead of relative hypervolume.",
)
def plot_results(resultsdir, trainsize, batchsize, fileprefix, ref, absvol):
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

    hv = Hypervolume(ref_point=torch.tensor(ref))

    # Load results by method
    hypervolumes = {}
    bdiversity = {}
    times = {}
    for m, files in method_files.items():
        if len(files) < 1:
            continue
        hypervolumes[m] = []
        bdiversity[m] = []
        times[m] = []
        for f in files:
            hypervolumes[m].append([])
            bdiversity[m].append([])
            d = np.load(f)
            y = torch.tensor(d["y"])
            X = np.array(["".join(x) for x in d["x"]])
            n = len(X)

            # First use a trainsize batch, then subsequent batches of size batchsize
            starts = [0] + list(range(trainsize, n, batchsize))
            ends = [trainsize] + [
                min(s + batchsize, n) for s in range(trainsize, n, batchsize)
            ]
            for start, end in zip(starts, ends):
                yb = y[:end, :]
                zb = is_non_dominated_strict(yb)
                hvol = hv.compute(yb[zb == 1, :])
                if len(hypervolumes[m][-1]) < 1:
                    init_hvol = hvol
                hvol = hvol / init_hvol if not absvol else hvol
                hypervolumes[m][-1].append(hvol)
                bdiversity[m][-1].append(diversity(X[np.arange(start, end)]))

            # Load times if they exist
            flog = Path(f).with_suffix(".log")
            with open(flog, "r") as file:
                lines = file.readlines()
            times[m].append(np.nan)
            for line in lines:
                match = re.search(
                    r"elapsed time\s*[:=]\s*([\d.]+)", line, re.IGNORECASE
                )
                if match:
                    times[m][-1] = float(match.group(1))
                    break

    # Sort by performance
    means, dmeans = {}, {}
    stds, dstds = {}, {}
    hypervols = []
    methods = []
    tmeans, tmins, tmaxs = {}, {}, {}
    for m, ys in hypervolumes.items():
        means[m] = np.mean(ys, axis=0)
        dmeans[m] = np.mean(bdiversity[m], axis=0)
        stds[m] = np.std(ys, axis=0)
        dstds[m] = np.std(bdiversity[m], axis=0)
        tmeans[m] = np.mean(times[m])
        tmins[m] = np.min(times[m])
        tmaxs[m] = np.max(times[m])
        methods.append(m)
        hypervols.append(means[m])
    methods = np.array(methods)[np.argsort(np.vstack(hypervols)[:, -1])[::-1]]

    # Plot by measure
    print("Last round results:")
    hname = "Relative Hyper-volume" if not absvol else "Hyper-volume"
    for p, f in ((hname, "hypervolume"), ("Diversity", "diversity")):
        fig, ax = plt.subplots(dpi=200)
        for m in methods:
            if f == "hypervolume":
                mean, std = means[m], stds[m]
                print(f"{m} (t={len(mean)}): {mean[-1]:.3f} ({std[-1]:.3f})")
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

    # Save times as a CSV
    print("Times")
    dftimes = pd.DataFrame(data=dict(mean=tmeans, min=tmins, max=tmaxs))
    newind = []
    for i in dftimes.index:
        newitem = str(i)
        for old, new in STR_REPS.items():
            newitem = newitem.replace(old, new)
        newind.append(newitem)
    dftimes.index = pd.Index(newind)
    dftimes.sort_index(inplace=True)
    dftimes.to_csv(basedir / "times.csv", sep="&", float_format="%.2f")
    print(dftimes)
