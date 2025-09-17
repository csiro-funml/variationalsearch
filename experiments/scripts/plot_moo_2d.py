"""Plot results of the Ehrlich MOBO experiment."""

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
np.set_printoptions(suppress=True, precision=2)
import matplotlib.pyplot as plt
from botorch.utils.multi_objective.hypervolume import Hypervolume

from experiments.metrics import diversity
from vsd.utils import is_non_dominated_strict

LINECYCLE = cycle(
    [
        (0, (1, 1)),
        (0, (3, 1, 1, 1)),
        (0, (5, 1)),
        (0, (3, 1, 1, 1, 1, 1)),
        "-.",
        ":",
        "--",
        "-",
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
    "rf": "reinf.",
}


@click.command()
@click.option(
    "--resultsdir",
    type=click.Path(exists=True, file_okay=False),
    default="ehlich_nat",
    help="Base results directory.",
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
    default=32,
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
    type=(float, float),
    default=(-1.0, -1.0),
    help="Reference point for hypervolume computation",
)
@click.option(
    "--max-iter", type=int, default=None, help="Limit the number of iterations."
)
@click.option(
    "--bars", is_flag=True, help="Use bars for errors instead of a region fill."
)
@click.option(
    "--absvol",
    is_flag=True,
    help="Use absolute instead of relative hypervolume.",
)
@click.option(
    "--order",
    type=click.Choice(
        [
            "performance",
            "alpha",
        ]
    ),
    default="performance",
    help="Display order sorting for methods.",
)
def plot_results(
    resultsdir,
    trainsize,
    batchsize,
    fileprefix,
    ref,
    max_iter,
    bars,
    absvol,
    order,
):
    basedir = Path(resultsdir)
    if not basedir.exists():
        print(f"Error: cannot find path {basedir}")
        sys.exit(-1)

    # Get all unique methods
    resfiles = [
        Path(f)
        for f in glob((basedir / "*.npz").as_posix())
        if "samples" not in f
    ]
    if len(resfiles) < 1:
        print("Cannot find any results files!")
        sys.exit(-1)

    get_method = lambda x: x.stem.split("_")[0]
    methods = set([get_method(f) for f in resfiles])
    method_files = {
        m: [f for f in resfiles if m == get_method(f)] for m in methods
    }

    cycler = plt.cycler(
        color=plt.cm.inferno(np.linspace(0.05, 0.90, len(methods)))
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

    if max_iter is not None:
        trim = trainsize + max_iter * batchsize

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
            if max_iter is not None:
                y = y[:trim]
                X = X[:trim]
            n = len(y)

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

            # Plot a pretty scatter plot of objectives
            batchno = np.zeros(n)
            for i, (start, end) in enumerate(zip(starts, ends)):
                batchno[start:end] = i

            zall = is_non_dominated_strict(y)
            y_p = y[zall, :]
            sort = np.argsort(y_p[:, 0])
            y_p = y_p[sort, :]
            fig, ax = plt.subplots(dpi=200)
            plt.scatter(
                x=y[:, 0],
                y=y[:, 1],
                c=batchno,
                cmap="viridis",
                label="samples",
            )
            plt.plot(*y_p.T, "k", label="Pareto front")
            plt.grid()
            plt.colorbar(label="Batch no.")
            plt.xlabel("$f_1$")
            plt.ylabel("$f_2$")
            plt.legend()
            plt.tight_layout()
            plt.savefig(basedir / (Path(f).stem + "_scatter.png"))
            plt.close()

    # Aggregate measures
    means, dmeans, tmeans = {}, {}, {}
    stds, dstds, tmins, tmaxs = {}, {}, {}, {}
    hypervols = []
    methods = []
    for m, ys in hypervolumes.items():
        means[m] = np.mean(ys, axis=0)
        stds[m] = np.std(ys, axis=0)
        dmeans[m] = np.mean(bdiversity[m], axis=0)
        dstds[m] = np.std(bdiversity[m], axis=0)
        tmeans[m] = np.mean(times[m])
        tmins[m] = np.min(times[m])
        tmaxs[m] = np.max(times[m])
        methods.append(m)
        hypervols.append(means[m])

    # Display order sorting
    if order == "performance":
        disp_sort = np.argsort(np.vstack(hypervols)[:, -1])[::-1]
        methods = np.array(methods)[disp_sort]
    elif order == "alpha":
        methods.sort()

    # Plot by measure
    print("Last round results:")
    hname = "Relative Hypervolume" if not absvol else "Hypervolume"
    for p, f in (
        (hname, "hypervolume"),
        ("Diversity", "diversity"),
    ):
        fig, ax = plt.subplots(dpi=200)
        for i, m in enumerate(methods):
            if f == "hypervolume":
                mean, std = means[m], stds[m]
                print(f"{m} (t={len(mean)}): {mean[-1]:.3f} ({std[-1]:.3f})")
            else:
                mean, std = dmeans[m], dstds[m]
            x = np.arange(len(mean))
            for old, new in STR_REPS.items():
                m = m.replace(old, new)
            if not bars:
                ax.fill_between(x, mean - std, mean + std, alpha=0.1)
                ax.plot(x, mean, label=m, linewidth=2)
            else:
                _, _, bars = ax.errorbar(
                    x + i * 0.05,
                    mean,
                    yerr=std,
                    label=m,
                    capsize=5,
                    capthick=2,
                    elinewidth=2,
                )
                [bar.set_alpha(0.5) for bar in bars]
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
