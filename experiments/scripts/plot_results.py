"""Plot results of the experiments."""

import sys
from glob import glob
from itertools import cycle
from pathlib import Path

import click
import matplotlib as mpl
import numpy as np
import pandas as pd

mpl.use("Agg")
import matplotlib.pyplot as plt

LINECYCLE = cycle(["--", ":", "-", "-."])


@click.command()
@click.option(
    "--resultsdir",
    type=click.Path(exists=True, file_okay=False),
    default="results",
    help="Base results directory.",
)
@click.option(
    "--datadir",
    type=click.Path(),
    default="DHFR",
    help="Directory of results for a specific dataset.",
)
@click.option("--logregret", is_flag=True, help="Plot regret on a log scale.")
def plot_results(resultsdir, datadir, logregret):
    basedir = Path(resultsdir) / datadir
    if not basedir.exists():
        print(f"Error: cannot find path {basedir}")
        sys.exit(-1)

    # Get all unique methods
    conffiles = glob((basedir / "seed*" / f"*_config.json").as_posix())
    conffiles = [Path(f).stem for f in conffiles]
    methods = sorted(set([f.replace("_config", "") for f in conffiles]))

    cycler = plt.cycler(
        color=plt.cm.viridis_r(np.linspace(0.1, 1.0, len(methods)))
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
    means = {}
    stdvs = {}
    for m in methods:
        resfiles = glob((basedir / "seed*" / f"*{m}_results.csv").as_posix())
        if len(resfiles) < 1:
            continue
        dfs = [pd.read_csv(f, index_col=0) for f in resfiles]
        ind, col = dfs[0].index, dfs[0].columns
        vals = [df.values for df in dfs]
        means[m] = pd.DataFrame(np.mean(vals, axis=0), index=ind, columns=col)
        stdvs[m] = pd.DataFrame(np.std(vals, axis=0), index=ind, columns=col)

    # Plot by measure
    for c in col:
        fig, ax = plt.subplots(dpi=200)
        for i, m in enumerate(methods):
            if m not in means:
                continue
            if c not in means[m]:
                continue
            x = means[m].index + i * 2e-2
            mean = means[m][c].values
            std = stdvs[m][c].values
            _, _, bars = ax.errorbar(
                x, mean, yerr=std, label=m, capsize=5, capthick=2, elinewidth=2
            )
            [bar.set_alpha(0.7) for bar in bars]
        ax.set_xticks(range(len(x)))
        ax.set_xticklabels(range(1, len(x) + 1))
        # ax.legend(loc='upper left', frameon=False,
        #           bbox_to_anchor=(0, 1), borderaxespad=0.1)
        ax.legend(loc="best", frameon=True, framealpha=0.5, ncols=2)
        ax.set_xlabel("Round", fontsize=14)
        ax.set_ylabel(c, fontsize=14)
        if c == "Simple regret" and logregret:
            ax.set_yscale("log")
        ax.grid(True, linestyle="--", alpha=0.7)
        fig.tight_layout()
        fname = f"{c}.png".replace(" ", "_").replace("(", "").replace(")", "")
        plt.savefig(basedir / fname)
        plt.close()
