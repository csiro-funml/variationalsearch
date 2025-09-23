### Synthetic function MOO experiments.###

import warnings
from itertools import cycle
from functools import partial
import random
import logging
from pathlib import Path

import click
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as fnn
from botorch import fit_gpytorch_mll
from botorch.acquisition.monte_carlo import qNoisyExpectedImprovement
from botorch.acquisition.multi_objective.monte_carlo import (
    qExpectedHypervolumeImprovement,
    qNoisyExpectedHypervolumeImprovement,
)
from botorch.acquisition.multi_objective.objective import (
    IdentityMCMultiOutputObjective,
)
from botorch.models import SingleTaskGP
from botorch.models.model_list_gp_regression import ModelListGP
from botorch.optim import optimize_acqf
from botorch.sampling.normal import SobolQMCNormalSampler
from botorch.test_functions.multi_objective import (
    DTLZ2,
    DTLZ7,
    ZDT3,
    BraninCurrin,
    GMM,
)
from botorch.utils.multi_objective.box_decompositions import (
    NondominatedPartitioning,
)
from botorch.utils.multi_objective.hypervolume import (
    Hypervolume,
    infer_reference_point,
)
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.mlls.sum_marginal_log_likelihood import SumMarginalLogLikelihood
from scipy.stats.qmc import LatinHypercube
from sklearn.decomposition import PCA
from torch import Size

from vsd.acquisition import VariationalPreferenceAcquisition
from vsd.condproposals import (
    ConditionalGaussianProposal,
    PreferenceSearchDistribution,
    fit_ml as fit_ml_cond,
)
from vsd.cpe import (
    PreferenceContinuousCPE,
    fit_cpe_labels,
    make_contrastive_alignment_data,
)
from vsd.generation import generate_candidates_iw
from vsd.preferences import EmpiricalPreferences, MixtureUnitNormal, UnitNormal
from vsd.proposals import fit_ml
from vsd.utils import is_non_dominated_strict
from vsd.labellers import ParetoAnnealed

warnings.filterwarnings("ignore")
mpl.use("Agg")
# torch.autograd.set_detect_anomaly(True)

# Settings and constants

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

# Config
SAMPLES = 256
NRESTARTS = 10
START_PERCENTILE = 0.75
GMM_K = 5
TOPK = False
PRIOR_VAL_PROP = 0.05
PRIOR_MAXITER = 4000
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

obj_mappings = {
    "DTLZ2": dict(
        f=DTLZ2,
        args=dict(dim=3, num_objectives=2, negate=True),
        ref_point=None,
    ),
    "DTLZ2-4": dict(
        f=DTLZ2,
        args=dict(dim=5, num_objectives=4, negate=True),
        ref_point=None,
    ),
    "DTLZ7": dict(
        f=DTLZ7,
        args=dict(dim=7, num_objectives=6, negate=True),
        ref_point=None,
    ),
    "BraninCurrin": dict(
        f=BraninCurrin, args=dict(negate=False), ref_point=None
    ),
    "ZDT3": dict(
        f=ZDT3,
        args=dict(dim=4, num_objectives=2, negate=True),
        ref_point=None,
    ),
    "GMM": dict(
        f=GMM, args=dict(num_objectives=2, negate=True), ref_point=None
    ),
}

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

#
# GP functions
#


def sanitize_inputs(X, Y):
    mask = ~torch.isnan(Y).any(dim=1)
    return X[mask], Y[mask]


def fit_gp_moo_model(X, Y):
    models = [SingleTaskGP(X, Y[:, i : i + 1]) for i in range(Y.shape[-1])]
    model = ModelListGP(*models)
    mll = SumMarginalLogLikelihood(model.likelihood, model)
    fit_gpytorch_mll(mll)
    return model


def fit_gp_sca_model(train_X, train_Y):
    M = train_Y.shape[1]
    weights = torch.rand(M)
    weights /= weights.sum()
    scalar_Y = (train_Y * weights).sum(dim=-1, keepdim=True)
    scalar_gp = SingleTaskGP(train_X, scalar_Y)
    mll = ExactMarginalLogLikelihood(scalar_gp.likelihood, scalar_gp)
    fit_gpytorch_mll(mll)
    return scalar_gp


def opt_acq_fn_wrapper(acq_func, bsize, D):
    candidates, _ = optimize_acqf(
        acq_function=acq_func,
        bounds=torch.stack([torch.zeros(D), torch.ones(D)]),
        q=bsize,
        num_restarts=NRESTARTS,
        raw_samples=SAMPLES,
        sequential=True,
        options={"batch_limit": bsize + 1, "maxiter": 200},
    )
    return candidates


def generate_candidates_qehvi(model, train_X, train_Y, ref_point, bsize):
    D = train_X.shape[1]
    sampler = SobolQMCNormalSampler(sample_shape=Size([SAMPLES]))
    partitioning = NondominatedPartitioning(ref_point=ref_point, Y=train_Y)
    acq_func = qExpectedHypervolumeImprovement(
        model=model,
        ref_point=ref_point.tolist(),
        partitioning=partitioning,
        sampler=sampler,
        objective=IdentityMCMultiOutputObjective(
            outcomes=list(range(train_Y.shape[-1]))
        ),
    )

    return opt_acq_fn_wrapper(acq_func, bsize, D)


def generate_candidates_qnehvi(model, train_X, train_Y, ref_point, bsize):
    D = train_X.shape[1]
    M = train_Y.shape[1]
    samples = min(SAMPLES, 64) if M > 4 else SAMPLES
    sampler = SobolQMCNormalSampler(sample_shape=torch.Size([samples]))
    acq_func = qNoisyExpectedHypervolumeImprovement(
        model=model,
        X_baseline=train_X,
        ref_point=ref_point.tolist(),
        sampler=sampler,
        objective=IdentityMCMultiOutputObjective(
            outcomes=list(range(train_Y.shape[-1]))
        ),
        prune_baseline=True,
    )

    return opt_acq_fn_wrapper(acq_func, bsize, D)


def generate_candidates_nparego(model, train_X, train_Y, ref_point, bsize):
    D = train_X.shape[1]
    acq_func = qNoisyExpectedImprovement(
        model=model,
        X_baseline=train_X,
        sampler=SobolQMCNormalSampler(sample_shape=torch.Size([SAMPLES])),
    )

    return opt_acq_fn_wrapper(acq_func, bsize, D)


#
# A-GPS implementation
#


def vsd_callback(i, loss, grads):
    if (i % 100) == 0:
        gmean = sum([g.detach().mean() for g in grads if g is not None]) / len(
            grads
        )
        print(f"  {i}: loss = {loss:.3f}, mean grad = {gmean:.3f}")


def callback(i, loss, vloss, logvloss=False):
    if (i % 100) == 0:
        if not logvloss:
            print(f"  {i}: loss = {loss:.3f}")
        else:
            print(f"  {i}: loss = {loss:.3f}, vloss = {vloss:.3f}")


class AGPS:

    def __init__(
        self, train_X, train_Y, ref_point, bsize, max_iter, preference, device
    ):
        D = train_X.shape[1]
        M = train_Y.shape[1]
        self.eps = 1e-6
        if preference == "empirical":
            self.preferences = EmpiricalPreferences()
        elif preference == "gmm":
            mu = torch.randn((GMM_K, M))
            self.preferences = MixtureUnitNormal(locs=mu, min_scale=self.eps)
        elif preference == "normal":
            self.preferences = UnitNormal(dim=M)
        else:
            raise ValueError(f"Unknown prefernce option {preference}")
        self.pareto_cpe = PreferenceContinuousCPE(
            x_dim=D,
            u_dims=M,
            latent_dim=min(16 * D, 128),
            dropoutp=0.1,
            hidden_layers=2,
        )
        self.preference_cpe = PreferenceContinuousCPE(
            x_dim=D,
            u_dims=M,
            latent_dim=min(16 * D, 128),
            dropoutp=0.1,
            hidden_layers=2,
        )
        prior = torch.distributions.Independent(
            torch.distributions.Uniform(
                low=(torch.zeros(D) - self.eps).to(device),
                high=(torch.ones(D) + self.eps).to(device),
            ),
            1,
        )
        cproposal = ConditionalGaussianProposal(
            x_dims=D,
            u_dims=M,
            latent_dim=min(16 * D, 256),
            hidden_layers=2 if D < 3 else 4,
            bias=True,
            dropout=0.2,
            low_rank_dim=0,
            x_invtransform=lambda x: torch.clamp(x, min=0, max=1),
            # x_transform=lambda x: torch.logit(x, eps=self.eps),
            # x_invtransform=lambda x: torch.sigmoid(x),
        )

        print("Initialising A-GPS q(x|u)")
        U = fnn.normalize(train_Y - ref_point, p=2, dim=1, eps=self.eps)
        fit_ml_cond(
            cproposal,
            X=train_X,
            U=U,
            optimizer_options=dict(lr=1e-3, weight_decay=1e-6),
            stop_options=dict(maxiter=PRIOR_MAXITER),
            batch_size=32,
            callback=partial(callback, logvloss=True),
            device=device,
            val_proportion=PRIOR_VAL_PROP,
        )
        cproposal.set_dropout_p(0.0)

        self.proposal = PreferenceSearchDistribution(
            cproposal=cproposal, preference=self.preferences
        )
        self.acq = VariationalPreferenceAcquisition(
            pareto_model=self.pareto_cpe,
            pref_model=self.preference_cpe,
            prior_dist=prior,
        )
        self.labeller = ParetoAnnealed(percentile=START_PERCENTILE, T=max_iter)
        self.ref_point = ref_point
        self.bsize = bsize
        self.device = device

    def fit(self, X, y, round):
        print("Running A-GPS...")
        z = torch.tensor(self.labeller(y), dtype=torch.float)
        U = fnn.normalize(y - self.ref_point, p=2, dim=1, eps=self.eps)

        # Augment dataset with misalignments
        Xa, Ua, za = make_contrastive_alignment_data(X, U)
        if isinstance(self.preferences, EmpiricalPreferences):
            self.preferences.set_preferences(U[z == 1, :] if round > 0 else U)
        else:
            print("Fitting preferences.")
            fit_ml(
                self.preferences,
                U[z == 1, :] if round > 0 else U,
                optimizer_options=dict(lr=1e-3, weight_decay=1e-8),
                callback=callback,
                batch_size=32,
                device=self.device,
            )
        cpe_opt_options = dict(lr=1e-3, weight_decay=1e-6)
        cpe_stop_options = dict(k=0.1)
        print("Fitting Pareto CPE.")
        print(f"Positive label p = {z.mean():.3f}")
        fit_cpe_labels(
            self.pareto_cpe,
            X,
            z,
            U,
            optimizer_options=cpe_opt_options,
            stop_options=cpe_stop_options,
            callback=callback,
            batch_size=32,
            device=self.device,
        )
        print("Fitting Alignment CPE.")
        fit_cpe_labels(
            self.preference_cpe,
            Xa,
            za,
            Ua,
            optimizer_options=cpe_opt_options,
            stop_options=cpe_stop_options,
            callback=callback,
            batch_size=32,
            device=self.device,
        )
        print("Fitting AGPS.")
        generate_candidates_iw(
            self.acq,
            self.proposal,
            optimizer_options=dict(lr=1e-5),
            stop_options=dict(miniter=3000),
            gradient_samples=SAMPLES,
            callback=vsd_callback,
        )
        if TOPK:
            Xs, Us = self.proposal.sample(torch.Size([SAMPLES]))
            asort = torch.argsort(self.pareto_cpe(Xs, Us), descending=True)
            Xs, Us = Xs[asort[: self.bsize]], Us[asort[: self.bsize]]
        else:
            Xs, Us = self.proposal.sample(torch.Size([self.bsize]))
        return Xs.cpu(), Us.cpu()


@click.command()
@click.option(
    "--bbox",
    type=click.Choice(obj_mappings.keys()),
    default="BraninCurrin",
    help="Black box.",
)
@click.option(
    "--max-iter", type=int, default=10, help="Maximum iterations to run."
)
@click.option(
    "--replicates", type=int, default=10, help="Replications to perform."
)
@click.option(
    "--bsize",
    type=int,
    default=5,
    help="Batch size for black box evaluation.",
)
@click.option(
    "--tsize",
    type=int,
    default=64,
    help="Training dataset size.",
)
@click.option(
    "--logdir",
    type=click.Path(file_okay=False),
    default="synth_fn",
    help="log and results directory.",
)
@click.option(
    "--device", type=str, default="cpu", help="device to use for solver."
)
@click.option("--seed", type=int, default=42, help="random seed.")
@click.option(
    "--preference",
    type=click.Choice(["gmm", "normal", "empirical"]),
    default="gmm",
    help="preference distribution.",
)
def main(
    bbox,
    max_iter,
    replicates,
    bsize,
    tsize,
    logdir,
    device,
    seed,
    preference,
):
    # Setup logging
    logdir = Path(logdir) / bbox
    logdir.mkdir(exist_ok=True, parents=True)
    logfile = logdir / Path(f"{bbox}.log")
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
        f"bb: {bbox}, max_iter: {max_iter}, replicates: {replicates}, "
        f"bsize: {bsize}, tsize: {tsize}, logdir: {logdir}, device: {device}, "
        f"seed: {seed}."
    )

    # Fix seeds
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if device == "cuda":
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    log.info("Setting up black-box ...")
    bb = obj_mappings[bbox]["f"](**obj_mappings[bbox]["args"])
    D = bb.dim if hasattr(bb, "dim") else 2

    log.info("Generating initial dataset ...")
    X_init = torch.tensor(LatinHypercube(d=D, rng=SEED).random(n=tsize)).float()
    y_init = bb(X_init)
    M = y_init.shape[1]
    ref_point = obj_mappings[bbox]["ref_point"]
    if ref_point is None:
        ref_point = infer_reference_point(y_init)
    log.info(f"Inferred Reference point: {ref_point} ...")

    # Baselines
    methods = ["A-GPS", "qNParEGO"]
    if M <= 3:
        methods += ["qEHVI", "qNEHVI"]
    elif M <= 4:
        methods += ["qNEHVI"]
    results = {m: [] for m in methods}
    log.info(f"Selected baselines: {methods}")

    # Run baselines
    for method in methods:
        log.info(f"METHOD: {method}")
        for r in range(replicates):
            log.info(f"REP: {r} ...")
            X, Y, hvs = X_init.clone(), y_init.clone(), []
            hv_comp = Hypervolume(ref_point=ref_point)
            hv_init = hv_comp.compute(Y[is_non_dominated_strict(Y)])
            hvs.append(1.0)
            if method == "A-GPS":
                agps = AGPS(
                    X_init,
                    y_init,
                    ref_point,
                    bsize,
                    max_iter,
                    preference,
                    device=device,
                )
            for i in range(max_iter):
                log.info(f"  Training {method}: round {i} ... ")
                if method == "A-GPS":
                    Xc, _ = agps.fit(X, Y, i)
                elif method == "qEHVI":
                    moo_model = fit_gp_moo_model(X, Y)
                    Xc = generate_candidates_qehvi(
                        moo_model, X, Y, ref_point, bsize
                    )
                elif method == "qNEHVI":
                    moo_model = fit_gp_moo_model(X, Y)
                    Xc = generate_candidates_qnehvi(
                        moo_model, X, Y, ref_point, bsize
                    )
                else:
                    sca_model = fit_gp_sca_model(X, Y)
                    Xc = generate_candidates_nparego(
                        sca_model, X, Y, ref_point, bsize
                    )
                yc = bb(Xc)
                X = torch.cat([X, Xc], dim=0)
                Y = torch.cat([Y, yc], dim=0)
                hv = hv_comp.compute(Y[is_non_dominated_strict(Y)]) / hv_init
                hvs.append(hv)
                log.info(f"  ... round hv = {hv:.3f}.")
            if method == "A-GPS":
                log.info("Plotting A-GPS preferences")
                plot_agps_prefs(
                    Y,
                    ref_point,
                    agps,
                    bb,
                    bbox,
                    logdir / (bbox + f"_prefs_{r}.png"),
                )
            log.info(f"  ... Done! Final rHV = {hv:.3f}")
            results[method].append(hvs)

    # Plot Results
    log.info("Plotting results ...")
    cycler = plt.cycler(
        color=plt.cm.viridis(np.linspace(0.05, 0.95, 4))
    ) + plt.cycler(linestyle=[next(LINECYCLE) for _ in range(4)])
    plt.rc("axes", prop_cycle=cycler)

    plt.figure(dpi=150, figsize=(8, 5))
    for method, hvs in results.items():
        hvs_array = np.array(hvs)
        mean = hvs_array.mean(axis=0)
        std = hvs_array.std(axis=0)
        lower = mean - std
        upper = mean + std
        log.info(f"{method} :")
        for i, (m, s) in enumerate(zip(mean, std)):
            log.info(f"  {i}: {m:.3f} ({s:.3f})")

        plt.plot(
            mean,
            label=method,
            marker="o" if method == "A-GPS" else "x",
            markersize=10,
            linewidth=3,
        )
        plt.fill_between(range(max_iter + 1), lower, upper, alpha=0.2)

    plt.xlabel("Round")
    plt.ylabel("Relative Hyper-volume")
    plt.title(bbox)
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig(logdir / (bbox + "_hypervol.png"))
    plt.close()


def plot_agps_prefs(Y_agps, ref_point, agps, bb, bb_name, path):
    np.set_printoptions(
        suppress=True, precision=2
    )  # Set the precision of the output to 3
    condsamples = 100
    yplt = Y_agps[is_non_dominated_strict(Y_agps)]
    M = Y_agps.shape[1]

    if M > 2:
        pca = PCA(n_components=2)
        yplt = torch.tensor(pca.fit_transform(yplt.numpy()))
        pref = pca.transform(ref_point.numpy()[np.newaxis, :]).flatten()
    else:
        pref = ref_point

    yplt -= pref
    uf1 = torch.tensor(
        [torch.quantile(yplt[:, 0], q=0.9), torch.quantile(yplt[:, 1], q=0.1)]
    )
    uf2 = torch.tensor([yplt[:, 0].mean(), yplt[:, 1].mean()])
    uf3 = torch.tensor(
        [torch.quantile(yplt[:, 0], q=0.1), torch.quantile(yplt[:, 1], q=0.9)]
    )

    start = (pref[0].item(), pref[1].item())
    cols = plt.cm.inferno([0.1, 0.5, 0.9])
    plt.figure(dpi=150, figsize=(8, 5))
    plt.plot(*start, "ks", label="Reference point")
    for uf, marker, col in zip([uf1, uf2, uf3], ["x", "+", "."], cols):
        if M > 2:
            ufq = torch.tensor(pca.inverse_transform(uf.numpy()))
        else:
            ufq = uf
        norm = torch.norm(ufq, p=2, dim=-1)
        ufq = ufq / torch.maximum(norm, torch.tensor(1e-5))
        ufs = torch.tile(ufq, (condsamples, 1))
        xcand, _ = agps.proposal.cproposal(ufs)
        ycand = bb(xcand)
        if M > 2:
            ycand = torch.tensor(pca.transform(ycand.numpy()))
        plt.plot(
            *ycand.T,
            marker,
            label=f"u = {ufq.numpy()}",
            c=col,
            markersize=10,
            alpha=0.7,
            mew=2,
        )
        dx, dy = uf[0] / 2, uf[1] / 2
        end = (start[0] + dx, start[1] + dy)
        plt.annotate(
            "",  # no text
            xy=end,  # arrow head at 'end'
            xytext=start,  # arrow tail at 'start'
            arrowprops={
                "arrowstyle": "->",
                "color": col,
                "lw": 2,
                "shrinkA": 0,
                "shrinkB": 0,
            },
        )
    plt.legend()
    plt.grid()
    plt.title(bb_name)
    plt.xlabel("$f_1$")
    plt.ylabel("$f_2$")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
