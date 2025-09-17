"""Class probability estimators and learning routines.

Provides compact neural models that output log-probabilities of "success"
(e.g., improvement or Pareto membership), plus training utilities.
"""

from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as fnn

from torch import Tensor
from torch.optim import Optimizer

from vsd.labellers import Labeller
from vsd.utils import (
    PositionalEncoding,
    Transpose,
    Skip,
    CatNorm,
    FuseTwo,
    batch_indices,
    SEPlateauStopping,
)

#
# Class probability estimators -- abstractions and meta-estamators
#


class ClassProbabilityModel(ABC, nn.Module):
    """Binary class-probability estimator for ``p(z=1|x)``.

    Subclasses implement ``_logits(X)`` and inherit a ``forward`` that returns
    ``logsigmoid(_logits(X))`` by default (or the raw logits via
    ``return_logits=True``).
    """

    @abstractmethod
    def _logits(self, X: Tensor) -> Tensor:
        pass

    def forward(self, X: Tensor, return_logits: bool = False) -> Tensor:
        logits = squeeze_1D(self._logits(X))
        if return_logits:
            return logits
        return nn.functional.logsigmoid(logits)

    def set_dropout_p(self, p: float):
        """Reset dropout p -- useful for multiple training steps."""
        for m in self.modules():
            if isinstance(m, torch.nn.Dropout):
                m.p = p


class PreferenceClassProbabilityModel(ABC, nn.Module):
    """Preference-conditioned estimator for ``p(z=1|x, u)``.

    Subclasses implement ``_logits(X, U)``. The default ``forward`` returns
    ``logsigmoid`` of the logits.
    """

    @abstractmethod
    def _logits(self, X: Tensor, U: Tensor) -> Tensor:
        pass

    def forward(
        self, X: Tensor, U: Tensor, return_logits: bool = False
    ) -> Tensor:
        logits = squeeze_1D(self._logits(X, U))
        if return_logits:
            return logits
        return nn.functional.logsigmoid(logits)

    def set_dropout_p(self, p: float):
        """Reset dropout p -- useful for multiple training steps."""
        for m in self.modules():
            if isinstance(m, torch.nn.Dropout):
                m.p = p


class EnsembleProbabilityModel(ClassProbabilityModel):
    """Simple deep ensemble wrapper averaging base-model probabilities."""

    def __init__(
        self,
        base_class: type[ClassProbabilityModel],
        init_kwargs: dict,
        ensemble_size: int = 10,
    ):
        super().__init__()
        self.base_class = base_class
        self.init_kwargs = init_kwargs
        self.ensemble_size = ensemble_size
        self.ensemble = torch.nn.ModuleList(
            [base_class(**init_kwargs) for _ in range(ensemble_size)]
        )

    def _logits(self, X: Tensor) -> Tensor:
        logits = [torch.sigmoid(m._logits(X)) for m in self.ensemble]
        probs = torch.mean(torch.stack(logits), dim=0)
        return torch.logit(probs, eps=1e-6)


#
# Class probability estimators -- Continuous X
#


class ContinuousCPEModel(ClassProbabilityModel):
    """MLP CPE for continuous inputs ``X``.

    Parameters
    ----------
    x_dim : int
        Input dimensionality.
    latent_dim : int
        Hidden width of residual blocks.
    dropoutp : float, default=0.0
        Dropout probability.
    hidden_layers : int, default=2
        Number of residual hidden blocks.
    """

    def __init__(
        self,
        x_dim: int,
        latent_dim: int,
        dropoutp: float = 0,
        hidden_layers: int = 2,
    ):
        super().__init__()
        hidden = [
            Skip(
                nn.Sequential(
                    nn.Linear(in_features=latent_dim, out_features=latent_dim),
                    nn.LayerNorm(normalized_shape=latent_dim),
                    nn.LeakyReLU(),
                    nn.Dropout(p=dropoutp),
                )
            )
            for _ in range(hidden_layers)
        ]
        self.nn = torch.nn.Sequential(
            nn.Linear(in_features=x_dim, out_features=latent_dim),
            nn.LayerNorm(normalized_shape=latent_dim),
            nn.LeakyReLU(),
            nn.Dropout(p=dropoutp),
            *hidden,
            nn.Linear(in_features=latent_dim, out_features=1),
        )

    def _logits(self, X: Tensor) -> Tensor:
        return self.nn(X)


class PreferenceContinuousCPE(PreferenceClassProbabilityModel):
    """Continuous CPE conditioned on a preference vector ``U``."""

    def __init__(
        self,
        x_dim: int,
        u_dims: int,
        latent_dim: int,
        dropoutp: float = 0,
        hidden_layers: int = 2,
    ):
        super().__init__()
        self.nn = ContinuousCPEModel(
            x_dim=x_dim + u_dims,
            latent_dim=latent_dim,
            dropoutp=dropoutp,
            hidden_layers=hidden_layers,
        )

    def _logits(self, X, U):
        return self.nn(torch.hstack((X, U)))


#
# Class probability estimators -- Sequence X
#


class SequenceProbabilityModel(ClassProbabilityModel):
    """Base for sequence CPEs with fixed length and alphabet size."""

    def __init__(self, seq_len, alpha_len) -> None:
        super().__init__()
        self.seq_len = seq_len
        self.alpha_len = alpha_len


class NNClassProbability(SequenceProbabilityModel):
    """Simple embedding + MLP CPE for sequences."""

    def __init__(
        self,
        seq_len,
        alpha_len,
        embedding_dim: Optional[int] = None,
        dropoutp: float = 0,
        hlsize: int = 64,
    ) -> None:
        super().__init__(seq_len, alpha_len)
        if embedding_dim is None:
            embedding_dim = max(2, self.alpha_len // 2)
        self.nn = torch.nn.Sequential(
            nn.Embedding(num_embeddings=alpha_len, embedding_dim=embedding_dim),
            nn.Dropout(p=dropoutp),
            nn.Flatten(),
            nn.Linear(in_features=embedding_dim * seq_len, out_features=hlsize),
            nn.LeakyReLU(),
            nn.Dropout(p=dropoutp / 2),
            nn.LayerNorm(normalized_shape=hlsize),
            nn.Linear(in_features=hlsize, out_features=1),
        )

    def _logits(self, X: Tensor) -> Tensor:
        return self.nn(X)


def _make_sequence_cnn(
    alpha_len: int,
    embedding_dim: Optional[int],
    pos_encoding: bool,
    out_features: int,
    dropoutp: float,
    cfilter_size: int,
    ckernel: int,
    xkernel: int,
    xstride: int,
) -> torch.nn.Module:
    if ckernel % 2 == 0:
        raise ValueError("ckernel can only be odd.")

    if embedding_dim is None:
        embedding_dim = max(8, alpha_len // 2)
    emb = [nn.Embedding(num_embeddings=alpha_len, embedding_dim=embedding_dim)]
    if pos_encoding:
        emb.append(PositionalEncoding(emb_size=embedding_dim))

    # Automatically figure out group normalisation size
    groups = min(8, cfilter_size)
    while cfilter_size % groups != 0 and groups > 1:
        groups -= 1
    pad = ckernel // 2

    f = torch.nn.Sequential(
        *emb,
        nn.Dropout(p=dropoutp),
        Transpose(),
        nn.Conv1d(
            in_channels=embedding_dim,
            out_channels=cfilter_size,
            kernel_size=ckernel,
            padding=pad,
        ),
        nn.GroupNorm(num_groups=groups, num_channels=cfilter_size),
        nn.LeakyReLU(),
        nn.Dropout1d(p=dropoutp / 2),
        FuseTwo(
            nn.AvgPool1d(kernel_size=xkernel, stride=xstride),
            nn.MaxPool1d(kernel_size=xkernel, stride=xstride),
        ),
        Skip(
            nn.Sequential(
                nn.Conv1d(
                    in_channels=cfilter_size,
                    out_channels=cfilter_size,
                    kernel_size=ckernel,
                    padding=pad,
                ),
                nn.GroupNorm(num_groups=groups, num_channels=cfilter_size),
                nn.LeakyReLU(),
            )
        ),
        FuseTwo(
            nn.AdaptiveAvgPool1d(output_size=1),
            nn.AdaptiveMaxPool1d(output_size=1),
        ),
        nn.Flatten(),
        nn.Linear(in_features=cfilter_size, out_features=out_features),
    )
    return f


class CNNClassProbability(SequenceProbabilityModel):
    """CNN CPE for sequences.

    Receptive field = (ckernel - 1) * (1 + xstride) + xkernel
    The default is 15.
    """

    def __init__(
        self,
        seq_len,
        alpha_len,
        embedding_dim: Optional[int] = None,
        ckernel: int = 5,
        xkernel: int = 3,
        xstride: int = 2,
        cfilter_size: int = 64,
        linear_size: int = 128,
        dropoutp: float = 0.2,
        pos_encoding: bool = True,
    ) -> None:
        super().__init__(seq_len, alpha_len)
        self.cnn = _make_sequence_cnn(
            alpha_len=alpha_len,
            embedding_dim=embedding_dim,
            pos_encoding=pos_encoding,
            out_features=linear_size,
            dropoutp=dropoutp,
            cfilter_size=cfilter_size,
            ckernel=ckernel,
            xkernel=xkernel,
            xstride=xstride,
        )
        self.mlp = torch.nn.Sequential(
            nn.LayerNorm(normalized_shape=linear_size),
            nn.LeakyReLU(),
            nn.Linear(in_features=linear_size, out_features=1),
        )

    def _logits(self, X: Tensor) -> Tensor:
        h = self.cnn(X)
        return self.mlp(h)


class PreferenceCNNClassProbability(PreferenceClassProbabilityModel):
    """
    Preference-conditioned CNN class probability model.

    Receptive field = (ckernel - 1) * (1 + xstride) + xkernel
    The default is 15.
    """

    def __init__(
        self,
        seq_len: int,
        alpha_len: int,
        u_dims: int,
        embedding_dim: Optional[int] = None,
        ckernel: int = 5,
        xkernel: int = 3,
        xstride: int = 2,
        cfilter_size: int = 64,
        linear_size: int = 128,
        dropoutp: float = 0.2,
        pos_encoding: bool = True,
    ) -> None:
        super().__init__()
        self.seq_len = seq_len
        self.alpha_len = alpha_len
        self.u_dims = u_dims
        self.cnn = _make_sequence_cnn(
            alpha_len=alpha_len,
            embedding_dim=embedding_dim,
            pos_encoding=pos_encoding,
            out_features=linear_size,
            dropoutp=dropoutp,
            cfilter_size=cfilter_size,
            ckernel=ckernel,
            xkernel=xkernel,
            xstride=xstride,
        )
        self.mlp = nn.Sequential(
            CatNorm((linear_size, u_dims)),
            nn.Linear(linear_size + u_dims, linear_size),
            nn.LeakyReLU(),
            nn.LayerNorm(normalized_shape=linear_size),
            Skip(
                nn.Sequential(
                    nn.Linear(linear_size, linear_size),
                    nn.LeakyReLU(),
                )
            ),
            nn.Linear(linear_size, 1),
        )

    def _logits(self, X: Tensor, U: Tensor) -> Tensor:
        return self.mlp((self.cnn(X), U))


#
# Contrastive data creation
#


def make_contrastive_alignment_data_random(
    X: Tensor, U: Tensor, negative_replicates: int = 9
) -> Tuple[Tensor, Tensor, Tensor]:
    """Construct alignment data using **random derangement** negatives.

    Returns (Xa, Ua, za) where positives are (x_i, u_i) and negatives are
    (x_i, u_{p(i)}) with p a derangement. Repeats `negative_replicates` times.

    Shapes
    ------
    Xa : ((1 + negative_replicates) * N, Dx)
    Ua : ((1 + negative_replicates) * N, Du)
    za : ((1 + negative_replicates) * N, 1)  # column vector of 1s then 0s
    """
    device = X.device
    N = U.size(0)

    # positives
    Xa_list = [X]
    Ua_list = [U]
    labels = [torch.ones(N, device=device)]

    # negatives via derangements
    for _ in range(max(0, negative_replicates)):
        p = _derangement(N, device=device)
        Xa_list.append(X)
        Ua_list.append(U[p])
        labels.append(torch.zeros(N, device=device))

    Xa = torch.vstack(Xa_list)
    Ua = torch.vstack(Ua_list)
    za = torch.cat(labels, dim=0).float()
    return Xa, Ua, za


def make_contrastive_alignment_data_knn(
    X: Tensor,
    U: Tensor,
    negative_replicates: int = 9,
    knn_k: int = 10,
) -> Tuple[Tensor, Tensor, Tensor]:
    """Construct alignment data using **semi-hard KNN** negatives in U-space.

    For each i, sample a negative u_j from the top-`knn_k` most similar
    (cosine) preferences to u_i, excluding i. Repeat `negative_replicates` times.

    Shapes are the same as `make_contrastive_alignment_data_random`.
    """
    device = X.device
    N = U.size(0)

    # Precompute KNN indices in U (cosine similarity)
    Un = fnn.normalize(U, dim=-1)
    S = Un @ Un.T  # (N,N)
    S.fill_diagonal_(-1.0)  # Remove self similarity
    k = max(1, min(knn_k, N - 1))
    knn_idx = torch.topk(S, k=k, dim=-1).indices  # (N,k)

    Xa_list = [X]
    Ua_list = [U]
    labels = [torch.ones(N, device=device)]

    for _ in range(max(0, int(negative_replicates))):
        rind = torch.randint(0, k, (N,), device=device)
        p = knn_idx[torch.arange(N, device=device), rind]  # (N,)
        Xa_list.append(X)
        Ua_list.append(U[p])
        labels.append(torch.zeros(N, device=device))

    Xa = torch.vstack(Xa_list)
    Ua = torch.vstack(Ua_list)
    za = torch.cat(labels, dim=0).float()
    return Xa, Ua, za


def make_contrastive_alignment_data(
    X: Tensor,
    U: Tensor,
    negative_replicates_random: int = 7,
    negative_replicates_knn: int = 2,
    knn_k: int = 10,
) -> Tuple[Tensor, Tensor, Tensor]:
    """Construct alignment data mixing **random** and **KNN** negatives.

    The final dataset concatenates the outputs of
    `make_contrastive_alignment_data_random` and
    `make_contrastive_alignment_data_knn` (sharing the same positives).

    Parameters
    ----------
    X, U : torch.Tensor
        Base pairs (x_i, u_i).
    negative_replicates_random : int
        Number of random-derangement negative replicates per positive.
    negative_replicates_knn : int
        Number of KNN semi-hard negative replicates per positive.
    knn_k : int
        Candidate pool size for KNN selection in U-space.

    Returns
    -------
    Xa, Ua, za : torch.Tensor
        Concatenated alignment dataset with labels stacked as a column vector.
    """
    Xa_r, Ua_r, za_r = make_contrastive_alignment_data_random(
        X, U, negative_replicates=negative_replicates_random
    )
    Xa_k, Ua_k, za_k = make_contrastive_alignment_data_knn(
        X, U, negative_replicates=negative_replicates_knn, knn_k=knn_k
    )

    Xa = torch.vstack(
        [Xa_r, Xa_k[len(X) :]]
    )  # avoid duplicating positives twice
    Ua = torch.vstack([Ua_r, Ua_k[len(U) :]])
    za = torch.concatenate([za_r, za_k[len(U) :]])
    return Xa, Ua, za


#
# Learning routines
#


def fit_cpe(
    model: ClassProbabilityModel,
    X: Tensor,
    y: Tensor,
    labeller: float | Callable[[Tensor], Tensor] | Labeller,
    U: Optional[Tensor] = None,
    X_val: Optional[Tensor] = None,
    y_val: Optional[Tensor] = None,
    U_val: Optional[Tensor] = None,
    batch_size: int = 32,
    optimizer: Optimizer = torch.optim.AdamW,
    optimizer_options: Optional[Dict[str, Any]] = None,
    stop_options: Optional[Dict[str, Any]] = None,
    device: str = "cpu",
    callback: Optional[Callable[[int, Tensor], None]] = None,
    seed: Optional[int] = None,
    stop_using_val_loss: bool = False,
):
    """
    Fit a Class Probability Estimator using labels derived from a threshold.

    Parameters
    ----------
    model : ClassProbabilityModel
        A neural network model or similar implementing the CPE interface.
    X : torch.Tensor
        Training input tensor of shape (N, D).
    y : torch.Tensor
        Target values for training inputs, of shape (N,).
    labeller : float, callable or Labeller
        Threshold used to define positive class labels. If a `Threshold`
        instance is provided, it is used directly to binarize `y`.
    X_val : torch.Tensor, optional
        Optional validation input tensor of shape (N_val, D).
    y_val : torch.Tensor, optional
        Optional validation targets of shape (N_val,).
    batch_size : int, default=32
        Batch size used for stochastic optimization.
    optimizer : torch.optim.Optimizer, default=torch.optim.AdamW
        Optimizer class to use for training.
    optimizer_options : dict, optional
        Dictionary of keyword arguments passed to the optimizer.
    stop_options : dict, optional
        Dictionary of keyword arguments passed to `ExpMAStoppingCriterion` for
        early stopping.
    device : str, default="cpu"
        Device to use for model training (e.g., "cpu" or "cuda").
    callback : callable, optional
        Optional function with signature `callback(iteration, train_loss,
        val_loss)` called each iteration.
    seed : int, optional
        Random seed for reproducibility of mini-batch selection.
    stop_using_val_loss : bool, default=False
        Whether to use validation loss as the criterion for early stopping.
        Requires `X_val` and `y_val`.

    Raises
    ------
    ValueError
        If `stop_using_val_loss` is True but `X_val` or `y_val` is not provided.

    Returns
    -------
    None
        The model is updated in-place and left in evaluation mode.
    """
    return fit_cpe_labels(
        model=model,
        X=X,
        z=_get_labels(y, labeller),
        U=U,
        X_val=X_val,
        z_val=None if y_val is None else _get_labels(y_val, labeller),
        U_val=U_val,
        batch_size=batch_size,
        optimizer=optimizer,
        optimizer_options=optimizer_options,
        stop_options=stop_options,
        device=device,
        callback=callback,
        seed=seed,
        stop_using_val_loss=stop_using_val_loss,
    )


def fit_cpe_labels(
    model: ClassProbabilityModel | PreferenceClassProbabilityModel,
    X: Tensor,
    z: Tensor,
    U: Optional[Tensor] = None,
    X_val: Optional[Tensor] = None,
    z_val: Optional[Tensor] = None,
    U_val: Optional[Tensor] = None,
    batch_size: int = 32,
    optimizer: Optimizer = torch.optim.AdamW,
    optimizer_options: Optional[Dict[str, Any]] = None,
    stop_options: Optional[Dict[str, Any]] = None,
    device: str = "cpu",
    callback: Optional[Callable[[int, Tensor], None]] = None,
    seed: Optional[int] = None,
    stop_using_val_loss: bool = False,
):
    """
    Fit a Preference Class Probability Estimator using labels provided

    Parameters
    ----------
    model : ClassProbabilityModel | PreferenceClassProbabilityModel
        A neural network model or similar implementing the Preference CPE
        interface.
    X : torch.Tensor
        Training input tensor of shape (N, D).
    z : torch.Tensor
        Label values for training inputs, of shape (N,).
    U : torch.Tensor, optional
        Training input preference tensor of shape (N, M).
    X_val : torch.Tensor, optional
        Optional validation input tensor of shape (N_val, D).
    z_val : torch.Tensor, optional
        Optional validation targets of shape (N_val,).
    U_val : torch.Tensor, optional
        Optional preference validation input tensor of shape (N_val, M).
    batch_size : int, default=32
        Batch size used for stochastic optimization.
    optimizer : torch.optim.Optimizer, default=torch.optim.AdamW
        Optimizer class to use for training.
    optimizer_options : dict, optional
        Dictionary of keyword arguments passed to the optimizer.
    stop_options : dict, optional
        Dictionary of keyword arguments passed to `ExpMAStoppingCriterion` for
        early stopping.
    device : str, default="cpu"
        Device to use for model training (e.g., "cpu" or "cuda").
    callback : callable, optional
        Optional function with signature `callback(iteration, train_loss,
        val_loss)` called each iteration.
    seed : int, optional
        Random seed for reproducibility of mini-batch selection.
    stop_using_val_loss : bool, default=False
        Whether to use validation loss as the criterion for early stopping.
        Requires `X_val` and `z_val`.

    Raises
    ------
    ValueError
        If `stop_using_val_loss` is True but `X_val` or `z_val` is not provided.

    Returns
    -------
    None
        The model is updated in-place and left in evaluation mode.
    """
    if (X_val is None) and stop_using_val_loss:
        raise ValueError("Need to specify X_val to stop_using_xval_loss")

    model.to(device)
    optimizer_options = {} if optimizer_options is None else optimizer_options
    stop_options = {} if stop_options is None else stop_options
    optim = optimizer(model.parameters(), **optimizer_options)
    lossfn = torch.nn.BCEWithLogitsLoss()
    stopping_criterion = SEPlateauStopping(**stop_options)  # type: ignore

    if X_val is None:
        vloss = torch.zeros([])
    else:
        if z_val is None:
            raise ValueError("If X_val is passed, z_val is required too.")
        X_val = X_val.to(device)
        z_val = z_val.to(device)
        if U_val is not None:
            U_val = U_val.to(device)

    def model_call(X, U):
        if U is None:
            return model(X, return_logits=True)
        else:
            return model(X, U, return_logits=True)

    Ub = None
    model.train()
    for i, bi in enumerate(batch_indices(len(z), batch_size, seed)):
        Xb = torch.atleast_2d(X[bi].to(device))
        zb = z[bi].to(device)
        if U is not None:
            Ub = torch.atleast_2d(U[bi].to(device))
        loss = lossfn(model_call(Xb, Ub), zb)
        loss.backward()
        optim.step()
        optim.zero_grad()

        sloss = loss.detach()
        if X_val is not None:
            model.eval()
            with torch.no_grad():
                vloss = lossfn(model_call(X_val, U_val), z_val).detach()
            if stop_using_val_loss:
                sloss = vloss
            model.train()

        if callback is not None:
            callback(i, loss, vloss)
        if stopping_criterion.evaluate(fvals=sloss):
            break
    model.eval()


def _get_labels(
    y: Tensor,
    labeller: float | Callable[[Tensor], Tensor] | Labeller,
) -> Tensor:
    if isinstance(labeller, float):
        return (y >= labeller).float()
    return labeller(y).float()


def _derangement(n: int, device=None, max_attempts=20):
    """Sample a permutation with no fixed points (derangement)."""
    ord = torch.arange(n, device=device)
    for _ in range(max_attempts):
        p = torch.randperm(n, device=device)
        if (p != ord).all():
            break
    return p


def squeeze_1D(x: Tensor) -> Tensor:
    """Squeeze to at least 1D (preserves scalar as shape ``(1,)``)."""
    return torch.atleast_1d(x.squeeze())
