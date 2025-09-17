"""Conditional variational distributions.

Implements conditional search distributions ``q(X|U)`` and utilities for
preference-aware training and inference.
"""

from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, Optional, Tuple

import torch
import torch.distributions as td
import torch.nn as nn
import torch.nn.functional as fnn
from torch import Tensor
from torch.optim import Optimizer

from vsd.preferences import PreferenceDistribution
from vsd.proposals import (
    SearchDistribution,
    _TestMixin,
    _LSTMMixin,
    _DTransformerMixin,
    _TransformerMutationMixin,
    LSTMProposal,
    DTransformerProposal,
    TransformerMLMProposal,
    clip_gradients,
)
from vsd.utils import Skip, batch_indices, batch_indices_val, SEPlateauStopping

#
# Abstract Conditional Search Distributions
#


class ConditionalSearchDistribution(torch.nn.Module, ABC):
    """Abstract base for conditional search distributions ``q(X|U)``.

    Subclasses define ``sample(U)`` and ``log_prob(X, U)``. The base ``forward``
    draws samples (without grad) and returns ``(X, log q(X|U))``.
    """

    prior_same_class = False  # Indicate this class can be used as its own prior

    def __init__(
        self,
        u_dims: int,
        samples: int = 100,
        clip_gradients: Optional[float] = None,
    ) -> None:
        torch.nn.Module.__init__(self)
        self.u_dims = u_dims
        self.samples = samples
        self.clip_gradients = clip_gradients

    def forward(self, U: Tensor) -> Tuple[Tensor, Tensor]:
        with torch.no_grad():
            Xs = self.sample(U)
        logqX = self.log_prob(Xs, U)
        return Xs, logqX

    @abstractmethod
    def sample(self, U: Tensor) -> Tensor:
        pass

    @abstractmethod
    def log_prob(self, X: Tensor, U: Tensor) -> Tensor:
        pass

    @abstractmethod
    def get_compatible_prior(
        self,
    ) -> SearchDistribution | torch.distributions.Distribution: ...

    def _save_constructor_args(self, local_vars: Dict[str, Any]):
        self._constructor_args = {
            k: v
            for k, v in local_vars.items()
            if k not in ("self", "__class__")
        }

    def get_constructor_args(self) -> Dict[str, Any]:
        if not hasattr(self, "_constructor_args"):
            raise ValueError("Consturctor arguments not saved!")
        return self._constructor_args.copy()

    def set_dropout_p(self, p: float):
        """Reset dropout p -- useful for multiple training steps."""
        for m in self.modules():
            if isinstance(m, torch.nn.Dropout):
                m.p = p


#
# Interface for combining ConditionalSearchDistributions and
#   PreferenceDistributions. This is used for learning.
#


class PreferenceSearchDistribution(SearchDistribution):
    """Joint search distribution with stochastic preferences ``q(X, U)``."""

    def __init__(
        self,
        cproposal: ConditionalSearchDistribution,
        preference: PreferenceDistribution,
        samples: int = 100,
        clip_gradients: Optional[float] = None,
    ) -> None:
        super().__init__(samples=samples, clip_gradients=clip_gradients)
        self.cproposal = cproposal
        self.preference = preference

    def forward(
        self, samples: Optional[int] = None
    ) -> Tuple[Tensor, Tensor, Tensor]:
        samples = self.samples if samples is None else samples
        Xs, Us = self.sample(torch.Size([samples]))
        logqX = self.log_prob(Xs, Us)
        return Xs, Us, logqX

    @torch.no_grad()
    def sample(
        self, sample_shape: torch.Size = torch.Size([1])
    ) -> Tuple[Tensor, Tensor]:
        device = next(self.cproposal.parameters()).device  # Needed, but why?
        Us = self.preference.sample(sample_shape).to(device)
        Xs = self.cproposal.sample(Us)
        return Xs, Us

    def log_prob(self, X: Tensor, U: Tensor) -> Tensor:
        """Only return conditional log q(X|U)."""
        return self.cproposal.log_prob(X, U)

    def train(self, mode=True):
        r = super().train(mode)
        self.preference.eval()
        self.preference.requires_grad_(False)
        return r

    def eval(self):
        r = super().eval()
        self.preference.eval()
        self.preference.requires_grad_(True)
        return r


#
# Continuous Conditional Search Distributions
#


class ConditionalContinuousSearchDistribution(
    ConditionalSearchDistribution, _TestMixin
):
    """Continuous conditional proposals with optional x-transform hooks."""

    def __init__(
        self,
        u_dims: int,
        x_transform: Optional[Callable[[Tensor], Tensor]] = None,
        x_invtransform: Optional[Callable[[Tensor], Tensor]] = None,
        samples: int = 100,
        clip_gradients: Optional[float] = None,
    ) -> None:
        ConditionalSearchDistribution.__init__(
            self, u_dims=u_dims, samples=samples, clip_gradients=clip_gradients
        )
        _TestMixin.__init__(self)
        self.x_transform = x_transform
        self.x_invtransform = x_invtransform

    @torch.no_grad()
    def sample(self, U: Tensor) -> Tensor:
        q = self._construct_q(U)
        Xs = q.sample()

        # Testing
        if self._test_sample_consistency:
            self._last_sample_log_prob = q.log_prob(Xs)

        if self.x_invtransform is not None:
            Xs = self.x_invtransform(Xs)

        return Xs

    def log_prob(self, X: Tensor, U: Tensor) -> Tensor:
        if self.x_transform is not None:
            X = self.x_transform(X)
        q = self._construct_q(U)
        return q.log_prob(X)

    def set_dropout_p(self, p: float):
        """Reset dropout p -- useful for multiple training steps."""
        for m in self.modules():
            if isinstance(m, torch.nn.Dropout):
                m.p = p

    @abstractmethod
    def _construct_q(self, U: Tensor) -> td.Distribution: ...


class _MLPConditioner(nn.Module):
    """Shared MLP conditioner backbone mapping ``U → params``."""

    def __init__(
        self,
        u_dims: int,
        latent_dim: int,
        out_dim: int,
        *,
        bias: bool = False,
        hidden_layers: int = 1,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        hidden = [
            Skip(
                nn.Sequential(
                    nn.Linear(
                        in_features=latent_dim,
                        out_features=latent_dim,
                        bias=bias,
                    ),
                    nn.LayerNorm(normalized_shape=latent_dim),
                    nn.SiLU(),
                )
            )
            for _ in range(hidden_layers)
        ]
        self.net = nn.Sequential(
            nn.Linear(in_features=u_dims, out_features=latent_dim, bias=bias),
            nn.LayerNorm(normalized_shape=latent_dim),
            nn.SiLU(),
            nn.Dropout(p=dropout),
            *hidden,
            nn.Linear(in_features=latent_dim, out_features=out_dim, bias=bias),
        )

    def forward(self, U: Tensor) -> Tensor:
        return self.net(U)


class ConditionalGaussianProposal(ConditionalContinuousSearchDistribution):
    """Single conditional Gaussian search distribution.

    Parameters
    ----------
    x_dims : int
        Dimensionality of X (event dim D).
    u_dims : int
        Dimensionality of conditioning vector U.
    latent_dim : int
        Hidden size for the conditioner MLP.
    low_rank_dim : int, default=0
        Rank of the low-rank covariance factor; 0 recovers a diagonal Gaussian.
    min_scale : float, default=1e-6
        Lower bound for standard deviations for numerical stability.
    """

    def __init__(
        self,
        x_dims: int,
        u_dims: int,
        latent_dim: int,
        *,
        low_rank_dim: int = 0,
        bias: bool = True,
        hidden_layers: int = 1,
        dropout: float = 0.0,
        min_scale: float = 1e-6,
        x_transform: Optional[Callable[[Tensor], Tensor]] = None,
        x_invtransform: Optional[Callable[[Tensor], Tensor]] = None,
        samples: int = 100,
        clip_gradients: Optional[float] = None,
    ) -> None:
        ConditionalContinuousSearchDistribution.__init__(
            self,
            u_dims=u_dims,
            x_transform=x_transform,
            x_invtransform=x_invtransform,
            samples=samples,
            clip_gradients=clip_gradients,
        )
        self.xdims = x_dims
        self.min_scale = min_scale
        if low_rank_dim < 0:
            raise ValueError("low_rank_dim must be non-negative")
        self.low_rank_dim = low_rank_dim
        out_dim = x_dims * (2 + low_rank_dim)
        self.cond_nn = _MLPConditioner(
            u_dims=u_dims,
            latent_dim=latent_dim,
            out_dim=out_dim,
            bias=bias,
            hidden_layers=hidden_layers,
            dropout=dropout,
        )

    def _construct_q(self, U: Tensor) -> td.Distribution:
        nnout = self.cond_nn(U)
        mu = nnout[:, : self.xdims]
        diag_start = self.xdims
        diag_end = diag_start + self.xdims
        scale = fnn.softplus(nnout[:, diag_start:diag_end]) + self.min_scale
        if self.low_rank_dim > 0:
            cov_diag = scale.square()
            cov_factor = nnout[:, diag_end:].view(
                nnout.size(0), self.xdims, self.low_rank_dim
            )
            return td.LowRankMultivariateNormal(
                loc=mu, cov_factor=cov_factor, cov_diag=cov_diag
            )
        norm = td.Independent(td.Normal(loc=mu, scale=scale), 1)
        return norm

    def get_compatible_prior(
        self, loc: Optional[Tensor] = None, scale: Optional[Tensor] = None
    ) -> td.Independent:
        loc = torch.zeros(self.xdims) if loc is None else loc
        scale = torch.ones(self.xdims) if scale is None else scale
        return td.Independent(td.Normal(loc=loc, scale=scale), 1)


class ConditionalGMMProposal(ConditionalContinuousSearchDistribution):
    """Conditional Mixture of Gaussians (diagonal) search distribution.

    q(X|U) = sum_{k=1..K} pi_k(U) * N(X; mu_k(U), diag(sigma_k^2(U)))

    Parameters
    ----------
    x_dims : int
        Dimensionality of X (event dim D).
    u_dims : int
        Dimensionality of conditioning vector U.
    latent_dim : int
        Hidden size for the conditioner MLP.
    n_components : int, default=5
        Number of mixture components K.
    min_scale : float, default=1e-6
        Lower bound for standard deviations for numerical stability.
    """

    def __init__(
        self,
        x_dims: int,
        u_dims: int,
        latent_dim: int,
        *,
        n_components: int = 5,
        bias: bool = True,
        hidden_layers: int = 1,
        dropout: float = 0.0,
        x_transform: Optional[Callable[[Tensor], Tensor]] = None,
        x_invtransform: Optional[Callable[[Tensor], Tensor]] = None,
        samples: int = 100,
        clip_gradients: Optional[float] = None,
        min_scale: float = 1e-6,
    ) -> None:
        super().__init__(
            u_dims=u_dims,
            x_transform=x_transform,
            x_invtransform=x_invtransform,
            samples=samples,
            clip_gradients=clip_gradients,
        )
        self.xdims = x_dims
        self.K = n_components
        self.min_scale = min_scale

        # Conditioner outputs: [logits_K, means_{K*D}, scales_{K*D}]
        out_dim = self.K * (2 * x_dims) + self.K
        self.cond_nn = _MLPConditioner(
            u_dims=u_dims,
            latent_dim=latent_dim,
            out_dim=out_dim,
            bias=bias,
            hidden_layers=hidden_layers,
            dropout=dropout,
        )

    def _construct_q(self, U: Tensor) -> td.Distribution:
        N, D, K = U.shape[0], self.xdims, self.K
        nnout = self.cond_nn(U)  # (N, K*(2D) + K)

        # Split parameters
        logits = nnout[:, :K]  # (N, K)
        mu_flat = nnout[:, K : K + K * D]  # (N, K*D)
        std_flat = nnout[:, K + K * D :]  # (N, K*D)
        mu = mu_flat.view(N, K, D)
        std = fnn.softplus(std_flat).view(N, K, D) + self.min_scale
        comp = td.Independent(td.Normal(loc=mu, scale=std), 1)  # event dim D
        mix = td.MixtureSameFamily(td.Categorical(logits=logits), comp)
        return mix

    def get_compatible_prior(
        self, loc: Optional[Tensor] = None, scale: Optional[Tensor] = None
    ) -> td.Independent:
        loc = torch.zeros(self.xdims) if loc is None else loc
        scale = torch.ones(self.xdims) if scale is None else scale
        return td.Independent(td.Normal(loc=loc, scale=scale), 1)


# Sequence Conditional Search Distributions


class SequenceCondSearchDistribution(ConditionalSearchDistribution):
    """Abstract base for conditional autoregressive sequence proposals."""

    def __init__(
        self,
        d_features: int,
        k_categories: int,
        u_dims: int,
        samples: int = 100,
        clip_gradients: float | None = None,
    ) -> None:
        ConditionalSearchDistribution.__init__(
            self, u_dims, samples, clip_gradients
        )
        self.d = d_features
        self.k = k_categories


class FiLM(torch.nn.Module):
    """Feature-wise Linear Modulation (FiLM) for conditioning embeddings.

    Given a context ``U``, produces per-feature ``(gamma, beta)`` to modulate
    embeddings as ``e' = e * (1 + gamma) + beta``.
    """

    def __init__(self, u_dims: int, embedding_dim: int):
        super().__init__()
        hidden_dim = max(u_dims, 2 * embedding_dim)

        # Careful with output weight initialisation, reduce initial impact
        gam_out = nn.Linear(hidden_dim, embedding_dim)
        bet_out = nn.Linear(hidden_dim, embedding_dim)
        with torch.no_grad():
            gam_out.weight.data.copy_(torch.zeros((embedding_dim, hidden_dim)))
            gam_out.bias.data.copy_(torch.zeros(embedding_dim))
            bet_out.weight.data.copy_(torch.zeros((embedding_dim, hidden_dim)))
            bet_out.bias.data.copy_(torch.zeros(embedding_dim))

        self.gamma = nn.Sequential(
            nn.Linear(u_dims, hidden_dim),
            nn.LayerNorm(normalized_shape=hidden_dim),
            nn.SiLU(),
            gam_out,
        )
        self.beta = nn.Sequential(
            nn.Linear(u_dims, hidden_dim),
            nn.LayerNorm(normalized_shape=hidden_dim),
            nn.SiLU(),
            bet_out,
        )

    def forward(self, U: Tensor) -> Tuple[Tensor, Tensor]:
        gam = self.gamma(U).unsqueeze(1)
        bet = self.beta(U).unsqueeze(1)
        return gam, bet


class CondLSTMProposal(SequenceCondSearchDistribution, _LSTMMixin):
    """Conditional LSTM Proposal for sequences, q(X|U).

    Uses FiLM for conditioning:
        Perez, E., Strub, F., de Vries, H., Dumoulin, V., & Courville, A.
        "FiLM: Visual Reasoning with a General Conditioning Layer."
        AAAI Conference on Artificial Intelligence (AAAI), 2018.
    """

    def __init__(
        self,
        d_features: int,
        k_categories: int,
        u_dims: int,
        embedding_dim: Optional[int] = None,
        hidden_size: Optional[int] = None,
        num_layers: int = 1,
        dropout: float = 0.0,
        samples: int = 100,
        clip_gradients: Optional[float] = None,
    ):
        self._save_constructor_args(locals())
        SequenceCondSearchDistribution.__init__(
            self,
            d_features=d_features,
            k_categories=k_categories,
            u_dims=u_dims,
            samples=samples,
            clip_gradients=clip_gradients,
        )
        _LSTMMixin.__init__(
            self,
            d_features=d_features,
            k_categories=k_categories,
            embedding_dim=embedding_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
        )
        self.u0 = nn.Linear(u_dims, self.e)
        self.film = FiLM(u_dims=u_dims, embedding_dim=self.e)

    def _change_embeddings(self, e: Tensor) -> Tensor:
        e = e * (1 + self.gam) + self.bet
        return e

    def _get_prefix(self, samples: int) -> Tensor:
        return self.e0

    def sample(self, U: Tensor) -> Tensor:
        self.gam, self.bet = self.film(U)
        self.e0 = self.u0(U)
        sample_shape = torch.Size([len(U)])
        Xs = self._sample(sample_shape=sample_shape)
        return Xs

    def log_prob(self, X: Tensor, U: Tensor) -> Tensor:
        self.gam, self.bet = self.film(U)
        self.e0 = self.u0(U)
        return self._log_prob(X)

    def get_compatible_prior(self) -> LSTMProposal:
        kwargs = self.get_constructor_args()
        # Pop irrelevant items
        kwargs.pop("u_dims")
        return LSTMProposal(**kwargs)


class CondDTransformerProposal(
    SequenceCondSearchDistribution, _DTransformerMixin
):
    """Decoder-only conditional transformer search distribution, q(X|U)."""

    def __init__(
        self,
        d_features: int,
        k_categories: int,
        u_dims: int,
        embedding_dim: Optional[int] = None,
        nhead: int = 2,
        dim_feedforward: int = 128,
        dropout: float = 0.0,
        num_layers: int = 1,
        samples: int = 100,
        clip_gradients: Optional[float] = None,
    ) -> None:
        self._save_constructor_args(locals())
        SequenceCondSearchDistribution.__init__(
            self,
            d_features=d_features,
            k_categories=k_categories,
            u_dims=u_dims,
            samples=samples,
            clip_gradients=clip_gradients,
        )
        _DTransformerMixin.__init__(
            self,
            d_features=d_features,
            k_categories=k_categories,
            embedding_dim=embedding_dim,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            num_layers=num_layers,
        )
        self.u0 = torch.nn.Linear(u_dims, self.e)
        self.film = FiLM(u_dims=u_dims, embedding_dim=self.e)

    def _change_embeddings(self, e: Tensor) -> Tensor:
        g, b = self.gam, self.bet
        if e.ndim == 2:
            g, b = g.squeeze(), b.squeeze()
        e = e * (1 + g) + b
        return e

    def _get_prefix(self, samples: int) -> Tensor:
        return self.e0

    def sample(self, U: Tensor) -> Tensor:
        self.gam, self.bet = self.film(U)
        self.e0 = self.u0(U)
        sample_shape = torch.Size([len(U)])
        Xs = self._sample(sample_shape=sample_shape)
        return Xs

    def log_prob(self, X: Tensor, U: Tensor) -> Tensor:
        self.gam, self.bet = self.film(U)
        self.e0 = self.u0(U)
        return self._log_prob(X)

    def get_compatible_prior(self) -> DTransformerProposal:
        kwargs = self.get_constructor_args()
        # Pop irrelevant items
        kwargs.pop("u_dims")
        return DTransformerProposal(**kwargs)


class CondMutationProposal(SequenceCondSearchDistribution):
    """Conditional mutation proposal for sequences, q(X|U)."""

    def __init__(
        self,
        d_features: int,
        k_categories: int,
        u_dims: int,
        X0: Optional[Tensor] = None,
        U0: Optional[Tensor] = None,
        num_mutations: int = 10,
        max_seq_align_dist: Optional[int] = 10,
        clip_gradients: Optional[float] = None,
        samples: int = 1,
    ):
        SequenceCondSearchDistribution.__init__(
            self,
            d_features=d_features,
            k_categories=k_categories,
            u_dims=u_dims,
            samples=samples,
            clip_gradients=clip_gradients,
        )
        self.X0 = X0
        self.U0 = U0
        self.X0s = None
        self.num_mutations = num_mutations
        self.max_seq_align_dist = max_seq_align_dist

    @abstractmethod
    def sample(self, U: Tensor) -> Tensor: ...

    @abstractmethod
    def log_prob(self, X: Tensor, U: Tensor) -> Tensor: ...

    def set_seeds(self, X0: Tensor, U0: Tensor):
        self.X0 = X0
        self.U0 = U0
        self.X0s = None

    def clear_seeds(self):
        self.X0 = None
        self.U0 = None
        self.X0s = None

    def _check_seeds(self) -> Tuple[Tensor, Tensor]:
        if self.X0 is None or self.U0 is None:
            raise ValueError(
                "Properties (X0, U0) required, instantiate the object with "
                "this property, or assign it using obj.set_seeds(X0, U0)."
            )
        return self.X0, self.U0

    @torch.no_grad()
    def _match_seeds(self, U: Tensor) -> Tensor:
        """Get seeds, X0, that have U0 most similar to U"""
        X0, U0 = self._check_seeds()
        idxs = torch.argmax(U @ U0.T, dim=1)
        self.X0s = X0[idxs]
        return self.X0s


class CondTransformerMutationProposal(
    CondMutationProposal, _TransformerMutationMixin
):
    """
    Conditional transformer mutation proposal distribution, q(X|U).

    Combines `CondMutationProposal` and `TransformerMutationMixin` to generate
    mutation proposals for sequences conditioned on preference vectors U.
    Supports an optional `pad_token` to prevent mutations at padded positions.

    Parameters
    ----------
    d_features : int
        Length of the sequence (number of features).
    k_categories : int
        Number of possible token categories.
    u_dims : int
        Dimensionality of the conditioning preference vector U.
    X0 : Optional[Tensor], default=None
        Initial sequence(s) used as the base for mutations.
    U0 : Optional[Tensor], default=None
        Initial directions(s) associated with X0.
    num_mutations : int, default=10
        Number of mutations to apply per sample.
    embedding_dim : Optional[int]
        Dimension of token embeddings.
    nhead : int, default=2
        Number of attention heads in the transformer.
    dim_feedforward : int, default=128
        Dimension of the feedforward network in transformer layers.
    dropout : float, default=0.0
        Dropout probability between transformer layers.
    num_layers : int, default=1
        Number of transformer encoder layers.
    pad_token : Optional[int], default=None
        Token index reserved for padding; padded positions will not be mutated.
    clip_gradients : Optional[float], default=None
        Maximum gradient norm for clipping updates.
    samples : int, default=1
        Number of samples drawn when sampling mutations.
    """

    def __init__(
        self,
        d_features: int,
        k_categories: int,
        u_dims: int,
        X0: Optional[Tensor] = None,
        U0: Optional[Tensor] = None,
        num_mutations: int = 10,
        embedding_dim: Optional[int] = None,
        nhead: int = 2,
        dim_feedforward: int = 128,
        dropout: float = 0.0,
        num_layers: int = 1,
        mask_cnn_kernel: int = 5,
        pad_token: Optional[int] = None,  # immutable
        replacement: bool = True,
        clip_gradients: Optional[float] = None,
        samples: int = 1,
    ):
        self._save_constructor_args(locals())
        CondMutationProposal.__init__(
            self,
            d_features=d_features,
            k_categories=k_categories,
            u_dims=u_dims,
            X0=X0,
            U0=U0,
            num_mutations=num_mutations,
            clip_gradients=clip_gradients,
            samples=samples,
        )
        _TransformerMutationMixin.__init__(
            self,
            k_categories=k_categories,
            num_mutations=num_mutations,
            embedding_dim=embedding_dim,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            num_layers=num_layers,
            mask_cnn_kernel=mask_cnn_kernel,
            pad_token=pad_token,
            replacement=replacement,
        )
        self.film = FiLM(u_dims=u_dims, embedding_dim=self.e)
        self.u0 = nn.Linear(u_dims, self.e)

    @torch.no_grad()
    def sample(self, U: Tensor) -> Tensor:
        X0s = self._match_seeds(U)
        self.gam, self.bet = self.film(U)
        self.e0 = self.u0(U)
        return self._sample(X0s)

    def log_prob(self, X: Tensor, U: Tensor) -> Tensor:
        if self.X0s is not None:  # Evaluate against previous samples
            if X.shape != self.X0s.shape:
                raise ValueError(
                    "X must be the same shape as the last sample, or call "
                    "`obj.clear_seeds()` first."
                )
            X0 = self.X0s
        else:  # Evaluate against internal seeds
            X0, _ = self._check_seeds()
            if X.shape != X0.shape:
                raise ValueError(
                    "X must be the same shape as the internal seeds, obj.X0!"
                )
        self.gam, self.bet = self.film(U)
        self.e0 = self.u0(U)
        return self._log_prob(X, X0)

    def _change_embeddings(self, e: Tensor) -> Tensor:
        e = e * (1 + self.gam) + self.bet
        return e

    def _get_prefix(self, samples: int) -> Tensor:
        return self.e0

    def get_compatible_prior(self) -> TransformerMLMProposal:
        kwargs = self.get_constructor_args()
        # Pop irrelevant items
        for i in (
            "mask_cnn_kernel",
            "num_mutations",
            "replacement",
            "u_dims",
            "U0",
        ):
            kwargs.pop(i)
        return TransformerMLMProposal(**kwargs)


#
# Fitting routines for conditional proposals q(X|U)
#


def fit_ml(
    cproposal: ConditionalSearchDistribution,
    X: Tensor,
    U: Tensor,
    batch_size: int = 512,
    optimizer: Optimizer = torch.optim.AdamW,  # type: ignore
    optimizer_options: Optional[Dict[str, Any]] = None,
    stop_options: Optional[Dict[str, Any]] = None,
    device: str | torch.device = "cpu",
    callback: Optional[Callable[[int, Tensor, Tensor], None]] = None,
    seed: Optional[int] = None,
    val_proportion: float = 0,
) -> None:
    """Fit a *conditional* proposal q(X|U) by maximum likelihood.

    Parameters
    ----------
    cproposal : ConditionalSearchDistribution
        The conditional search distribution to fit.
    X, U : Tensor
        Training pairs. Must have the same first dimension (batch size).
    batch_size : int
        Minibatch size.
    optimizer : torch.optim.Optimizer class
        Optimizer class to instantiate.
    optimizer_options : dict
        Keyword arguments passed to the optimizer constructor.
    stop_options : dict
        Keyword arguments for `SEPlateauStopping`.
    device : str | torch.device
        Device on which to run training.
    callback : Callable[[int, Tensor, Tensor], None]
        Optional callback invoked each iteration with (step, train_loss, val_loss).
    seed : Optional[int]
        Seed controlling minibatch shuffling.
    val_proportion : float
        If ``> 0``, withhold this proportion of data as a fixed validation set
        and report validation loss every iteration using the same held-out
        indices. Set to 0 to disable validation.
    """
    if X.shape[0] != U.shape[0]:
        raise ValueError(
            f"X and U must have the same batch size, got {X.shape[0]} and {U.shape[0]}."
        )

    cproposal.to(device)
    optimizer_options = {} if optimizer_options is None else optimizer_options
    stop_options = {} if stop_options is None else stop_options

    # Optional gradient clipping via parameter hooks (same as proposals.fit_ml)
    clip_gradients(cproposal)  # type: ignore

    optim = optimizer(cproposal.parameters(), **optimizer_options)  # type: ignore
    stopping_criterion = SEPlateauStopping(**stop_options)  # type: ignore

    if val_proportion > 0:
        batch_gen = batch_indices_val(len(X), batch_size, seed, val_proportion)
    else:
        batch_gen = ((i, None) for i in batch_indices(len(X), batch_size, seed))

    vloss = torch.zeros(1)

    cproposal.train()
    for i, (bi, oobi) in enumerate(batch_gen):
        Xb = X[bi].to(device)
        Ub = U[bi].to(device)

        # Negative conditional log-likelihood
        loss = -cproposal.log_prob(Xb, Ub).mean()
        loss.backward()
        optim.step()
        optim.zero_grad()

        # Validation
        sloss = loss.detach()
        if oobi is not None:
            cproposal.eval()
            with torch.no_grad():
                X_val, U_val = X[oobi].to(device), U[oobi].to(device)
                vloss = -cproposal.log_prob(X_val, U_val).mean().detach()
                sloss = vloss
            cproposal.train()

        if callback is not None:
            callback(i, loss, vloss)
        if stopping_criterion.evaluate(fvals=sloss):
            break

    cproposal.eval()
