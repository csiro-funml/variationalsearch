"""Variational distributions and priors."""

import warnings
from abc import ABC, abstractmethod
from contextlib import contextmanager
from typing import Any, Callable, Dict, Optional, Tuple, Sequence

import torch
import torch.distributions as td
import torch.nn.functional as fnn

from torch import Tensor
from torch.optim import Optimizer

from vsd.augmentation import AugmentGenerator
from vsd.utils import (
    batch_indices,
    batch_indices_val,
    PositionalEncoding,
    Skip,
    Transpose,
    FuseNorm,
    SEPlateauStopping,
)


class _TestMixin:
    """Sample consistency unit test context manager"""

    def __init__(self):
        # Test-only recording hooks
        self._test_sample_consistency: bool = False
        self._last_sample_log_prob: Tensor | float = 0.0

    @contextmanager
    def record_sample_log_prob(self):
        """Context manager: record the last sample and its log-prob.

        For unit testing.

        When enabled, `sample` will compute and store the log-probability of the
        returned samples using `self.log_prob` and make it available via
        `last_sample_log_prob()` for unit tests.
        """
        prev = self._test_sample_consistency
        self._test_sample_consistency = True
        try:
            yield self
        finally:
            self._test_sample_consistency = prev


class SearchDistribution(ABC, torch.nn.Module):
    """Abstract base class for variational distributions."""

    prior_same_class = True  # Indicates this class can be used as its own prior

    def __init__(
        self, samples: int = 100, clip_gradients: Optional[float] = None
    ) -> None:
        # initialize Module explicitly to avoid MRO conflicts
        torch.nn.Module.__init__(self)
        self.samples = samples
        self.clip_gradients = clip_gradients

    def forward(self, samples: Optional[int] = None) -> Tuple[Tensor, Tensor]:
        """Draw samples and return their log-density under ``q``.

        Parameters
        ----------
        samples : int, optional
            Number of samples to draw. Defaults to the value configured at
            initialisation.

        Returns
        -------
        tuple[Tensor, Tensor]
            Pair ``(X, logqX)`` where ``X`` has shape ``(samples, ...)`` and
            ``logqX`` is the element-wise log-probability.
        """
        samples = self.samples if samples is None else samples
        with torch.no_grad():
            Xs = self.sample(torch.Size([samples]))
        logqX = self.log_prob(Xs)
        return Xs, logqX

    @abstractmethod
    def sample(self, sample_shape: torch.Size = torch.Size([1])) -> Tensor:
        """Generate samples from the proposal.

        Parameters
        ----------
        sample_shape : torch.Size, optional
            Leading batch dimensions for the sample call. Defaults to a single
            sample along the first dimension.

        Returns
        -------
        Tensor
            Samples drawn from the proposal ``q``.
        """

    @abstractmethod
    def log_prob(self, X: Tensor) -> Tensor:
        """Evaluate ``log q(X)`` element-wise.

        Parameters
        ----------
        X : Tensor
            Batch of samples at which to evaluate the log-density.

        Returns
        -------
        Tensor
            Log-probabilities with one entry per row of ``X``.
        """

    def set_dropout_p(self, p: float):
        """Reset dropout probability on all contained modules.

        Useful when alternating between train/eval style loops.

        Parameters
        ----------
        p : float
            Dropout probability to assign to every ``Dropout``
            /``MultiheadAttention`` module.
        """
        for m in self.modules():
            if isinstance(m, torch.nn.Dropout):
                m.p = p


class SimpleSearchDistribution(SearchDistribution, _TestMixin):
    """Convenience base class for proposals defined via distributions.

    Subclasses implement ``_construct_q`` returning a ``torch.distributions``
    object; sampling and log-probability are handled centrally.
    """

    def __init__(
        self, samples: int = 100, clip_gradients: Optional[float] = None
    ) -> None:
        SearchDistribution.__init__(
            self, samples=samples, clip_gradients=clip_gradients
        )
        _TestMixin.__init__(self)

    def log_prob(self, X: Tensor) -> Tensor:
        """Evaluate ``log q(X)`` via the constructed distribution."""
        return self._construct_q().log_prob(X)

    @torch.no_grad()
    def sample(self, sample_shape: torch.Size = torch.Size([1])) -> Tensor:
        """Draw samples with optional recording for unit tests."""
        q = self._construct_q()
        Xs = q.sample(sample_shape)

        # Testing
        if self._test_sample_consistency:
            self._last_sample_log_prob = q.log_prob(Xs)

        return Xs

    @abstractmethod
    def _construct_q(self) -> td.Distribution: ...


#
# Continuous data
#


class GaussianKDEProposal(SimpleSearchDistribution):
    """Gaussian KDE/mixture search distribution.

    Parameters
    ----------
    d_features : int
        Dimensionality of the design space.
    k_components : int, default=20
        Number of mixture components.
    scale : float, default=1
        Shared standard deviation across mixture components.
    mu_scale_init : float, default=1
        Scaling factor for the initial component means.
    samples : int, default=100
        Default number of samples to draw per ``forward`` call.
    """

    def __init__(
        self,
        d_features: int,
        k_components: int = 20,
        scale: float = 1,
        mu_scale_init: float = 1,
        samples: int = 100,
    ) -> None:
        super().__init__(samples)
        self.mixl = torch.nn.Parameter(torch.ones(k_components))
        init_mus = torch.randn(k_components, d_features) * mu_scale_init
        self.mus = torch.nn.Parameter(init_mus)
        self.scale = torch.nn.Parameter(torch.log(torch.scalar_tensor(scale)))

    def _construct_q(self) -> td.Distribution:
        mix = td.Categorical(logits=self.mixl)
        kern = td.Independent(
            td.Normal(loc=self.mus, scale=torch.exp(self.scale)), 1
        )
        kde = td.MixtureSameFamily(mix, kern)
        return kde


#
# Sequence data -- multi-categorical
#


class SequenceSearchDistribution(SimpleSearchDistribution):
    """Abstract base for proposals over discrete sequences.

    Parameters
    ----------
    d_features : int
        Sequence length.
    k_categories : int
        Alphabet size for each position.
    samples : int, default=100
        Default number of samples to draw.
    clip_gradients : float, optional
        Gradient clipping value applied during optimisation.
    """

    def __init__(
        self,
        d_features: int,
        k_categories: int,
        samples: int = 100,
        clip_gradients: float | None = None,
    ) -> None:
        super().__init__(samples, clip_gradients)
        self.d = d_features
        self.k = k_categories


class SequenceUninformativePrior(SequenceSearchDistribution):
    """Uniform prior over sequences with no learnable parameters.

    Parameters
    ----------
    d_features : int
        Sequence length.
    k_categories : int
        Alphabet size; each position is uniform over ``k_categories``.
    """

    def __init__(self, d_features: int, k_categories: int):
        super().__init__(d_features=d_features, k_categories=k_categories)
        self.register_buffer("l", torch.ones(self.d, self.k))

    def _construct_q(self):
        return td.Independent(td.Categorical(logits=self.l), 1)


class MultiCategoricalProposal(SequenceSearchDistribution):
    """Independent/mean-field multi-categorical search distribution.

    Parameters
    ----------
    d_features : int
        Sequence length.
    k_categories : int
        Alphabet size.
    samples : int, default=100
        Default number of samples to draw.
    uniform_init : bool, default=False
        If ``True``, initialise logits uniformly; otherwise use random
        symmetric initialisation.
    """

    def __init__(
        self,
        d_features: int,
        k_categories: int,
        samples: int = 100,
        uniform_init: bool = False,
    ) -> None:
        super().__init__(
            d_features=d_features, k_categories=k_categories, samples=samples
        )
        if uniform_init:
            logits = torch.zeros(self.d, self.k)
        else:
            logits = _rinit(torch.Size((self.d, self.k)), self.k)
        self.phi = torch.nn.Parameter(logits)

    def _construct_q(self) -> td.Distribution:
        cat = td.Independent(td.Categorical(logits=self.phi), 1)
        return cat


#
# Sequence data -- auto-regressive
#


class AutoRegressiveSearchDistribution(SearchDistribution):
    """Abstract base for autoregressive sequence proposals.

    Parameters
    ----------
    d_features : int
        Sequence length.
    k_categories : int
        Alphabet size.
    samples : int, default=100
        Default number of samples drawn by ``forward``.
    clip_gradients : float, optional
        Gradient clipping value applied during optimisation.
    """

    def __init__(
        self,
        d_features: int,
        k_categories: int,
        samples: int = 100,
        clip_gradients: Optional[float] = None,
    ) -> None:
        super().__init__(samples, clip_gradients)
        self.d = d_features
        self.k = k_categories


class _LSTMMixin(torch.nn.Module, _TestMixin):
    """Mixin providing LSTM backbone for autoregressive proposals."""

    def __init__(
        self,
        d_features: int,
        k_categories: int,
        embedding_dim: Optional[int] = None,
        hidden_size: Optional[int] = None,
        num_layers: int = 1,
        dropout: float = 0.0,
    ):
        torch.nn.Module.__init__(self)
        _TestMixin.__init__(self)
        self.d = d_features
        if embedding_dim is None:
            embedding_dim = max(8, k_categories // 2)
        self.e = embedding_dim
        if hidden_size is None:
            hidden_size = 8 * embedding_dim
        self.pos = PositionalEncoding(emb_size=self.e)
        self.emb = torch.nn.Embedding(
            num_embeddings=k_categories, embedding_dim=self.e
        )
        self.lstm = torch.nn.LSTM(
            input_size=self.e,
            hidden_size=hidden_size,
            batch_first=True,
            num_layers=num_layers,
            dropout=dropout,
        )
        self.dec = torch.nn.Linear(hidden_size, k_categories)

    @abstractmethod
    def _get_prefix(self, samples: int) -> Tensor: ...

    def _change_embeddings(self, e: Tensor) -> Tensor:
        return e

    def _sample(self, sample_shape: torch.Size = torch.Size([1])) -> Tensor:
        if len(sample_shape) > 1:
            raise ValueError(
                "sample_shape of more than one dimension not implemented."
            )
        samples = sample_shape[0]
        e0 = self._get_prefix(samples)
        device = e0.device
        Xs = torch.zeros(samples, self.d, device=device, dtype=torch.long)
        e = self.pos(e0.unsqueeze(1))
        hc = None

        for i in range(self.d):
            e = self._change_embeddings(e)
            o, hc = self.lstm(e, hc)
            q = td.Categorical(logits=self.dec(o))
            xs = q.sample()
            Xs[:, i] = xs.squeeze(-1)
            e = self.pos(self.emb(xs), pos_ind=i + 1)

            # Testing only
            if self._test_sample_consistency:
                self._last_sample_log_prob += q.log_prob(xs).squeeze(-1)

        return Xs

    def _log_prob(self, X: Tensor) -> Tensor:
        e0 = self._get_prefix(X.shape[0])
        e = self.pos(torch.hstack((e0.unsqueeze(1), self.emb(X[:, :-1]))))
        e = self._change_embeddings(e)
        o, _ = self.lstm(e, None)
        q = td.Independent(td.Categorical(logits=self.dec(o)), 1)
        return q.log_prob(X)

    def load(self, other: "_LSTMMixin"):
        for attr in ("emb", "pos", "lstm", "dec"):
            self_attr = getattr(self, attr)
            other_attr = getattr(other, attr, None)
            if self_attr is not None and other_attr is not None:
                self_attr.load_state_dict(other_attr.state_dict())


class LSTMProposal(AutoRegressiveSearchDistribution, _LSTMMixin):
    """Long short-term memory RNN search distribution.

    Parameters
    ----------
    d_features : int
        Sequence length.
    k_categories : int
        Alphabet size.
    embedding_dim : int, optional
        Token embedding dimension. Defaults to ``max(8, k//2)``.
    hidden_size : int, optional
        Hidden size of the LSTM; defaults to ``8 * embedding_dim``.
    num_layers : int, default=1
        Number of stacked LSTM layers.
    dropout : float, default=0.0
        Dropout probability applied between LSTM layers.
    samples : int, default=100
        Default number of samples returned by ``forward``.
    clip_gradients : float, optional
        Gradient clipping value applied during optimisation.
    """

    def __init__(
        self,
        d_features: int,
        k_categories: int,
        embedding_dim: Optional[int] = None,
        hidden_size: Optional[int] = None,
        num_layers: int = 1,
        dropout: float = 0.0,
        samples: int = 100,
        clip_gradients: Optional[float] = None,
    ) -> None:
        AutoRegressiveSearchDistribution.__init__(
            self,
            d_features=d_features,
            k_categories=k_categories,
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
        self.e0 = torch.nn.Parameter(_rinit(torch.Size([self.e]), self.e))

    def _get_prefix(self, samples: int) -> Tensor:
        return self.e0.tile(samples, 1)

    def sample(self, sample_shape: torch.Size = torch.Size([1])) -> Tensor:
        """Draw autoregressive samples using the learned LSTM."""
        Xs = self._sample(sample_shape=sample_shape)
        return Xs

    def log_prob(self, X: Tensor) -> Tensor:
        """Evaluate the autoregressive log-probability under the LSTM."""
        return self._log_prob(X=X)


class _TransformerBackbone(torch.nn.Module):
    """Shared embedding, positional encoding, and transformer encoder stack.

    Parameters
    ----------
    k_categories : int
        Vocabulary size for the embedding.
    embedding_dim : int, optional
        Dimensionality of token embeddings. Defaults to ``max(8, k//2) * nhead``.
    nhead : int, default=2
        Number of attention heads.
    dim_feedforward : int, default=128
        Hidden dimension of the feed-forward sublayers.
    dropout : float, default=0.0
        Dropout probability applied throughout the transformer.
    num_layers : int, default=1
        Number of encoder layers.
    add_mask_token : bool, default=False
        Whether to append an extra embedding slot used as a mask token.
    """

    def __init__(
        self,
        k_categories: int,
        embedding_dim: Optional[int] = None,
        nhead: int = 2,
        dim_feedforward: int = 128,
        dropout: float = 0.0,
        num_layers: int = 1,
        add_mask_token: bool = False,  # Add a mask token to the embedding of k+1
    ):
        super().__init__()

        # Embedding
        if embedding_dim is None:
            embedding_dim = max(8, k_categories // 2) * nhead
        self.e = embedding_dim
        self.k = k_categories
        self.emb = torch.nn.Embedding(
            k_categories + 1 if add_mask_token else k_categories, embedding_dim
        )

        # Positional encoding
        self.pos = PositionalEncoding(emb_size=embedding_dim)

        # Encoder
        enc_layer = torch.nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.tfm = torch.nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.dec = torch.nn.Linear(embedding_dim, k_categories)

    def _change_embeddings(self, e: Tensor) -> Tensor:
        return e

    def encode(self, X: Tensor) -> torch.Tensor:
        """Return encoded hidden states with bi-directional masking.

        Parameters
        ----------
        X : Tensor
            Token indices of shape ``(B, L)``.

        Returns
        -------
        Tensor
            Encoded representations with shape ``(B, L, e)``.
        """
        return self.tfm(self.pos(self._change_embeddings(self.emb(X))))

    def load(self, other: "_TransformerBackbone"):
        for attr in ("emb", "pos", "tfm", "dec"):
            self_attr = getattr(self, attr)
            other_attr = getattr(other, attr, None)
            if self_attr is not None and other_attr is not None:
                self_attr.load_state_dict(other_attr.state_dict())

    def set_dropout_p(self, p: float):
        """Reset dropout p -- useful for multiple training steps."""
        for m in self.modules():
            if isinstance(m, torch.nn.Dropout):
                m.p = p
            # MultiheadAttention also has its own dropout parameter:
            if isinstance(m, torch.nn.MultiheadAttention):
                m.dropout = p


class _DTransformerMixin(_TransformerBackbone, _TestMixin):
    """Mixin implementing decoder-only transformer sampling utilities."""

    def __init__(
        self,
        d_features: int,
        k_categories: int,
        embedding_dim: Optional[int] = None,
        nhead: int = 2,
        dim_feedforward: int = 128,
        dropout: float = 0.0,
        num_layers: int = 1,
    ):
        _TransformerBackbone.__init__(
            self,
            k_categories=k_categories,
            embedding_dim=embedding_dim,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            num_layers=num_layers,
        )
        _TestMixin.__init__(self)
        self.d = d_features

    @abstractmethod
    def _get_prefix(self, samples: int) -> Tensor: ...

    def _sample(self, sample_shape: torch.Size = torch.Size([1])) -> Tensor:
        if len(sample_shape) > 1:
            raise ValueError(
                "sample_shape of more than one dimension not implemented."
            )
        samples = sample_shape[0]
        e0 = self._get_prefix(samples)
        device = e0.device
        e = self.pos(torch.zeros(samples, self.d + 1, self.e, device=device))
        e[:, 0] = self._change_embeddings(e[:, 0] + e0)
        Xs = torch.zeros(samples, self.d, device=device, dtype=torch.long)
        mask = torch.nn.Transformer.generate_square_subsequent_mask(self.d)
        mask = mask.to(device)

        for i, j in zip(range(self.d), range(1, self.d + 1)):
            o = self.tfm(e[:, :j], mask=mask[:j, :j])[:, -1]
            q = td.Categorical(logits=self.dec(o))
            xs = q.sample()
            Xs[:, i] = xs
            e[:, j] = self._change_embeddings(e[:, j] + self.emb(xs))

            # Testing only
            if self._test_sample_consistency:
                self._last_sample_log_prob += q.log_prob(xs).squeeze(-1)

        return Xs

    def _log_prob(self, X: Tensor) -> Tensor:
        e0 = self._get_prefix(X.shape[0])
        e = self.pos(torch.hstack((e0.unsqueeze(1), self.emb(X[:, :-1]))))
        e = self._change_embeddings(e)
        mask = torch.nn.Transformer.generate_square_subsequent_mask(self.d)
        mask = mask.to(X.device)
        o = self.tfm(e, mask=mask)
        q = td.Independent(td.Categorical(logits=self.dec(o)), 1)
        return q.log_prob(X)


class DTransformerProposal(
    AutoRegressiveSearchDistribution, _DTransformerMixin
):
    """Causal decoder-only transformer search distribution.

    Parameters
    ----------
    d_features : int
        Sequence length.
    k_categories : int
        Alphabet size.
    embedding_dim : int, optional
        Token embedding dimension. Defaults to ``max(8, k//2) * nhead``.
    nhead : int, default=2
        Number of attention heads.
    dim_feedforward : int, default=128
        Hidden dimension of the feed-forward sublayers.
    dropout : float, default=0.0
        Dropout probability used across the transformer stack.
    num_layers : int, default=1
        Number of decoder blocks.
    samples : int, default=100
        Default number of samples returned by ``forward``.
    clip_gradients : float, optional
        Gradient clipping value applied during optimisation.
    """

    def __init__(
        self,
        d_features: int,
        k_categories: int,
        embedding_dim: Optional[int] = None,
        nhead: int = 2,
        dim_feedforward: int = 128,
        dropout: float = 0.0,
        num_layers: int = 1,
        samples: int = 100,
        clip_gradients: Optional[float] = None,
    ) -> None:
        AutoRegressiveSearchDistribution.__init__(
            self,
            d_features=d_features,
            k_categories=k_categories,
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
        self.e0 = torch.nn.Parameter(_rinit(torch.Size([self.e]), self.e))

    def _get_prefix(self, samples: int) -> Tensor:
        return self.e0.tile(samples, 1)

    def sample(self, sample_shape: torch.Size = torch.Size([1])) -> Tensor:
        """Draw autoregressive samples using the decoder-only transformer."""
        Xs = self._sample(sample_shape=sample_shape)
        return Xs

    def log_prob(self, X: Tensor) -> Tensor:
        """Evaluate the autoregressive log-probability under the transformer."""
        return self._log_prob(X=X)


#
# Sequence data -- masking and mutation
#


class MaskedSearchDistribution(SequenceSearchDistribution):
    """Abstract base for masked/mutation proposals ``q(X | X0)``.

    Parameters
    ----------
    d_features : int
        Sequence length.
    k_categories : int
        Alphabet size.
    X0 : Tensor, optional
        Seed sequences the proposal mutates from.
    samples : int, default=100
        Default number of samples per ``forward`` call.
    clip_gradients : float, optional
        Gradient clipping value applied during optimisation.
    """

    def __init__(
        self,
        d_features: int,
        k_categories: int,
        X0: Optional[Tensor] = None,
        samples: int = 100,
        clip_gradients: Optional[float] = None,
    ):
        super().__init__(
            d_features=d_features,
            k_categories=k_categories,
            samples=samples,
            clip_gradients=clip_gradients,
        )
        self.X0 = X0
        self.X0s = None

    @abstractmethod
    def log_prob(self, X: Tensor) -> Tensor: ...

    @abstractmethod
    def sample(self, sample_shape: torch.Size = torch.Size([1])) -> Tensor: ...

    def set_seeds(self, X0: Tensor):
        """Register seed sequences used during sampling."""
        self.X0 = X0
        self.X0s = None

    def clear_seeds(self):
        """Remove cached seeds to force re-sampling on next draw."""
        self.X0 = None
        self.X0s = None

    @torch.no_grad()
    def _sample_seeds(self, samples: int) -> Tensor:
        """Draw a minibatch of seeds with/without replacement."""
        X0 = self._check_seeds()
        N0 = len(X0)
        if N0 < samples:  # With replacement
            self.X0s = X0[torch.randint(high=N0, size=torch.Size([samples]))]
        else:  # Without replacement
            self.X0s = X0[torch.randperm(N0)[:samples]]
        return self.X0s

    def _check_seeds(self) -> Tensor:
        if self.X0 is None:
            raise ValueError(
                "Property X0 required, instantiate the object with this "
                "property, or assign it using obj.set_seeds(X0)."
            )
        return self.X0

    def _construct_q(self, X0: Tensor) -> td.Distribution:
        raise NotImplementedError


class _TransformerMLMBackbone(_TransformerBackbone):
    """Backbone for masked language-model style proposals."""

    def __init__(
        self,
        k_categories: int,
        embedding_dim: int | None = None,
        nhead: int = 2,
        dim_feedforward: int = 128,
        dropout: float = 0,
        num_layers: int = 1,
        pad_token: Optional[int] = None,
    ):
        _TransformerBackbone.__init__(
            self,
            k_categories=k_categories,
            embedding_dim=embedding_dim,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            num_layers=num_layers,
            add_mask_token=True,
        )
        self.pad_token = pad_token
        self.mask_token = self.k

    @torch.no_grad()
    def _get_diffs(self, X0: Tensor, X: Tensor) -> Tensor:
        if X0.shape != X.shape:
            raise ValueError("X0 and X have to be the same shape.")
        mask = X != X0
        return mask

    def _construct_token_q(
        self,
        X: Tensor,
        mask: Tensor,
        replacement: bool = True,
    ) -> torch.distributions.Categorical:
        # Mask out tokens to sample
        X_mask = X.clone().masked_fill(mask, self.mask_token)

        # Get logits
        logits = self.dec(self.encode(X_mask))

        # Don't allow padding token generation
        if self.pad_token is not None:
            logits[..., self.pad_token] = float("-inf")

        # Don't re-sample existing token
        if not replacement:
            forbidden = fnn.one_hot(X, num_classes=self.k).bool()
            forbidden = forbidden & mask.unsqueeze(-1)
            logits[forbidden] = float("-inf")

        q = torch.distributions.Categorical(logits=logits)
        return q

    def _log_probx_pad_aware(
        self,
        q: td.Distribution,
        mask: Tensor,
        X: Tensor,
        X0: Optional[Tensor] = None,
    ) -> Tensor:
        if self.pad_token is not None:
            Xp = X if X0 is None else X0
            mask = mask & (Xp != self.pad_token)
        logq = q.log_prob(X).masked_fill_(~mask, 0).sum(dim=1)
        return logq

    def _log_probm_pad_aware(
        self, logits: Tensor, mask: Tensor, X: Tensor
    ) -> Tensor:
        if self.pad_token is not None:
            mask = mask & (X != self.pad_token)
        logq = torch.log_softmax(logits, dim=1).masked_fill(~mask, 0).sum(dim=1)
        return logq


class _TransformerMLMMixin(_TransformerMLMBackbone, _TestMixin):
    """Mixin providing MLM-style masking, sampling, and scoring."""

    def __init__(
        self,
        k_categories: int,
        embedding_dim: int | None = None,
        nhead: int = 2,
        dim_feedforward: int = 128,
        dropout: float = 0,
        num_layers: int = 1,
        mask_p: float = 0.15,
        pad_token: Optional[int] = None,
    ):
        _TransformerMLMBackbone.__init__(
            self,
            k_categories=k_categories,
            embedding_dim=embedding_dim,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            num_layers=num_layers,
            pad_token=pad_token,
        )
        _TestMixin.__init__(self)
        self.mask_p = mask_p

    @torch.no_grad()
    def _sample_mask(self, X: Tensor) -> Tensor:
        probs = torch.full(X.shape, self.mask_p, device=X.device)

        # Don't change padded token positions
        if self.pad_token is not None:
            probs = probs.masked_fill(X == self.pad_token, 0.0)

        mask = torch.bernoulli(input=probs).bool()
        return mask

    def _log_prob(self, X: Tensor, X0: Optional[Tensor] = None) -> Tensor:
        if X0 is None:
            mask = self._sample_mask(X)
            q = self._construct_token_q(X, mask, replacement=True)  # self mask
        else:
            mask = self._get_diffs(X0, X)
            if self._test_sample_consistency:
                mask = self._last_mask
            q = self._construct_token_q(X0, mask, replacement=True)  # seed mask

        # Token prob
        logqX = self._log_probx_pad_aware(q, mask, X, X0)
        return logqX

    @torch.no_grad()
    def _sample(
        self,
        X0: Tensor,
        gibbs_steps: int = 20,
    ) -> Tensor:
        Xs = X0.clone()
        for _ in range(gibbs_steps):
            mask = self._sample_mask(Xs)
            q = self._construct_token_q(Xs, mask, replacement=True)
            Xs[mask] = q.sample()[mask]

        # Testing only
        if self._test_sample_consistency:
            logqX = self._log_probx_pad_aware(q, mask, Xs, X0)
            self._last_mask = mask  # Consistent masking for tests
            self._last_sample_log_prob = logqX

        return Xs


class TransformerMLMProposal(MaskedSearchDistribution, _TransformerMLMMixin):
    """Masked Transformer model that randomly mutates.

    Parameters
    ----------
    d_features : int
        Sequence length.
    k_categories : int
        Alphabet size.
    mask_p : float, default=0.15
        Probability of masking each token during sampling/training.
    X0 : Tensor, optional
        Seed sequences to mutate from when acting as a transition model.
    embedding_dim : int, optional
        Token embedding dimension. Defaults to ``max(8, k//2) * nhead``.
    nhead : int, default=2
        Number of attention heads.
    dim_feedforward : int, default=128
        Hidden dimension of the feed-forward sublayers.
    dropout : float, default=0.0
        Dropout probability used throughout the transformer.
    num_layers : int, default=1
        Number of encoder layers.
    pad_token : int, optional
        Token index treated as padding/immutable.
    gibbs_steps : int, default=20
        Number of masked language model refinement steps during sampling.
    samples : int, default=100
        Default number of samples returned by ``forward``.
    clip_gradients : float, optional
        Gradient clipping value applied during optimisation.
    """

    def __init__(
        self,
        d_features: int,
        k_categories: int,
        mask_p: float = 0.15,
        X0: Optional[Tensor] = None,
        embedding_dim: Optional[int] = None,
        nhead: int = 2,
        dim_feedforward: int = 128,
        dropout: float = 0.0,
        num_layers: int = 1,
        pad_token: Optional[int] = None,
        gibbs_steps: int = 20,
        samples: int = 100,
        clip_gradients: Optional[float] = None,
    ):
        MaskedSearchDistribution.__init__(
            self,
            d_features=d_features,
            k_categories=k_categories,
            X0=X0,
            samples=samples,
            clip_gradients=clip_gradients,
        )
        _TransformerMLMMixin.__init__(
            self,
            k_categories=k_categories,
            mask_p=mask_p,
            embedding_dim=embedding_dim,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            num_layers=num_layers,
            pad_token=pad_token,
        )
        self.gibbs_steps = gibbs_steps

    @torch.no_grad()
    def sample(
        self,
        sample_shape: torch.Size = torch.Size([1]),
    ) -> Tensor:
        """Draw samples by repeatedly masking and resampling tokens."""
        if len(sample_shape) > 1:
            raise ValueError("Sample shapes of dim > 1 not implemented.")
        samples = int(sample_shape[0])
        X0s = self._sample_seeds(samples=samples)
        Xs = self._sample(X0s, gibbs_steps=self.gibbs_steps)
        return Xs

    def log_prob(self, X: Tensor) -> Tensor:
        """Compute the masked-language-model log-probability of ``X``."""
        X0 = None  # Evaluate against masked X0
        if self.X0s is not None:  # Evaluate against previous samples
            if X.shape != self.X0s.shape:
                raise ValueError(
                    "X must be the same shape as the last sample, or call "
                    "`self.clear_seeds()` first."
                )
            X0 = self.X0s
        return self._log_prob(X, X0)


class _TransformerMutationMixin(_TransformerMLMBackbone, _TestMixin):
    """Mixin that augments the MLM backbone with a learned mask decoder."""

    def __init__(
        self,
        k_categories: int,
        num_mutations: int = 10,
        embedding_dim: Optional[int] = None,
        nhead: int = 2,
        dim_feedforward: int = 128,
        dropout: float = 0.0,
        num_layers: int = 1,
        mask_cnn_kernel: int = 5,
        pad_token: Optional[int] = None,  # immutable
        replacement: bool = True,  # allow original token replacement
    ):
        _TransformerMLMBackbone.__init__(
            self,
            k_categories=k_categories,
            embedding_dim=embedding_dim,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            num_layers=num_layers,
            pad_token=pad_token,
        )
        _TestMixin.__init__(self)
        self.num_mutations = num_mutations
        self.replacement = replacement
        if mask_cnn_kernel % 2 == 0:
            raise ValueError("mask_cnn_kernel must be odd.")

        out_logit = torch.nn.Linear(in_features=self.e, out_features=1)
        with torch.no_grad():
            out_logit.bias.data.copy_(torch.tensor([-2.0]))  # start mask p low

        # Mask model
        self.mask_dec = torch.nn.Sequential(
            FuseNorm(self.e, alpha0=1e-5),
            Skip(
                torch.nn.Sequential(
                    Transpose(),
                    torch.nn.Conv1d(
                        in_channels=self.e,
                        out_channels=self.e,
                        kernel_size=mask_cnn_kernel,
                        dilation=1,
                        padding=mask_cnn_kernel // 2,
                    ),
                    torch.nn.SiLU(),
                    Transpose(),
                )
            ),
            torch.nn.LayerNorm(normalized_shape=self.e),
            torch.nn.LeakyReLU(),  # "Crisper" logits
            out_logit,
        )

    def load(self, other: _TransformerMLMBackbone):
        _TransformerMLMBackbone.load(self, other)
        if hasattr(other, "mask_dec"):
            self.mask_dec.load_state_dict(other.mask_dec.state_dict())

    def _mask_logits_pad_aware(self, X0: Tensor) -> Tensor:
        # Skip around the backbone transformer if needed
        emb = self.pos(self._change_embeddings(self.emb(X0)))
        enc = self.tfm(emb)
        logits = self.mask_dec((emb, enc)).squeeze(-1)

        # Don't change padded token positions
        if self.pad_token is not None:
            logits = logits.masked_fill(X0 == self.pad_token, float("-inf"))

        return logits

    def _log_prob(self, X: Tensor, X0: Tensor) -> Tensor:
        mask = self._get_diffs(X0, X)

        with torch.no_grad():
            dmuts = mask.sum(dim=1)
            if (dmuts > self.num_mutations).any():
                warnings.warn(
                    f"More than {self.num_mutations} encountered!",
                    RuntimeWarning,
                )

        # Mask prob
        logits = self._mask_logits_pad_aware(X0)
        logqm = self._log_probm_pad_aware(logits, mask, X0)

        # Token prob
        q = self._construct_token_q(X0, mask, replacement=self.replacement)
        logqX = self._log_probx_pad_aware(q, mask, X, X0)

        return logqX + logqm

    @torch.no_grad()
    def _sample(
        self,
        X0: Tensor,
    ) -> Tensor:
        Xs = X0.clone()

        # Sample mask/mutations
        logits = self._mask_logits_pad_aware(X0)

        pos = torch.multinomial(
            fnn.softmax(logits, dim=-1),
            num_samples=self.num_mutations,
            replacement=False,  # require num_mutations count in sample
        )
        mask = torch.zeros_like(Xs).bool()
        mask.scatter_(dim=1, index=pos, value=True)

        # Sample tokens
        q = self._construct_token_q(X0, mask, replacement=self.replacement)
        Xs[mask] = q.sample()[mask]

        # Testing only
        if self._test_sample_consistency:
            logqX = self._log_probx_pad_aware(q, mask, Xs, X0)
            logqm = self._log_probm_pad_aware(logits, mask, X0)
            self._last_sample_log_prob = logqX + logqm

        return Xs


class TransformerMutationProposal(
    MaskedSearchDistribution, _TransformerMutationMixin
):
    """Masked Transformer model that learns mutation locations and values.

    Parameters
    ----------
    d_features : int
        Sequence length.
    k_categories : int
        Alphabet size.
    num_mutations : int, default=10
        Number of positions to mutate per sample.
    X0 : Tensor, optional
        Seed sequences to mutate from.
    embedding_dim : int, optional
        Token embedding dimension. Defaults to ``max(8, k//2) * nhead``.
    nhead : int, default=2
        Number of attention heads.
    dim_feedforward : int, default=128
        Hidden dimension of the feed-forward sublayers.
    dropout : float, default=0.0
        Dropout probability used across the transformer and mask decoder.
    num_layers : int, default=1
        Number of encoder layers.
    mask_cnn_kernel : int, default=5
        Kernel size of the 1D convolution used in the mask decoder (must be odd).
    pad_token : int, optional
        Token index treated as immutable.
    replacement : bool, default=False
        Whether mutations may keep the original token.
    samples : int, default=100
        Default number of samples returned by ``forward``.
    clip_gradients : float, optional
        Gradient clipping value applied during optimisation.
    """

    prior_same_class = False  # Use the MLM as a prior

    def __init__(
        self,
        d_features: int,
        k_categories: int,
        num_mutations: int = 10,
        X0: Optional[Tensor] = None,
        embedding_dim: Optional[int] = None,
        nhead: int = 2,
        dim_feedforward: int = 128,
        dropout: float = 0.0,
        num_layers: int = 1,
        mask_cnn_kernel: int = 5,
        pad_token: Optional[int] = None,
        replacement: bool = False,  # allow original token replacement
        samples: int = 100,
        clip_gradients: Optional[float] = None,
    ):
        self._save_constructor_args(locals())
        MaskedSearchDistribution.__init__(
            self,
            d_features=d_features,
            k_categories=k_categories,
            X0=X0,
            samples=samples,
            clip_gradients=clip_gradients,
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

    @torch.no_grad()
    def sample(
        self,
        sample_shape: torch.Size = torch.Size([1]),
    ) -> Tensor:
        """Sample by selecting mutation sites and resampling masked tokens."""
        if len(sample_shape) > 1:
            raise ValueError("Sample shapes of dim > 1 not implemented.")
        samples = int(sample_shape[0])
        X0s = self._sample_seeds(samples=samples)
        Xs = self._sample(X0s)
        return Xs

    def log_prob(self, X: Tensor) -> Tensor:
        """Return ``log q(X | X0)`` for the learnt mutation model."""
        if self.X0s is not None:  # Evaluate against previous samples
            if X.shape != self.X0s.shape:
                raise ValueError(
                    "X must be the same shape as the last sample, or call "
                    "`obj.clear_seeds()` first."
                )
            X0 = self.X0s
        else:  # Evaluate against internal seeds
            X0 = self._check_seeds()
            if X.shape != X0.shape:
                raise ValueError(
                    "X must be the same shape as the internal seeds, obj.X0!"
                )
        return self._log_prob(X, X0)

    def _save_constructor_args(self, local_vars: Dict[str, Any]):
        """Record constructor arguments from ``locals()`` for later reuse."""
        self._constructor_args = {
            k: v
            for k, v in local_vars.items()
            if k not in ("self", "__class__")
        }

    def get_constructor_args(self) -> Dict[str, Any]:
        """Return a copy of the arguments used to instantiate the proposal."""
        if not hasattr(self, "_constructor_args"):
            raise ValueError("Consturctor arguments not saved!")
        return self._constructor_args.copy()

    def get_compatible_prior(self) -> TransformerMLMProposal:
        """Instantiate a matching :class:`TransformerMLMProposal` prior."""
        kwargs = self.get_constructor_args()
        # Pop irrelevant items
        for i in ("mask_cnn_kernel", "num_mutations", "replacement"):
            kwargs.pop(i)
        return TransformerMLMProposal(**kwargs)


#
# Fitting routines
#


def fit_ml(
    proposal: SearchDistribution,
    X: Tensor,
    X0: Optional[Tensor] = None,
    batch_size: int = 32,
    optimizer: Optimizer = torch.optim.AdamW,  # type: ignore
    optimizer_options: Optional[Dict[str, Any]] = None,
    stop_options: Optional[Dict[str, Any]] = None,
    device: str | torch.device = "cpu",
    callback: Optional[Callable[[int, Tensor, Tensor], None]] = None,
    seed: Optional[int] = None,
    val_proportion: float = 0,
    augmenter: Optional[AugmentGenerator] = None,
    augmentation_p: float = 0.1,
):
    """Fit a proposal distribution q(X) or q(X|X0) by maximum likelihood.

    Parameters
    ----------
    proposal : SearchDistribution
        The search distribution to fit.
    X : Tensor
        Training samples.
    X0 : Optional[Tensor]
        Conditioning sequences for transition proposals; must match X in batch size.
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
    augmenter : Optional[AugmentGenerator]
        Optional data augmenter for marginal models.
    augmentation_p : float
        Proportion of each batch to replace with augmented samples when augmenter is provided.
    """
    istransition = isinstance(proposal, MaskedSearchDistribution)
    haveX0 = X0 is not None
    if istransition and not haveX0:
        istransition = False  # train as a masked language model if supported
    elif haveX0 and not istransition:
        raise ValueError(
            "X0 data givn, but proposal is not TransitionSearchDistribution."
        )

    proposal.to(device)
    optimizer_options = {} if optimizer_options is None else optimizer_options
    stop_options = {} if stop_options is None else stop_options

    clip_gradients(proposal)
    optim = optimizer(proposal.parameters(), **optimizer_options)  # type: ignore
    stopping_criterion = SEPlateauStopping(**stop_options)  # type: ignore

    vloss = torch.zeros(1)
    gen_samples = 0
    if augmenter is not None:
        if X0 is not None:
            raise ValueError(
                "Data augmenter does not work with transition data."
            )
        augmenter.fit(X)
        gen_samples = max(1, round(batch_size * augmentation_p))

    if val_proportion > 0:
        batch_gen = batch_indices_val(len(X), batch_size, seed, val_proportion)
    else:
        batch_gen = ((i, None) for i in batch_indices(len(X), batch_size, seed))

    proposal.train()
    for i, (bi, oobi) in enumerate(batch_gen):
        Xb = X[bi]
        if gen_samples > 0 and augmenter is not None:  # Data augmentation
            Xa = augmenter.generate(gen_samples)
            Xb = torch.vstack((Xb, Xa))
        Xb = Xb.to(device)

        # Log likelihood
        if istransition:
            proposal.set_seeds(X0[bi].to(device))
        loss = -proposal.log_prob(Xb).mean()
        loss.backward()
        optim.step()
        optim.zero_grad()

        # Cross validation
        sloss = loss.detach()
        if oobi is not None:
            proposal.eval()
            if istransition:
                proposal.set_seeds(X0[oobi].to(device))
            with torch.no_grad():
                vloss = -proposal.log_prob(X[oobi].to(device)).mean().detach()
                sloss = vloss
            proposal.train()

        if callback is not None:
            callback(i, loss, vloss)
        if stopping_criterion.evaluate(fvals=sloss):
            break

    # Reset transition sequences
    if istransition:
        proposal.clear_seeds()
    proposal.eval()


#
# Utils
#


def clip_gradients(proposal_distribution: SearchDistribution):
    """Register per-parameter gradient clamp hooks once per module.

    Parameters
    ----------
    proposal_distribution : SearchDistribution
        Proposal whose parameters will have gradient clamps applied.

    Notes
    -----
    The hooks are set at most once per module to avoid accumulating duplicates
    across repeated training loops.
    """
    cg = proposal_distribution.clip_gradients
    if cg is None:
        return
    # Guard against registering hooks multiple times
    if getattr(proposal_distribution, "_grad_clip_hooks_set", False):
        return
    for p in proposal_distribution.parameters():
        p.register_hook(lambda grad, c=float(cg): torch.clamp(grad, -c, c))
    setattr(proposal_distribution, "_grad_clip_hooks_set", True)


#
# Private utils
#


def _rinit(shape: torch.Size | Sequence | int, features: int) -> Tensor:
    """Uniform random initialisation symmetric around zero.

    Parameters
    ----------
    shape : torch.Size | Sequence | int
        Desired tensor shape.
    features : int
        Feature count used to set the scaling factor ``1 / sqrt(features)``.

    Returns
    -------
    Tensor
        Random tensor in ``[-scale, scale]`` where ``scale = features**-0.5``.
    """
    scale = features ** (-0.5)
    init = torch.rand(shape) * 2 * scale - scale
    return init
