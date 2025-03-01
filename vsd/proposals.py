"""Variational distributions and priors."""

from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, Optional, Tuple

import torch
import torch.distributions as td
from botorch.optim.stopping import ExpMAStoppingCriterion
from torch import Tensor
from torch.optim import Optimizer

from vsd.augmentation import AugementGenerator
from vsd.utils import (
    PositionalEncoding,
    RemovePadding,
    SequenceTensor,
    Transpose,
    batch_indices,
)


class SearchDistribution(ABC, torch.nn.Module):
    """Abstract base class for variational distributions."""

    def __init__(
        self, samples: int = 100, clip_gradients: Optional[float] = None
    ) -> None:
        super().__init__()
        self.samples = samples
        self.clip_gradients = clip_gradients

    def forward(self, samples: Optional[int] = None) -> Tuple[Tensor, Tensor]:
        samples = self.samples if samples is None else samples
        with torch.no_grad():
            Xs = self.sample(torch.Size([samples]))
        logqX = self.log_prob(Xs)
        return Xs, logqX

    @abstractmethod
    def sample(self, sample_shape: torch.Size = torch.Size([1])) -> Tensor:
        pass

    @abstractmethod
    def log_prob(self, X: Tensor) -> Tensor:
        pass


class SimpleSearchDistribution(SearchDistribution):
    """A simplified interface for variational distributions."""

    def forward(self, samples: Optional[int] = None) -> Tuple[Tensor, Tensor]:
        samples = self.samples if samples is None else samples
        q = self._construct_q()
        with torch.no_grad():
            Xs = q.sample(torch.Size([samples]))
        logqX = q.log_prob(Xs)
        return Xs, logqX

    def log_prob(self, X: Tensor) -> Tensor:
        return self._construct_q().log_prob(X)

    def sample(self, sample_shape: torch.Size = torch.Size([1])) -> Tensor:
        return self._construct_q().sample(sample_shape)

    @abstractmethod
    def _construct_q(self) -> td.Distribution:
        pass


#
# Continuous data
#


class GaussianProposal(SimpleSearchDistribution):
    """Single Gaussian search distribution."""

    def __init__(self, dims, scale=1, samples: int = 100) -> None:
        super().__init__(samples)
        self.mu = torch.nn.Parameter(torch.randn(dims))
        logscale = torch.log(Tensor((scale,)))
        self.std = torch.nn.Parameter(torch.ones(dims) * logscale)

    def _construct_q(self) -> td.Distribution:
        norm = td.Independent(
            td.Normal(loc=self.mu, scale=torch.exp(self.std)), 1
        )
        return norm


class GaussianKDEProposal(SimpleSearchDistribution):
    """Gaussian KDE/mixture search distribution."""

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
    """Abstract base for search distributions over sequences."""

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
    """Uniform prior over sequences -- no learnable parameters."""

    def __init__(self, d_features: int, k_categories: int):
        super().__init__(d_features=d_features, k_categories=k_categories)
        self.register_buffer("l", torch.zeros(self.d, self.k))

    def _construct_q(self):
        return td.Independent(td.Categorical(logits=self.l), 1)


class MultiCategoricalProposal(SequenceSearchDistribution):
    """Independent/Mean field multi-categorical search distribution."""

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
            logits = _rinit((self.d, self.k), self.k)
        self.phi = torch.nn.Parameter(logits)

    def _construct_q(self) -> td.Distribution:
        cat = td.Independent(td.Categorical(logits=self.phi), 1)
        return cat


#
# Sequence data -- auto-regressive
#


class AutoRegressiveSearchDistribution(SearchDistribution):
    """Abstract base for autoregressive search distributions over sequences."""

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


class LSTMProposal(AutoRegressiveSearchDistribution):
    """Long short-term memory RNN search distribution."""

    def __init__(
        self,
        d_features: int,
        k_categories: int,
        embedding_dim: Optional[int] = None,
        hidden_size: Optional[int] = None,
        num_layers: int = 1,
        samples: int = 100,
        clip_gradients: Optional[float] = None,
    ) -> None:
        super().__init__(
            d_features=d_features,
            k_categories=k_categories,
            samples=samples,
            clip_gradients=clip_gradients,
        )
        if embedding_dim is None:
            embedding_dim = max(2, self.k // 2)
        self.e = embedding_dim
        if hidden_size is None:
            hidden_size = 8 * embedding_dim
        self.e0 = torch.nn.Parameter(_rinit(embedding_dim, embedding_dim))
        self.pos = PositionalEncoding(emb_size=self.e)
        self.emb = torch.nn.Embedding(
            num_embeddings=self.k, embedding_dim=self.e
        )
        self.lstm = torch.nn.LSTM(
            input_size=self.e,
            hidden_size=hidden_size,
            batch_first=True,
            num_layers=num_layers,
        )
        self.dec = torch.nn.Linear(in_features=hidden_size, out_features=self.k)

    def sample(self, sample_shape: torch.Size = torch.Size([1])) -> Tensor:
        device = self.e0.device
        Xs = torch.zeros(
            sample_shape + torch.Size([self.d]), device=device, dtype=torch.long
        )
        e, hc = self.pos(self.e0.tile(sample_shape + torch.Size([1, 1]))), None
        for i in range(self.d):
            o, hc = self.lstm(e, hc)
            q = td.Categorical(logits=self.dec(o))
            xs = q.sample()
            Xs[:, i] = xs.squeeze()
            e = self.pos(self.emb(xs), pos_ind=i + 1)
        return Xs

    def log_prob(self, X: Tensor) -> Tensor:
        n = len(X)
        e = self.pos(torch.hstack((self.e0.tile(n, 1, 1), self.emb(X[:, :-1]))))
        o, _ = self.lstm(e, None)
        q = td.Independent(td.Categorical(logits=self.dec(o)), 1)
        return q.log_prob(X)


class DTransformerProposal(AutoRegressiveSearchDistribution):
    """Causal decoder-only transformer search distribution."""

    def __init__(
        self,
        d_features: int,
        k_categories: int,
        embedding_dim: Optional[int] = None,
        nhead: Optional[int] = 2,
        dim_feedforward: Optional[int] = 128,
        dropout: Optional[float] = 0.0,
        num_layers: int = 1,
        samples: int = 100,
        clip_gradients: Optional[float] = None,
    ) -> None:
        super().__init__(
            d_features=d_features,
            k_categories=k_categories,
            samples=samples,
            clip_gradients=clip_gradients,
        )
        if embedding_dim is None:
            embedding_dim = max(2, self.k // 2) * nhead
        self.e = embedding_dim
        self.e0 = torch.nn.Parameter(_rinit(embedding_dim, embedding_dim))
        self.emb = torch.nn.Embedding(
            num_embeddings=self.k, embedding_dim=self.e
        )
        self.pos = PositionalEncoding(emb_size=self.e)
        dtlayer = torch.nn.TransformerDecoderLayer(
            d_model=self.e,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.dtfm = torch.nn.TransformerDecoder(
            decoder_layer=dtlayer, num_layers=num_layers
        )
        self.dec = torch.nn.Linear(in_features=self.e, out_features=self.k)

    def log_prob(self, X: Tensor) -> Tensor:
        n = len(X)
        e = self.pos(torch.hstack((self.e0.tile(n, 1, 1), self.emb(X[:, :-1]))))
        mask = torch.nn.Transformer.generate_square_subsequent_mask(n)
        o = self.dtfm(
            tgt=e,
            memory=e,
            tgt_mask=mask,
            tgt_is_causal=True,
            memory_mask=mask,
            memory_is_causal=True,
        )
        q = td.Independent(td.Categorical(logits=self.dec(o)), 1)
        return q.log_prob(X)

    def sample(self, sample_shape: torch.Size = torch.Size([1])) -> Tensor:
        device = self.e0.device
        e = torch.zeros(
            sample_shape + torch.Size([self.d + 1, self.e]), device=device
        )
        e[:, 0] = self.e0
        e = self.pos(e)
        Xs = torch.zeros(
            sample_shape + torch.Size([self.d]), device=device, dtype=torch.long
        )
        mask = torch.nn.Transformer.generate_square_subsequent_mask(self.d)
        for i, j in zip(range(self.d), range(1, self.d + 1)):
            o = self.dtfm(
                tgt=e[:, :j],
                memory=e[:, :j],
                tgt_mask=mask[:j, :j],
                tgt_is_causal=True,
                memory_mask=mask[:j, :j],
                memory_is_causal=True,
            )[:, i]
            q = td.Categorical(logits=self.dec(o))
            xs = q.sample()
            Xs[:, i] = xs
            e[:, j] = e[:, j] + self.emb(xs)
        return Xs


#
# Sequence data -- transition
#


class TransitionSearchDistribution(SequenceSearchDistribution):
    """Abstract base class for a transition search distribution."""

    def __init__(
        self,
        d_features: int,
        k_categories: int,
        samples: int = 100,
        clip_gradients: Optional[float] = None,
    ):
        super().__init__(
            d_features=d_features,
            k_categories=k_categories,
            samples=samples,
            clip_gradients=clip_gradients,
        )
        self.X0 = None

    def forward(
        self,
        samples: Optional[int] = None,
        X0: Optional[Tensor] = None,
        return_X0: bool = False,
    ) -> Tuple[Tensor, Tensor]:
        samples = self.samples if samples is None else samples
        if (self.X0 is None) and (X0 is None):
            raise ValueError("Input X0 required, or call this.update(X0).")
        X0 = self.X0 if X0 is None else X0
        q = self._construct_q(X0)
        Xs = q.sample([samples])
        logqX = q.log_prob(Xs)
        Xs, logqX = Xs.reshape(-1, X0.shape[-1]), logqX.flatten()

        if return_X0:
            return Xs, logqX, X0.tile((samples, 1))
        return Xs, logqX

    def log_prob(self, X: Tensor, X0: Tensor) -> Tensor:
        return self._construct_q(X0).log_prob(X)

    def sample(
        self, X0: Tensor, sample_shape: torch.Size = torch.Size([1])
    ) -> Tensor:
        xs = self._construct_q(X0).sample(sample_shape)
        return xs.reshape(-1, X0.shape[-1])

    def update(self, X0: Tensor):
        self.X0 = X0

    def get_X0(self) -> Tensor:
        return self.X0

    @abstractmethod
    def _construct_q(self, X0: Tensor) -> td.Distribution:
        pass


class _NNTransitionProposal(TransitionSearchDistribution):
    """NN Transition proposal base class."""

    def __init__(
        self,
        d_features: int,
        k_categories: int,
        embedding_dim: Optional[int] = None,
        samples: int = 100,
        clip_gradients: Optional[float] = None,
    ):
        super().__init__(
            d_features=d_features,
            k_categories=k_categories,
            samples=samples,
            clip_gradients=clip_gradients,
        )
        if embedding_dim is None:
            embedding_dim = max(2, self.k // 2)
        self.e = embedding_dim
        self.nn = None

    def _construct_q(self, X0: SequenceTensor) -> td.Distribution:
        logits = self.nn(X0)
        cat = td.Independent(td.Categorical(logits=logits), 1)
        return cat


class _CNNETransitionProposal(_NNTransitionProposal):
    """CNN encoder transition proposal base class."""

    def __init__(
        self,
        d_features: int,
        k_categories: int,
        embedding_dim: Optional[int] = None,
        latent_k: Optional[int] = None,
        kernel_size: int = 5,
        samples: int = 100,
        clip_gradients: Optional[float] = None,
    ):
        super().__init__(
            d_features=d_features,
            k_categories=k_categories,
            embedding_dim=embedding_dim,
            samples=samples,
            clip_gradients=clip_gradients,
        )
        self.lk = latent_k if latent_k is not None else self.k * 2
        self.padding = kernel_size // 2
        self.enc = torch.nn.Sequential(
            torch.nn.Embedding(num_embeddings=self.k, embedding_dim=self.e),
            Transpose(),
            torch.nn.Conv1d(
                in_channels=self.e,
                out_channels=self.lk,
                kernel_size=kernel_size,
                padding=self.padding,
            ),
            Transpose(),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(self.lk, self.lk),
            Transpose(),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(self.d, self.d),
            torch.nn.LeakyReLU(),
            Transpose(),
        )
        self.dec = None

    def _construct_q(self, X0: SequenceTensor) -> td.Distribution:
        Z = self.enc(X0)
        L = self.dec(Z)
        cat = td.Independent(td.Categorical(logits=L), 1)
        return cat


class TransitionCNNProposal(_CNNETransitionProposal):
    """Encoder-decoder style CNN search distribution."""

    def __init__(
        self,
        d_features: int,
        k_categories: int,
        embedding_dim: Optional[int] = None,
        latent_k: Optional[int] = None,
        kernel_size: int = 5,
        samples: int = 100,
        clip_gradients: Optional[float] = None,
    ) -> None:
        super().__init__(
            d_features=d_features,
            k_categories=k_categories,
            embedding_dim=embedding_dim,
            latent_k=latent_k,
            kernel_size=kernel_size,
            samples=samples,
            clip_gradients=clip_gradients,
        )
        self.dec = torch.nn.Sequential(
            torch.nn.Linear(self.lk, self.lk),
            torch.nn.LeakyReLU(),
            Transpose(),
            torch.nn.ConvTranspose1d(
                in_channels=self.lk,
                out_channels=self.k,
                kernel_size=kernel_size,
            ),
            RemovePadding(padding_size=self.padding),
            Transpose(),
        )


class TransitionCNNEProposal(TransitionCNNProposal):
    """Encoder only version of `TransitionCNNProposal`."""

    def __init__(
        self,
        d_features: int,
        k_categories: int,
        embedding_dim: Optional[int] = None,
        latent_k: Optional[int] = None,
        kernel_size: int = 5,
        samples: int = 100,
        clip_gradients: Optional[float] = None,
    ) -> None:
        super().__init__(
            d_features=d_features,
            k_categories=k_categories,
            embedding_dim=embedding_dim,
            latent_k=latent_k,
            kernel_size=kernel_size,
            samples=samples,
            clip_gradients=clip_gradients,
        )
        self.dec = torch.nn.Linear(self.lk, self.k)


class TransitionAEProposal(_NNTransitionProposal):
    """Auto-encoder style transition search distribution."""

    def __init__(
        self,
        d_features: int,
        k_categories: int,
        embedding_dim: Optional[int] = None,
        samples: int = 100,
        clip_gradients: Optional[float] = None,
    ) -> None:
        super().__init__(
            d_features=d_features,
            k_categories=k_categories,
            embedding_dim=embedding_dim,
            samples=samples,
            clip_gradients=clip_gradients,
        )
        self.nn = torch.nn.Sequential(
            torch.nn.Embedding(num_embeddings=self.k, embedding_dim=self.e),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(self.e, self.e),
            Transpose(),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(self.d, self.d),
            torch.nn.LeakyReLU(),
            Transpose(),
            torch.nn.Linear(self.e, self.e),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(self.e, self.k),
        )


class TransitionCNNDProposal(_NNTransitionProposal):
    """Decoder only version of `TransitionCNNProposal`."""

    def __init__(
        self,
        d_features: int,
        k_categories: int,
        embedding_dim: Optional[int] = None,
        kernel_size: int = 5,
        samples: int = 100,
        clip_gradients: Optional[float] = None,
    ) -> None:
        super().__init__(
            d_features=d_features,
            k_categories=k_categories,
            embedding_dim=embedding_dim,
            samples=samples,
            clip_gradients=clip_gradients,
        )
        padding = kernel_size // 2
        self.nn = torch.nn.Sequential(
            torch.nn.Embedding(num_embeddings=self.k, embedding_dim=self.e),
            torch.nn.LeakyReLU(),
            Transpose(),
            torch.nn.Linear(self.d, self.d),
            torch.nn.LeakyReLU(),
            Transpose(),
            torch.nn.Linear(self.e, self.e),
            Transpose(),
            torch.nn.ConvTranspose1d(
                in_channels=self.e,
                out_channels=self.k,
                kernel_size=kernel_size,
            ),
            RemovePadding(padding_size=padding),
            Transpose(),
        )


#
# Fitting routines
#


def fit_ml(
    proposal: SearchDistribution,
    X: Tensor,
    X_val: Optional[Tensor] = None,
    batch_size: int = 512,
    optimizer: Optimizer = torch.optim.AdamW,
    optimizer_options: Optional[Dict[str, Any]] = None,
    stop_options: Optional[Dict[str, Any]] = None,
    device: str = "cpu",
    callback: Optional[Callable[[int, Tensor, Tensor], None]] = None,
    seed: Optional[int] = None,
    stop_using_xval_loss: bool = False,
    augmenter: Optional[AugementGenerator] = None,
    augmentation_p: float = 0.1,
):
    """Fit a proposal distribution using ML."""
    if (X_val is None) and stop_using_xval_loss:
        raise ValueError("Need to specify X_val to stop_using_xval_loss")

    proposal.to(device)
    optimizer_options = {} if optimizer_options is None else optimizer_options
    stop_options = {} if stop_options is None else stop_options

    clip_gradients(proposal)
    optim = optimizer(proposal.parameters(), **optimizer_options)
    stopping_criterion = ExpMAStoppingCriterion(**stop_options)  # type: ignore

    if X_val is None:
        vloss = torch.zeros([])
    else:
        X_val = X_val.to(device)

    gen_samples = 0
    if augmenter is not None:
        augmenter.fit(X)
        gen_samples = max(1, round(batch_size * augmentation_p))

    proposal.train()
    for i, bi in enumerate(batch_indices(len(X), batch_size, seed)):
        Xb = X[bi]
        if gen_samples > 0:  # Data augmentation
            Xa = augmenter.generate(gen_samples)
            Xb = torch.vstack((Xb, Xa))
        Xb = Xb.to(device)

        # Log likelihood
        loss = -proposal.log_prob(Xb).mean()
        loss.backward()
        optim.step()
        optim.zero_grad()

        # Cross validation
        sloss = loss.detach()
        if X_val is not None:
            proposal.eval()
            with torch.no_grad():
                vloss = -proposal.log_prob(X_val).mean().detach()
            if stop_using_xval_loss:
                sloss = vloss
            proposal.train()

        if callback is not None:
            callback(i, loss, vloss)
        if stopping_criterion.evaluate(fvals=sloss):
            break

    proposal.eval()


#
# Utils
#


def clip_gradients(proposal_distribution: SearchDistribution):
    if proposal_distribution.clip_gradients is not None:
        for p in proposal_distribution.parameters():
            p.register_hook(
                lambda grad: torch.clamp(
                    grad,
                    -proposal_distribution.clip_gradients,
                    proposal_distribution.clip_gradients,
                )
            )


#
# Private utils
#


def _rinit(shape: torch.Size, features: int) -> Tensor:
    scale = features ** (-0.5)
    init = torch.rand(shape) * 2 * scale - scale
    return init
