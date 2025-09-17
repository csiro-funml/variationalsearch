"""Preference direction distributions.

Simple priors over unit vectors representing user preferences or directions.
"""

from typing import Optional

import torch
import torch.nn.functional as fnn
import torch.distributions as td
import zuko
from torch import Tensor

from vsd.proposals import SearchDistribution
from vsd.utils import inv_softplus

#
# Preference distributions (on spheres)
#


class PreferenceDistribution(SearchDistribution):
    """Abstract base for distributions over preference vectors ``U``."""


class EmpiricalPreferences(PreferenceDistribution):
    """Empirical (resampling) distribution over a fixed set of preferences."""

    def __init__(
        self,
        U: Optional[Tensor] = None,
        samples: int = 100,
        clip_gradients: Optional[float] = None,
    ):
        super().__init__(samples, clip_gradients)
        self.register_buffer("U", U)
        if U is not None:
            self.set_preferences(U)

    @torch.no_grad()
    def sample(self, sample_shape: torch.Size = torch.Size([1])) -> Tensor:
        if self.U is None:
            raise ValueError("Preferences, U, must be set.")
        idx = torch.randint(low=0, high=self.n, size=sample_shape)
        return self.U[idx]

    def log_prob(self, X):
        raise NotImplementedError(
            "Log probability not valid for empirical distribution."
        )

    def set_preferences(self, U: Tensor):
        self.n = len(U)
        self.U = U


class UnitNormal(PreferenceDistribution):
    """Isotropic-normal projected to the unit sphere (2+ dims)."""

    def __init__(
        self,
        dim: int,
        loc: Optional[Tensor] = None,
        scale: Optional[Tensor] = None,
        samples: int = 100,
        min_scale: float = 1e-6,
        eps: float = 1e-6,
        clip_gradients: Optional[float] = None,
    ):
        super().__init__(samples=samples, clip_gradients=clip_gradients)
        self.dim = dim
        self.min_scale = min_scale
        if loc is None:
            loc = fnn.normalize(torch.randn(dim), p=2, dim=-1, eps=eps)
        self.loc = torch.nn.Parameter(loc.float())
        self.scale = torch.nn.Parameter(inv_softplus(scale or torch.ones(dim)))
        self.eps = eps

    @torch.no_grad()
    def sample(self, sample_shape=torch.Size([1])):
        norm = self._make_normal()
        Us = fnn.normalize(norm.sample(sample_shape), p=2, dim=-1, eps=self.eps)
        return Us

    def log_prob(self, U: Tensor) -> Tensor:
        U = fnn.normalize(U, p=2, dim=-1, eps=self.eps)
        norm = self._make_normal()
        logp = norm.log_prob(U)
        # constrain loc to be close to unit norm
        reward = logp - (torch.linalg.vector_norm(self.loc, ord=2) - 1) ** 2
        return reward

    def _make_normal(self):
        scale = torch.clamp(fnn.softplus(self.scale), min=self.min_scale)
        norm = td.Independent(td.Normal(loc=self.loc, scale=scale), 1)
        return norm


class MixtureUnitNormal(PreferenceDistribution):
    """Mixture of unit Normal distributions for directional preferences."""

    def __init__(
        self,
        locs: Tensor,
        scales: Optional[Tensor] = None,
        weights: Optional[Tensor] = None,
        samples: int = 100,
        min_scale: float = 1e-6,
        eps: float = 1e-6,
        clip_gradients: Optional[float] = None,
    ):
        super().__init__(samples=samples, clip_gradients=clip_gradients)
        self.eps = eps
        self.min_scale = min_scale

        self.K = locs.shape[0]
        self.locs = torch.nn.Parameter(
            fnn.normalize(locs.float(), p=2, dim=-1, eps=eps)
        )

        if scales is None:
            scales = torch.ones_like(locs)
        self.scales = torch.nn.Parameter(inv_softplus(scales))

        # mixture weights
        if weights is None:
            weights = torch.ones(self.K) / self.K
        elif weights.shape[0] != self.K:
            raise ValueError(
                f"weights must have shape ({self.K},), got {weights.shape}"
            )
        self.logits = torch.nn.Parameter(torch.log(weights.float()))

    @torch.no_grad()
    def sample(self, sample_shape: torch.Size = torch.Size([1])) -> Tensor:
        mix = self._buildmix()
        Us = fnn.normalize(mix.sample(sample_shape), p=2, dim=-1, eps=self.eps)
        return Us

    def log_prob(self, U: Tensor) -> Tensor:
        # ensure unit vectors
        U = fnn.normalize(U, p=2, dim=-1, eps=self.eps)
        mix = self._buildmix()
        logp = mix.log_prob(U)
        # constrain loc to be close to unit norm
        normp = (fnn.normalize(self.locs, p=2, dim=-1, eps=self.eps) - 1) ** 2
        reward = logp - normp.mean()
        return reward

    def _buildmix(self) -> td.Distribution:
        scales = torch.clamp(fnn.softplus(self.scales), min=self.min_scale)
        comps = td.Independent(
            td.Normal(
                loc=self.locs,
                scale=scales,
            ),
            1,
        )
        mix = td.MixtureSameFamily(
            td.Categorical(logits=self.logits),
            comps,
        )
        return mix


class UnitVonMises(PreferenceDistribution):
    """Wrapped von Mises on S1 (2D unit circle)."""

    def __init__(
        self,
        loc: Optional[Tensor] = None,
        concentration: Optional[Tensor] = None,
        samples: int = 100,
        clip_gradients: Optional[float] = None,
    ):
        super().__init__(samples=samples, clip_gradients=clip_gradients)
        if loc is None:
            loc = torch.ones((1, 2)) / torch.sqrt(torch.tensor(2))
        elif loc.shape[0] != 2:
            raise ValueError("Von Mises only defined on 2-dimensional U!")
        self.loc = torch.nn.Parameter(loc.float())
        self.conc = torch.nn.Parameter(
            inv_softplus(concentration or torch.zeros(1) + 1e-2)
        )

    @torch.no_grad()
    def sample(self, sample_shape=torch.Size([1])):
        vm = td.VonMises(
            loc=_radians(self.loc), concentration=fnn.softplus(self.conc)
        )
        rs = vm.sample(sample_shape=sample_shape)
        return _cartesian(rs)

    def log_prob(self, U):
        if U.shape[1] != 2:
            raise ValueError("Von Mises only defined on 2-dimensional U!")
        r = _radians(U)
        vm = td.VonMises(
            loc=_radians(self.loc), concentration=fnn.softplus(self.conc)
        )
        logpU = vm.log_prob(r)
        return logpU


class SphericalPreferenceFlow(PreferenceDistribution):
    """Spherical Normalizing Flows.

    Rezende, D.J., Papamakarios, G., Racaniere, S., Albergo, M., Kanwar, G.,
    Shanahan, P. and Cranmer, K., 2020, November. Normalizing flows on tori and
    spheres. In International Conference on Machine Learning (pp. 8083-8092).
    PMLR.

    https://zuko.readthedocs.io/stable/api/zuko.flows.spline.html#zuko.flows.spline.NCSF
    """

    def __init__(
        self,
        dim: int,
        hidden_features: int = 64,
        num_layers: int = 5,
    ):
        super().__init__()
        self.dim = dim

        # Build unconditional NSF (context = 0)
        self.flow = zuko.flows.NSF(
            features=dim,
            context=0,
            transforms=num_layers,
            hidden_features=[hidden_features] * num_layers,
        )

    @torch.no_grad()
    def sample(
        self, sample_shape: torch.Size = torch.Size([1])
    ) -> torch.Tensor:
        samples = self.flow(None).sample(sample_shape)
        return fnn.normalize(samples, p=2, dim=-1)

    def log_prob(self, U: torch.Tensor) -> torch.Tensor:
        U = fnn.normalize(U, p=2, dim=-1)
        return self.flow(None).log_prob(U)


def _radians(cartesian: Tensor) -> Tensor:
    if cartesian.shape[1] > 2:
        raise ValueError("Can only convert x.shape[1] == 2 to radians!")
    r = torch.atan2(cartesian[:, 1], cartesian[:, 0]).float()
    return r


def _cartesian(radians: Tensor) -> Tensor:
    x = torch.hstack([torch.cos(radians), torch.sin(radians)]).float()
    return x
