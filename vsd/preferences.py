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
    """Empirical (resampling) distribution over a fixed set of preferences.

    Parameters
    ----------
    U : Tensor, optional
        Preference vectors stored row-wise. Must be set before sampling.
    samples : int, default=100
        Default number of samples returned by ``forward``.
    clip_gradients : float, optional
        Gradient clipping value applied during optimisation.
    """

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
        """Resample preferences uniformly from the stored bank."""
        if self.U is None:
            raise ValueError("Preferences, U, must be set.")
        idx = torch.randint(low=0, high=self.n, size=sample_shape)
        return self.U[idx]

    def log_prob(self, X):
        """Log-density is undefined for purely empirical resampling."""
        raise NotImplementedError(
            "Log probability not valid for empirical distribution."
        )

    def set_preferences(self, U: Tensor):
        """Register the preference bank used during resampling."""
        self.n = len(U)
        self.U = U


class UnitNormal(PreferenceDistribution):
    """Isotropic normal projected to the unit sphere (2+ dims).

    Parameters
    ----------
    dim : int
        Dimensionality of the preference vectors.
    loc : Tensor, optional
        Mean direction before projection; defaults to a random unit vector.
    scale : Tensor, optional
        Standard deviation per dimension before projection. Defaults to ones.
    samples : int, default=100
        Default number of samples produced by ``forward``.
    min_scale : float, default=1e-6
        Lower bound applied to the standard deviation.
    eps : float, default=1e-6
        Epsilon used for stable normalisation.
    clip_gradients : float, optional
        Gradient clipping value applied during optimisation.
    """

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
        """Draw unit-norm samples by projecting a Gaussian to the sphere."""
        norm = self._make_normal()
        Us = fnn.normalize(norm.sample(sample_shape), p=2, dim=-1, eps=self.eps)
        return Us

    def log_prob(self, U: Tensor) -> Tensor:
        """Compute a regularised log-score for unit vectors.

        Notes
        -----
        Returns ``log p(U)`` under the projected normal minus a quadratic
        penalty encouraging the mean direction to remain unit-normalised.
        """
        U = fnn.normalize(U, p=2, dim=-1, eps=self.eps)
        norm = self._make_normal()
        logp = norm.log_prob(U)
        # constrain loc to be close to unit norm
        reward = logp - (torch.linalg.vector_norm(self.loc, ord=2) - 1) ** 2
        return reward

    def _make_normal(self):
        """Return the underlying multivariate normal prior to projection."""
        scale = torch.clamp(fnn.softplus(self.scale), min=self.min_scale)
        norm = td.Independent(td.Normal(loc=self.loc, scale=scale), 1)
        return norm


class MixtureUnitNormal(PreferenceDistribution):
    """Mixture of projected Normal distributions for directional preferences.

    Parameters
    ----------
    locs : Tensor
        Component means with shape ``(K, D)``.
    scales : Tensor, optional
        Component scales matching ``locs``; defaults to ones.
    weights : Tensor, optional
        Mixture weights of shape ``(K,)``; defaults to uniform.
    samples : int, default=100
        Default number of samples produced by ``forward``.
    min_scale : float, default=1e-6
        Lower bound applied to standard deviations.
    eps : float, default=1e-6
        Epsilon used for stable normalisation.
    clip_gradients : float, optional
        Gradient clipping value applied during optimisation.
    """

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
        """Sample unit vectors by mixing projected Gaussians."""
        mix = self._buildmix()
        Us = fnn.normalize(mix.sample(sample_shape), p=2, dim=-1, eps=self.eps)
        return Us

    def log_prob(self, U: Tensor) -> Tensor:
        """Evaluate the regularised log-score of unit vectors under the mixture.

        Notes
        -----
        Returns ``log p(U)`` under the projected mixture minus a penalty that
        encourages component means to remain unit-normalised.
        """
        # ensure unit vectors
        U = fnn.normalize(U, p=2, dim=-1, eps=self.eps)
        mix = self._buildmix()
        logp = mix.log_prob(U)
        # constrain loc to be close to unit norm
        normp = (fnn.normalize(self.locs, p=2, dim=-1, eps=self.eps) - 1) ** 2
        reward = logp - normp.mean()
        return reward

    def _buildmix(self) -> td.Distribution:
        """Construct the underlying mixture before projecting to the sphere."""
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
    """Wrapped von Mises distribution on the 2D unit circle.

    Parameters
    ----------
    loc : Tensor, optional
        Mean direction on the unit circle. Defaults to ``[1/√2, 1/√2]``.
    concentration : Tensor, optional
        Concentration parameter ``κ``. Defaults to ``1e-2``.
    samples : int, default=100
        Default number of samples returned by ``forward``.
    clip_gradients : float, optional
        Gradient clipping value applied during optimisation.
    """

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
        """Draw samples on the unit circle via a von Mises distribution."""
        vm = td.VonMises(
            loc=_radians(self.loc), concentration=fnn.softplus(self.conc)
        )
        rs = vm.sample(sample_shape=sample_shape)
        return _cartesian(rs)

    def log_prob(self, U):
        """Evaluate the log-density of unit-circle points under the von Mises."""
        if U.shape[1] != 2:
            raise ValueError("Von Mises only defined on 2-dimensional U!")
        r = _radians(U)
        vm = td.VonMises(
            loc=_radians(self.loc), concentration=fnn.softplus(self.conc)
        )
        logpU = vm.log_prob(r)
        return logpU


class SphericalPreferenceFlow(PreferenceDistribution):
    """Spherical normalising flow preference model.

    Implements the NCSF spherical flow of Rezende et al. (2020).
    See https://zuko.readthedocs.io/stable/api/zuko.flows.spline.html

    Parameters
    ----------
    dim : int
        Dimensionality of the preference vectors.
    hidden_features : int, default=64
        Width of each hidden layer in the flow network.
    num_layers : int, default=5
        Number of autoregressive transforms.
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
        """Sample unit vectors from the learnt spherical flow."""
        samples = self.flow(None).sample(sample_shape)
        return fnn.normalize(samples, p=2, dim=-1)

    def log_prob(self, U: torch.Tensor) -> torch.Tensor:
        """Evaluate the log-density of unit vectors under the flow."""
        U = fnn.normalize(U, p=2, dim=-1)
        return self.flow(None).log_prob(U)


def _radians(cartesian: Tensor) -> Tensor:
    """Convert 2D Cartesian coordinates to angles in radians."""
    if cartesian.shape[1] > 2:
        raise ValueError("Can only convert x.shape[1] == 2 to radians!")
    r = torch.atan2(cartesian[:, 1], cartesian[:, 0]).float()
    return r


def _cartesian(radians: Tensor) -> Tensor:
    """Map radians on the unit circle back to 2D Cartesian coordinates."""
    x = torch.hstack([torch.cos(radians), torch.sin(radians)]).float()
    return x
