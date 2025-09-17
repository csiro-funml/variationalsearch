"""Data augmentation generators.

Lightweight interfaces to fit simple data-driven augmenters and to draw
augmented samples for training variational models.
"""

from abc import ABC, abstractmethod

import numpy as np
import torch
from torch import Tensor


class AugementGenerator(ABC):
    """Abstract base for augmenters.

    Methods
    -------
    fit(X):
        Learn any statistics from data ``X``.
    generate(samples):
        Produce ``samples`` augmented examples with the same dtype/shape logic
        as the fitted data.
    """

    def __init__(self): ...

    @abstractmethod
    def fit(self, X: Tensor):
        """Fit the augmenter on dataset ``X``."""

    @abstractmethod
    def generate(self, samples: int) -> Tensor:
        """Generate ``samples`` augmented items."""


class TransitionAugmenter(AugementGenerator):
    """Sequence augmenter using an empirical first-order transition model.

    Parameters
    ----------
    max_mutations : int, default=5
        Maximum number of site mutations applied per generated sequence.
    """

    def __init__(self, max_mutations: int = 5):
        super().__init__()
        self.max_mutations = max_mutations

    def fit(self, X):
        """Estimate transition probabilities from integer-encoded sequences."""
        self.device = X.device
        self.X = X
        self.n, self.d = X.shape
        self.A = fit_transition_matrix_from_int_matrix(X).to(self.device)
        self.pA = [torch.distributions.Categorical(probs=a) for a in self.A]

    def generate(self, samples: int) -> Tensor:
        """Draw augmented sequences by sampling local transitions."""

        # Get samples from the training data with replacement
        samples = torch.Size([samples])
        Xs = self.X[torch.randint(low=0, high=self.n, size=samples)]

        # Randomly mutate each sequence based on transition probabilities
        for x in Xs:
            nmut = torch.randint(low=1, high=self.max_mutations, size=[1])
            site = torch.randint(low=1, high=self.d, size=torch.Size(nmut))
            x[site] = torch.hstack([self.pA[j].sample() for j in x[site - 1]])
        return Xs


def fit_transition_matrix_from_int_matrix(sequences: Tensor) -> Tensor:
    """Estimate a first-order transition matrix from integer sequences.

    Parameters
    ----------
    sequences : Tensor
        Tensor of shape ``(N, L)`` where rows are integer states ``[0..K-1]``.

    Returns
    -------
    Tensor
        Row-stochastic transition matrix of shape ``(K, K)`` where entry
        ``[i, j]`` is the probability of transitioning ``i → j``.
    """
    sequences = sequences.detach().cpu().numpy()

    # Extract the number of states
    n_states = sequences.max() + 1  # Assumes states are 0-indexed integers

    # Flatten and extract transition pairs
    from_states = sequences[:, :-1].ravel()
    to_states = sequences[:, 1:].ravel()

    # Count transitions
    count_matrix = np.zeros((n_states, n_states), dtype=np.int64)
    np.add.at(count_matrix, (from_states, to_states), 1)

    # Normalize rows to obtain probabilities
    row_sums = count_matrix.sum(axis=1, keepdims=True)
    transition_matrix = np.divide(
        count_matrix,
        row_sums,
        out=np.zeros_like(count_matrix, dtype=float),
        where=row_sums != 0,
    )

    return torch.tensor(transition_matrix, dtype=torch.float32)
