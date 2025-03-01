"""Data augmentation generators."""

from abc import ABC, abstractmethod

import numpy as np
import torch
from torch import Tensor


class AugementGenerator(ABC):

    def __init__(self): ...

    @abstractmethod
    def fit(self, X: Tensor): ...

    @abstractmethod
    def generate(self, samples: int) -> Tensor: ...


class TransitionAugmenter(AugementGenerator):

    def __init__(self, max_mutations: int = 5):
        super().__init__()
        self.max_mutations = max_mutations

    def fit(self, X):
        self.X = X
        self.n, self.d = X.shape
        self.A = fit_transition_matrix_from_int_matrix(X)
        self.pA = [torch.distributions.Categorical(probs=a) for a in self.A]

    def generate(self, samples: int) -> Tensor:

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
    """
    Fits a transition matrix from (integer) sequential data

    Parameters:
    - sequences: 2D tensor where each row is a sequence of integer states.

    Returns:
    - transition_matrix: A 2D tensor containing transition probabilities.
    """
    sequences = sequences.detach().numpy()

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
