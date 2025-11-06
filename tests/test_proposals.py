import pytest
import torch

from vsd.proposals import (
    GaussianKDEProposal,
    MultiCategoricalProposal,
    SequenceUninformativePrior,
    LSTMProposal,
    DTransformerProposal,
    TransformerMLMProposal,
    TransformerMutationProposal,
)

from conftest import assert_close


def test_gaussian_kde():
    q = GaussianKDEProposal(d_features=3, k_components=5, scale=0.5)
    with q.record_sample_log_prob():
        Xs, logqX = q(samples=7)
    assert Xs.shape == (7, 3)
    assert logqX.shape == (7,)
    assert_close(logqX, q._last_sample_log_prob)


def test_multicategorical():
    q = MultiCategoricalProposal(
        d_features=6, k_categories=8, uniform_init=True
    )
    with q.record_sample_log_prob():
        Xs, logqX = q(samples=10)
    assert Xs.shape == (10, 6)
    assert logqX.shape == (10,)
    assert_close(logqX, q._last_sample_log_prob)


def test_uniform_prior():
    q = SequenceUninformativePrior(d_features=4, k_categories=3)
    with q.record_sample_log_prob():
        Xs, logqX = q(samples=6)
    assert Xs.shape == (6, 4)
    assert logqX.shape == (6,)
    assert_close(logqX, q._last_sample_log_prob)


def test_lstm_autoregressive():
    q = LSTMProposal(d_features=5, k_categories=7, dropout=0.0)
    q.eval()
    with q.record_sample_log_prob():
        Xs, logqX = q(samples=4)
    assert Xs.shape == (4, 5)
    assert logqX.shape == (4,)
    assert_close(logqX, q._last_sample_log_prob)


def test_dtransformer_autoregressive():
    q = DTransformerProposal(
        d_features=4,
        k_categories=6,
        dropout=0.0,
        num_layers=1,
        nhead=2,
    )
    q.eval()
    with q.record_sample_log_prob():
        Xs, logqX = q(samples=3)
    assert Xs.shape == (3, 4)
    assert logqX.shape == (3,)
    assert_close(logqX, q._last_sample_log_prob)


@pytest.mark.parametrize("k, pad_token", [[4, None], [6, 5], [5, 2]])
def test_transformer_mlm(k, pad_token):
    d = 6
    X0 = torch.randint(0, k, (2, d))
    q = TransformerMLMProposal(
        d_features=d,
        k_categories=k,
        X0=X0,
        gibbs_steps=1,
        pad_token=pad_token,
    )
    q.eval()
    with q.record_sample_log_prob():
        Xs, logqX = q(samples=10)
    assert Xs.shape == (10, d)
    assert logqX.shape == (10,)
    assert_close(logqX, q._last_sample_log_prob)


@pytest.mark.parametrize("k, pad_token", [[4, None], [6, 5], [5, 2]])
def test_transformer_mutation(k, pad_token):
    d = 7
    X0 = torch.randint(0, k, (2, d))
    q = TransformerMutationProposal(
        d_features=d,
        k_categories=k,
        X0=X0,
        num_mutations=2,
        pad_token=pad_token,
        replacement=False,  # need this for the test to pass
    )
    q.eval()
    with q.record_sample_log_prob():
        Xs, logqX = q(samples=10)
    assert Xs.shape == (10, d)
    assert logqX.shape == (10,)
    assert_close(logqX, q._last_sample_log_prob)
