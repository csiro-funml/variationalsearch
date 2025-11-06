import pytest
import torch

from vsd.condproposals import (
    ConditionalGaussianProposal,
    ConditionalGMMProposal,
    CondLSTMProposal,
    CondDTransformerProposal,
    CondTransformerMutationProposal,
)
from vsd.preferences import UnitNormal

from conftest import assert_close


def test_cond_gaussian():
    l = 3
    p = UnitNormal(dim=l)
    q = ConditionalGaussianProposal(x_dims=3, u_dims=l, latent_dim=8)
    U = p.sample(sample_shape=torch.Size([7]))
    with q.record_sample_log_prob():
        Xs, logqX = q(U)
    assert Xs.shape == (7, 3)
    assert logqX.shape == (7,)
    assert_close(logqX, q._last_sample_log_prob)


def test_cond_gmm():
    l = 2
    p = UnitNormal(dim=l)
    q = ConditionalGMMProposal(x_dims=3, u_dims=l, latent_dim=8)
    U = p.sample(sample_shape=torch.Size([9]))
    with q.record_sample_log_prob():
        Xs, logqX = q(U)
    assert Xs.shape == (9, 3)
    assert logqX.shape == (9,)
    assert_close(logqX, q._last_sample_log_prob)


def test_cond_lstm_autoregressive():
    l = 2
    p = UnitNormal(dim=l)
    q = CondLSTMProposal(d_features=5, k_categories=7, u_dims=l, dropout=0.0)
    U = p.sample(sample_shape=torch.Size([4]))
    q.eval()
    with q.record_sample_log_prob():
        Xs, logqX = q(U)
    assert Xs.shape == (4, 5)
    assert logqX.shape == (4,)
    assert_close(logqX, q._last_sample_log_prob)


def test_cond_dtransformer_autoregressive():
    l = 4
    p = UnitNormal(dim=l)
    q = CondDTransformerProposal(
        d_features=4,
        k_categories=6,
        u_dims=l,
        dropout=0.0,
        num_layers=1,
        nhead=2,
    )
    U = p.sample(sample_shape=torch.Size([3]))
    q.eval()
    with q.record_sample_log_prob():
        Xs, logqX = q(U)
    assert Xs.shape == (3, 4)
    assert logqX.shape == (3,)
    assert_close(logqX, q._last_sample_log_prob)


@pytest.mark.parametrize("k, pad_token", [[4, None], [6, 5], [5, 2]])
def test_cond_transformer_mutation(k, pad_token):
    d = 7
    l = 3
    p = UnitNormal(dim=l)
    X0 = torch.randint(0, k, (10, d))
    q = CondTransformerMutationProposal(
        d_features=d,
        k_categories=k,
        u_dims=l,
        X0=X0,
        num_mutations=2,
        pad_token=pad_token,
        replacement=False,  # need this for the test to pass
    )
    U = p.sample(sample_shape=torch.Size([10]))
    q.set_seeds(X0, U)
    q.eval()
    with q.record_sample_log_prob():
        Xs, logqX = q(U)
    assert Xs.shape == (10, d)
    assert logqX.shape == (10,)
    assert_close(logqX, q._last_sample_log_prob)
