import torch
import pytest


def assert_close(a: torch.Tensor, b: torch.Tensor):
    assert torch.allclose(a, b, atol=1e-6, rtol=1e-5), f"Mismatch: {a} vs {b}"


@pytest.fixture(autouse=True)
def _seed():
    torch.manual_seed(0)
