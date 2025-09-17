import torch

from vsd.solvers import train_val_split


def test_train_val_split_balanced():
    # 60 zeros, 40 ones
    z = torch.tensor([0] * 60 + [1] * 40, dtype=torch.int64)
    val_prop = 0.25
    train_idx, val_idx = train_val_split(z, val_prop)

    # Expected per-class validation counts
    n_pos = 40
    n_neg = 60
    exp_val_pos = max(1, round(n_pos * val_prop))
    exp_val_neg = max(1, round(n_neg * val_prop))

    # Check sizes
    assert len(val_idx) == exp_val_pos + exp_val_neg
    assert len(train_idx) == (n_pos - exp_val_pos) + (n_neg - exp_val_neg)

    # Check stratification and no-overlap for large classes
    pos_idx = torch.nonzero(z.to(torch.bool), as_tuple=False).squeeze(-1)
    neg_idx = torch.nonzero(~z.to(torch.bool), as_tuple=False).squeeze(-1)

    val_pos = len(set(val_idx.tolist()).intersection(set(pos_idx.tolist())))
    val_neg = len(set(val_idx.tolist()).intersection(set(neg_idx.tolist())))
    assert val_pos == exp_val_pos
    assert val_neg == exp_val_neg

    train_pos = len(set(train_idx.tolist()).intersection(set(pos_idx.tolist())))
    train_neg = len(set(train_idx.tolist()).intersection(set(neg_idx.tolist())))
    assert train_pos == n_pos - exp_val_pos
    assert train_neg == n_neg - exp_val_neg

    # No overlap for large classes
    assert len(set(train_idx.tolist()).intersection(set(val_idx.tolist()))) == 0


def test_train_val_split_small_overlap():
    # 50 zeros (large), 10 ones (small)
    z = torch.tensor([0] * 50 + [1] * 10, dtype=torch.int64)
    val_prop = 0.2
    overlap_threshold = 20
    train_idx, val_idx = train_val_split(z, val_prop, overlap_if_class_size_lt=overlap_threshold)

    pos_idx = torch.nonzero(z.to(torch.bool), as_tuple=False).squeeze(-1)
    neg_idx = torch.nonzero(~z.to(torch.bool), as_tuple=False).squeeze(-1)

    n_pos = len(pos_idx)
    n_neg = len(neg_idx)
    exp_val_pos = max(1, round(n_pos * val_prop))  # 2
    exp_val_neg = max(1, round(n_neg * val_prop))  # 10

    # Validation counts per class
    val_pos = len(set(val_idx.tolist()).intersection(set(pos_idx.tolist())))
    val_neg = len(set(val_idx.tolist()).intersection(set(neg_idx.tolist())))
    assert val_pos == exp_val_pos
    assert val_neg == exp_val_neg

    # Small class should be fully in train (overlap allowed)
    train_pos = len(set(train_idx.tolist()).intersection(set(pos_idx.tolist())))
    assert train_pos == n_pos
    # And there should be exact overlap with validation set for that class
    overlap_pos = len(
        set(train_idx.tolist()).intersection(set(val_idx.tolist())).intersection(
            set(pos_idx.tolist())
        )
    )
    assert overlap_pos == exp_val_pos

    # Large class should not overlap
    train_neg = len(set(train_idx.tolist()).intersection(set(neg_idx.tolist())))
    assert train_neg == n_neg - exp_val_neg


def test_train_val_split_no_val():
    z = torch.tensor([0, 1, 0, 1, 1, 0, 0], dtype=torch.int64)
    train_idx, val_idx = train_val_split(z, 0.0)
    assert len(val_idx) == 0
    assert set(train_idx.tolist()) == set(range(len(z)))


def test_train_val_split_types():
    z_int = torch.tensor([0, 1, 0, 1, 1, 0], dtype=torch.int64)
    z_float = z_int.to(torch.float32)
    z_bool = z_int.to(torch.bool)

    t_i, v_i = train_val_split(z_int, 0.3)
    t_f, v_f = train_val_split(z_float, 0.3)
    t_b, v_b = train_val_split(z_bool, 0.3)

    # All produce valid partitions of indices of the same size
    assert len(t_i) == len(t_f) == len(t_b)
    assert len(v_i) == len(v_f) == len(v_b)
