"""Convenience classes for wrapping datasets."""

from abc import ABC, abstractmethod
from itertools import product
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch import Tensor

from experiments import config as cf
from experiments.oracles import BaseCNN
from vsd.utils import SequenceArray

AA_TO_INT = {a: i for i, a in enumerate(cf.AMINO_ALPHA)}
INT_TO_AA = {i: a for i, a in enumerate(cf.AMINO_ALPHA)}
NA_TO_INT = {a: i for i, a in enumerate(cf.NUCLEIC_ALPHA)}
INT_TO_NA = {i: a for i, a in enumerate(cf.NUCLEIC_ALPHA)}


def seq2arr(S: SequenceArray) -> np.ndarray:
    X = np.vstack([np.array(list(s)).view("U1") for s in S])
    return X


def arr2seq(X: np.ndarray) -> SequenceArray:
    S = ["".join(list(x)) for x in X]
    return S


def seq2intarr(S: SequenceArray, is_amino=True) -> Tensor:
    Xc = seq2arr(S)
    c2int = AA_TO_INT.__getitem__ if is_amino else NA_TO_INT.__getitem__
    Xi = np.vectorize(c2int)(Xc)
    return torch.tensor(Xi).long()


def intarr2seq(X: Tensor, is_amino=True) -> SequenceArray:
    int2c = INT_TO_AA.__getitem__ if is_amino else INT_TO_NA.__getitem__
    Xc = np.vectorize(int2c)(X.detach().numpy())
    S = arr2seq(Xc)
    return S


class DataClass(ABC):

    fullspace = False

    @abstractmethod
    def make_training_data(self) -> Tuple[SequenceArray, np.ndarray]:
        pass

    @staticmethod
    def load_data(load_path: Path | str) -> Tuple[SequenceArray, np.ndarray]:
        data = pd.read_csv(load_path)
        S = list(data["sequence"].values)
        y = data["target"].values
        return S, y

    @staticmethod
    def save_data(
        save_path: Path | str,
        S: SequenceArray,
        y: np.ndarray,
        other_data: Optional[Dict[str, np.ndarray]] = None,
    ):
        path = Path(save_path)
        data = {"sequence": S, "target": y}
        if other_data is not None:
            data |= other_data
        pd.DataFrame(data).to_csv(path)

    @abstractmethod
    def get_fitness(self, S: SequenceArray) -> np.ndarray:
        pass

    @abstractmethod
    def get_max_fitness(self) -> Tuple[float, str]:
        pass

    @abstractmethod
    def get_min_fitness(self) -> Tuple[float, str]:
        pass

    @abstractmethod
    def get_seq_len(self) -> int:
        pass

    @abstractmethod
    def get_alpha_len(self) -> int:
        pass


class GFP(DataClass):

    config = cf.GFP_DATA
    Swt = (
        "SKGEELFTGVVPILVELDGDVNGHKFSVSGEGEGDATYGKLTLKFICTTGKLPVPWPTLVTTLSYG"
        "VQCFSRYPDHMKQHDFFKSAMPEGYVQERTIFFKDDGNYKTRAEVKFEGDTLVNRIELKGIDFKEDG"
        "NILGHKLEYNYNSHNVYIMADKQKNGIKVNFKIRHNIEDGSVQLADHYQQNTPIGDGPVLLPDNHYL"
        "STQSALSKDPNEKRDHMVLLEFVTAAGITHGMDELYK"
    )

    def __init__(self, device="cpu"):
        # Load data
        self.device = device
        self.dataloc = Path(self.config["data_source"])
        df = pd.read_csv(self.dataloc)
        df = df[df["augmented"] == 0]
        # Drop duplicate sequences
        df = df.drop_duplicates(
            subset=self.config["mutation_column"], keep="first"
        )
        self.y = df[self.config["target_column"]].values
        self.S = list(df[self.config["mutation_column"]].values)

        # Load oracle
        self.oracle = BaseCNN()
        state = torch.load(
            self.config["oracle_path"],
            map_location=torch.device(self.device),
            weights_only=True,
        )["state_dict"]
        state = {k.replace("predictor.", ""): v for k, v in state.items()}
        self.oracle.load_state_dict(state_dict=state)
        self.oracle.to(device).eval()

        self.lookup = {s: y for s, y in zip(self.S, self.y)}

    def make_training_data(
        self,
        size: int,
        max_fitness: float,
        min_fitness: float,
        save_path: Path | str,
        oracle_targets: bool = False,
        seed: Optional[None] = None,
    ) -> Tuple[SequenceArray, np.ndarray]:
        # S, y = self.load_data(save_path)  # for Kirjner
        rnd = np.random.RandomState(seed)
        ind = _random_threshold_fitness(
            self.y, max_fitness, size, rnd, min_fitness
        )
        Str = [self.S[i] for i in ind]

        if oracle_targets:
            ytr = self.get_fitness(Str)
        else:
            ytr = self.y[ind]

        self.save_data(save_path, Str, ytr)
        return Str, ytr

    def get_fitness(self, S: SequenceArray) -> np.ndarray:
        n = len(S)
        rmask = np.array([S[i] in self.lookup for i in range(n)])
        nin = sum(rmask)
        y = np.zeros(n)

        # Lookup
        if nin > 0:
            rin = np.where(rmask)[0]
            for i in rin:
                y[i] = self.lookup[S[i]]

        # Use oracle
        if nin < n:
            rout = np.where(~rmask)[0]
            Sout = [S[i] for i in rout]
            X = seq2intarr(Sout).to(self.device)
            y[rout] = self.oracle(X).detach().to("cpu").numpy()

        return y

    def get_max_fitness(self) -> Tuple[float, str]:
        i = np.argmax(self.y)
        return self.y[i], self.S[i]

    def get_min_fitness(self) -> Tuple[float, str]:
        i = np.argmin(self.y)
        return self.y[i], self.S[i]

    def get_seq_len(self) -> int:
        return len(self.S[0])

    def get_alpha_len(self) -> int:
        return len(cf.AMINO_ALPHA)


class AAV(GFP):

    config = cf.AAV_DATA
    Swt = "DEEEIRTTNPVATEQYGSVSTNLQRGNR"


class DHFR(DataClass):

    alpha = cf.NUCLEIC_ALPHA
    fullspace = True

    def __init__(self, default_fitness: float = -1.0, device="cpu"):
        super().__init__()
        self.default_fitness = default_fitness

        # Load data
        self.dataloc = Path(cf.DHFR_DATA["data_source"])
        df = pd.read_csv(self.dataloc, index_col=0)
        self.y = df[cf.DHFR_DATA["target_column"]].values
        self.S = list(df[cf.DHFR_DATA["mutation_column"]].values)

        # Get wildtype information
        self.Ywt = self.y[0]
        self.Swt = self.S[0]

        # Make a sequence-fitness lookup table
        self.lookup = {s: y for s, y in zip(self.S, self.y)}

    def make_training_data(
        self,
        save_path: Path | str,
        size: int,
        max_fitness: float,
        include_wt: bool = False,
        include_scan: bool = False,
        seed: Optional[int] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        # Random state
        rnd = np.random.RandomState(seed)

        # Sweep data
        Str = []
        if include_scan:
            for i, a in product(range(len(self.Swt)), self.alpha):
                s = self.Swt[:i] + a + self.Swt[(i + 1) :]
                Str.append(s)
        elif include_wt:
            Str.append(self.Swt)

        # Random low fitness data
        ind = _random_threshold_fitness(self.y, max_fitness, size, rnd)
        Slow = [self.S[i] for i in ind if self.S[i] not in Str]

        # Join
        Str.extend(Slow)
        ytr = self.get_fitness(Str)
        self.save_data(save_path, Str, ytr)
        return Str, ytr

    def get_fitness(self, S: np.ndarray) -> np.ndarray:
        y = np.zeros(len(S))
        for i, s in enumerate(S):
            y[i] = self.lookup.get(s, self.default_fitness)
        return y

    def get_max_fitness(self) -> Tuple[float, str]:
        i = np.argmax(self.y)
        return self.y[i], self.S[i]

    def get_min_fitness(self) -> Tuple[float, str]:
        i = np.argmin(self.y)
        return self.y[i], self.S[i]

    def get_seq_len(self) -> int:
        return len(self.S[0])

    def get_alpha_len(self) -> int:
        return len(self.alpha)

    def get_good_sequences(self, best_f) -> SequenceArray:
        gind = self.y > best_f
        return [self.S[i] for i in gind]


class TRPB(DHFR):

    alpha = cf.AMINO_ALPHA

    def __init__(self, default_fitness: float = -0.2, device="cpu"):
        super().__init__()
        self.default_fitness = default_fitness

        # Load data
        self.dataloc = Path(cf.TRPB_DATA["data_source"])
        df = pd.read_csv(self.dataloc)

        # Remove all "stop" codons
        df = df.loc[df["# Stop"] < 1]
        self.y = df[cf.TRPB_DATA["target_column"]].values
        self.S = list(df[cf.TRPB_DATA["mutation_column"]].values)

        # Make a sequence-fitness lookup table
        self.lookup = {s: y for s, y in zip(self.S, self.y)}

        # Get wildtype information
        self.Swt = "VFVS"
        self.Ywt = self.lookup[self.Swt]


class _TFBIND(DHFR):

    config = None

    def __init__(
        self,
        default_fitness: float = -1.0,
        device="cpu",
    ):
        super().__init__()
        self.default_fitness = default_fitness

        # Load data
        self.dataloc = Path(self.config["data_source"])
        df = pd.read_csv(self.dataloc)
        # Drop duplicate sequences
        df = df.drop_duplicates(
            subset=self.config["mutation_column"], keep="first"
        )
        self.y = df[self.config["target_column"]].values
        self.S = list(df[self.config["mutation_column"]].values)

        # Get wildtype information -- just use median
        medind = np.argwhere(np.abs(self.y - np.median(self.y)) < 1e-5)[0, 0]
        self.Ywt = self.y[medind]
        self.Swt = self.S[medind]

        # Make a sequence-fitness lookup table
        self.lookup = {s: y for s, y in zip(self.S, self.y)}

    def make_training_data(
        self,
        save_path: Path | str,
        size: int,
        max_fitness: float,
        include_scan: bool = False,
        seed: Optional[int] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        return super().make_training_data(
            save_path, size, max_fitness, include_scan, seed
        )


class TFBIND8(_TFBIND):
    config = cf.TFBIND8_DATA


# class TFBIND10(_TFBIND):
#     config = cf.TFBIND10_DATA


#
#  Dataset register
#

DATASETS = {
    "GFP": {"data": GFP, "config": cf.GFP_DATA},
    "AAV": {"data": AAV, "config": cf.AAV_DATA},
    "DHFR": {"data": DHFR, "config": cf.DHFR_DATA},
    "DHFR_FL": {"data": DHFR, "config": cf.DHFR_FL_DATA},
    "TFBIND8": {"data": TFBIND8, "config": cf.TFBIND8_DATA},
    "TFBIND8_FL": {"data": TFBIND8, "config": cf.TFBIND8_FL_DATA},
    "TRPB": {"data": TRPB, "config": cf.TRPB_DATA},
    "TRPB_FL": {"data": TRPB, "config": cf.TRPB_FL_DATA},
}


#
# Private stuff
#


def _random_threshold_fitness(y, ymax, n, rnd, ymin=None):
    fit_mask = y <= ymax
    if ymin is not None:
        fit_mask &= y > ymin
    fit_ind = np.where(fit_mask)[0]
    if len(fit_ind) > n:
        tr_ind = rnd.choice(fit_ind, size=n, replace=False)
        return tr_ind
    return fit_ind
