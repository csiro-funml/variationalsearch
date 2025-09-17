"""Addition black-boxes compatible with the Poli Interface"""

import numpy as np
import torch
import typing as T
import re
from poli.core.abstract_black_box import AbstractBlackBox
from poli.core.abstract_problem_factory import AbstractProblemFactory
from poli.core.black_box_information import BlackBoxInformation
from poli.core.problem import Problem
from poli.core.util.seeding import seed_python_numpy_and_torch
from transformers import (
    AutoModelForCausalLM,
    AutoModelForMaskedLM,
    AutoTokenizer,
)
from experiments.ngrams.ngram_utils import ResidueTokenizer

#
# Custom Black Boxes
#


class ProtGPT2Naturalness(AbstractBlackBox):
    """
    Wrap a pretrained protein LM (ProtGPT2) as a poli black‐box objective.
    """

    def __init__(
        self,
        model_name: str = "nferruz/ProtGPT2",
        device: str = "cpu",
    ):
        super().__init__()
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        # ensure padding token is set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = (
            AutoModelForCausalLM.from_pretrained(model_name).to(device).eval()
        )
        # parameter space: categorical alphabet tokens for each sequence position
        self.alphabet = list(self.tokenizer.get_vocab().keys())

    def _black_box(self, x: np.ndarray, context=None) -> np.ndarray:
        like_list = []
        for seq in x:
            s = "".join(seq)
            inputs = self.tokenizer(
                s, return_tensors="pt", add_special_tokens=False
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            with torch.no_grad():
                outputs = self.model(**inputs, labels=inputs["input_ids"])
            like = np.exp(-outputs.loss.item())  # Flip to 'naturalness'
            like_list.append([like])
        return np.array(like_list)

    def get_black_box_info(self) -> BlackBoxInformation:
        return BlackBoxInformation(
            name="ProtGPT2Naturalness",
            max_sequence_length=None,
            aligned=True,
            fixed_length=False,
            deterministic=True,
            alphabet=self.alphabet,
            log_transform_recommended=False,
            discrete=True,
            padding_token="",
        )


class ProtGPT2NaturalnessFactory(AbstractProblemFactory):
    """
    Factory for ProtGPT2 naturalness BlackBox.
    """

    def __init__(self):
        super().__init__()
        self.name = "ProtGPT2Naturalness"

    def create(self, **kwargs) -> AbstractBlackBox:
        return ProtGPT2Naturalness(
            model_name=kwargs.get("model_name", "nferruz/ProtGPT2"),
            device=kwargs.get("device", "cpu"),
        )


class HFMlmNaturalness(AbstractBlackBox):
    """
    Wrap a HuggingFace **masked language model** (e.g., ProtBert/DistilProtBert)
    as a poli black-box objective using **pseudo log-likelihood (PLL)**.

    Scoring: For a sequence x_1..x_L, mask each position i, obtain
    p(x_i | x_{-i}) from the model, compute mean log-prob, and return
    exp(mean_logprob) as a 'naturalness' score (higher is better).
    """

    def __init__(
        self, model_name: str, prepend_M: bool = True, device: str = "cpu"
    ):
        super().__init__()
        if not model_name or not isinstance(model_name, str):
            raise ValueError(
                "`model_name` must be a valid HF repo id, e.g. 'Rostlab/prot_bert'."
            )
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.mask_token is None:
            raise ValueError(
                "Tokenizer has no mask token; need a MaskedLM-compatible model."
            )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = (
                self.tokenizer.eos_token
                if hasattr(self.tokenizer, "eos_token")
                and self.tokenizer.eos_token is not None
                else self.tokenizer.mask_token
            )
        self.model = (
            AutoModelForMaskedLM.from_pretrained(model_name).to(device).eval()
        )
        # Restrict to canonical 20 amino acids by default
        self.alphabet = list("ACDEFGHIKLMNPQRSTVWY")
        self.prepend_M = prepend_M

    def _black_box(self, x: np.ndarray, context=None) -> np.ndarray:
        scores = []
        for seq in x:
            if self.prepend_M:
                seq = np.concatenate((["M"], seq))
            toks = self.tokenizer(
                seq.tolist(),
                is_split_into_words=True,
                return_tensors="pt",
                add_special_tokens=False,
            )
            input_ids = toks["input_ids"].to(self.device)  # (1, L)
            attention_mask = toks.get("attention_mask", None)
            if attention_mask is not None:
                attention_mask = attention_mask.to(self.device)
            L = input_ids.size(1)
            # Build batch of L masked variants
            masked_inputs = input_ids.repeat(L, 1)  # (L, L)
            arange_idx = torch.arange(L, device=self.device)
            mask_id = self.tokenizer.mask_token_id
            true_ids = input_ids[0].clone()  # (L,)
            masked_inputs[arange_idx, arange_idx] = mask_id
            if attention_mask is not None:
                masked_am = attention_mask.repeat(L, 1)
            else:
                masked_am = None
            with torch.no_grad():
                out = self.model(
                    input_ids=masked_inputs, attention_mask=masked_am
                )
                logits = out.logits  # (L, L, V)
                # We need logits at the masked positions i for token i
                logits_i = logits[arange_idx, arange_idx, :]  # (L, V)
                log_probs = torch.nn.functional.log_softmax(
                    logits_i, dim=-1
                )  # (L, V)
                token_lp = log_probs[arange_idx, true_ids]  # (L,)
            mean_lp = token_lp.mean().item()
            score = np.exp(mean_lp)  # pseudo likelihood
            scores.append([score])
        return np.array(scores)

    def get_black_box_info(self) -> BlackBoxInformation:
        return BlackBoxInformation(
            name="HFMlmNaturalness",
            max_sequence_length=None,
            aligned=True,
            fixed_length=False,
            deterministic=True,
            alphabet=self.alphabet,
            log_transform_recommended=False,
            discrete=True,
            padding_token="",
        )


class ProtBertNaturalnessFactory(AbstractProblemFactory):
    """Factory for ProtBert masked-LM naturalness (PLL) scorer."""

    def __init__(self):
        super().__init__()
        self.name = "ProtBertNaturalness"

    def create(self, **kwargs) -> AbstractBlackBox:
        return HFMlmNaturalness(
            model_name=kwargs.get("model_name", "Rostlab/prot_bert"),
            device=kwargs.get("device", "cpu"),
        )


class DistilProtBertNaturalnessFactory(AbstractProblemFactory):
    """Factory for DistilProtBert masked-LM naturalness (PLL) scorer."""

    def __init__(self):
        super().__init__()
        self.name = "DistilProtBertNaturalness"

    def create(self, **kwargs) -> AbstractBlackBox:
        return HFMlmNaturalness(
            model_name=kwargs.get("model_name", "yarongef/DistilProtBert"),
            device=kwargs.get("device", "cpu"),
        )


class NgramBlackBox(AbstractBlackBox):
    def __init__(
        self,
        tokenizer=None,
        regex_list=None,
        min_len=32,
        max_len=36,
        allow_len_change=True,
        max_score_per_dim=None,
        eval_pref=None,
        max_ngram_size=None,
        log_prefix="ngram",
        batch_size=1,
        parallelize=False,
        num_workers=1,
        evaluation_budget=10000,
    ):
        super().__init__(
            batch_size=batch_size,
            parallelize=parallelize,
            num_workers=num_workers,
            evaluation_budget=evaluation_budget,
        )

        self.alphabet = list("ACDEFGHIKLMNPQRSTVWY")
        self.tokenizer = tokenizer or ResidueTokenizer()
        self.regex_list = regex_list or ["(?=AV)", "(?=VC)", "(?=CA)"]
        self.min_len = min_len
        self.max_len = max_len
        self.allow_len_change = allow_len_change
        self.max_score_per_dim = max_score_per_dim or 18
        self.eval_pref = eval_pref or [1 / len(self.regex_list)] * len(
            self.regex_list
        )
        self.max_ngram_size = max_ngram_size
        self.log_prefix = log_prefix

        self._info = BlackBoxInformation(
            name="ngram",
            max_sequence_length=self.max_len,
            aligned=True,
            fixed_length=True,
            alphabet=self.alphabet,
            log_transform_recommended=False,
            deterministic=True,
            discrete=True,
        )

    def get_black_box_info(self) -> BlackBoxInformation:
        return self._info

    def _black_box(self, x: np.ndarray, context=None) -> np.ndarray:
        scores = []
        for row in x:
            seq = "".join(row)
            counts = [
                min(len(re.findall(pattern, seq)), self.max_score_per_dim)
                for pattern in self.regex_list
            ]
            scores.append(counts)
        return np.array(scores)


class NgramProblemFactory(AbstractProblemFactory):
    def create(
        self,
        tokenizer=None,
        regex_list=None,
        min_len=32,
        max_len=36,
        allow_len_change=True,
        max_score_per_dim=18,
        eval_pref=None,
        max_ngram_size=1,
        log_prefix="ngram",
        num_start_examples=512,
        batch_size=16,
        parallelize=False,
        num_workers=1,
        evaluation_budget=10000,
        seed=None,
        **kwargs,
    ) -> Problem:
        if seed is not None:
            seed_python_numpy_and_torch(seed)

        tokenizer = tokenizer or ResidueTokenizer()

        bb = NgramBlackBox(
            tokenizer=tokenizer,
            regex_list=regex_list,
            min_len=min_len,
            max_len=max_len,
            allow_len_change=allow_len_change,
            max_score_per_dim=max_score_per_dim,
            eval_pref=eval_pref,
            max_ngram_size=max_ngram_size,
            log_prefix=log_prefix,
            batch_size=batch_size,
            parallelize=parallelize,
            num_workers=num_workers,
            evaluation_budget=evaluation_budget,
            **kwargs,
        )

        x0 = np.random.choice(bb.alphabet, size=(num_start_examples, max_len))

        return Problem(bb, x0)


class CombinedBlackBox(AbstractBlackBox):

    def __init__(
        self,
        blackboxes: T.Sequence[AbstractBlackBox],
        offsets: T.Optional[np.ndarray] = None,
        scales: T.Optional[np.ndarray] = None,
        upperbounds: T.Optional[np.ndarray] = None,
        lowerbounds: T.Optional[np.ndarray] = None,
        alphabet_index: int = 0,
    ):
        super().__init__()
        self.blackboxes = blackboxes
        self.alphabet = blackboxes[alphabet_index].alphabet
        n = len(blackboxes)
        self.offsets = offsets if offsets is not None else np.zeros(n)
        self.scales = scales if scales is not None else np.ones(n)
        self.upperbounds = (
            upperbounds if upperbounds is not None else np.full(n, np.inf)
        )
        self.lowerbounds = (
            lowerbounds if lowerbounds is not None else np.full(n, -np.inf)
        )

    def _black_box(self, x: np.ndarray, context=None) -> np.ndarray:
        objs = [bb(x) for bb in self.blackboxes]
        objs = (np.hstack(objs) - self.offsets) / self.scales
        objs = np.maximum(objs, self.lowerbounds)
        objs = np.minimum(objs, self.upperbounds)
        return objs

    def get_black_box_info(self) -> BlackBoxInformation:
        return BlackBoxInformation(
            name="CombinedBlackBox",
            max_sequence_length=None,
            aligned=True,
            fixed_length=False,
            deterministic=True,
            alphabet=self.blackboxes[0].alphabet,
            log_transform_recommended=False,
            discrete=True,
            padding_token="",
        )

    def __call__(self, x: np.ndarray):
        if x.ndim < 2:
            x = np.array([np.array(list(s)) for s in x])
        return super().__call__(x)


class CombinedBlackBoxFactory(AbstractProblemFactory):

    def __init__(
        self,
        blackboxes: T.Sequence[AbstractBlackBox],
        offsets: T.Optional[np.ndarray] = None,
        scales: T.Optional[np.ndarray] = None,
        upperbounds: T.Optional[np.ndarray] = None,
        lowerbounds: T.Optional[np.ndarray] = None,
        alphabet_index: int = 0,
    ):
        super().__init__()
        self.blackboxes = blackboxes
        self.offsets = offsets
        self.scales = scales
        self.alphabet_index = alphabet_index
        self.upperbounds = upperbounds
        self.lowerbounds = lowerbounds

    def create(
        self, x0: T.Optional[np.ndarray] = None, **kwargs
    ) -> T.Tuple[AbstractBlackBox, np.ndarray | None]:
        bb = CombinedBlackBox(
            blackboxes=self.blackboxes,
            offsets=self.offsets,
            scales=self.scales,
            upperbounds=self.upperbounds,
            lowerbounds=self.lowerbounds,
            alphabet_index=self.alphabet_index,
        )
        if x0 is None:
            return bb, None
        else:
            return bb, bb(x0)
