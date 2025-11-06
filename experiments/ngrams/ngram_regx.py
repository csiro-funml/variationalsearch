import re

import numpy as np

from experiments.ngrams.base import BaseTask
from experiments.ngrams.ngram_utils import random_strings


class RegexTask(BaseTask):
    def __init__(
        self,
        regex_list,
        max_len,
        min_len,
        num_start_examples,
        tokenizer,
        obj_dim,
        transform=lambda x: x,
        seed: int | None = None,
        **kwargs,
    ):
        super().__init__(tokenizer, obj_dim, max_len, transform, **kwargs)
        self.regex_list = regex_list
        self.min_len = min_len
        self.num_start_examples = num_start_examples
        self.max_reward_per_dim = kwargs["max_score_per_dim"]
        self.seed = seed

    def task_setup(self, *args, **kwargs):
        num_examples = 0
        selected_seqs = []
        selected_targets = []
        rand = np.random.RandomState(seed=self.seed)
        while num_examples < self.num_start_examples:
            # account for start and stop tokens
            all_seqs = random_strings(
                self.num_start_examples,
                self.min_len,
                self.max_len - 2,
                self.tokenizer.non_special_vocab,
                rand=rand,
            )
            all_targets = self.score(all_seqs)
            positive_example_mask = (all_targets > 0).sum(-1).astype(bool)
            num_positive = positive_example_mask.astype(int).sum()
            num_negative = all_targets.shape[0] - num_positive
            num_selected = min(num_positive, num_negative)

            selected_seqs.append(all_seqs[positive_example_mask][:num_selected])
            selected_targets.append(
                all_targets[positive_example_mask][:num_selected]
            )
            selected_seqs.append(
                all_seqs[~positive_example_mask][:num_selected]
            )
            selected_targets.append(
                all_targets[~positive_example_mask][:num_selected]
            )
            num_examples += num_selected

        all_seqs = np.concatenate(selected_seqs)[: self.num_start_examples]
        all_targets = np.concatenate(selected_targets)[
            : self.num_start_examples
        ]

        x0 = np.array([list(s) for s in all_seqs])
        y0 = self.score(x0)
        return x0, y0
        # return all_seqs, all_targets

    def score(self, candidates):
        str_array = np.array(candidates)
        scores = []
        for regex in self.regex_list:
            scores.append(
                np.array(
                    [
                        len(re.findall(regex, str(x))) / self.max_reward_per_dim
                        for x in str_array
                    ]
                ).reshape(-1)
            )
        scores = np.stack(scores, axis=-1).astype(np.float64)
        return scores
