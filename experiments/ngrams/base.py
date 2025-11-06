import numpy as np

class BaseTask:
    def __init__(self, tokenizer, obj_dim, max_len, transform=lambda x: x, batch_size=1, **kwargs):
        self.tokenizer = tokenizer
        self.obj_dim = obj_dim
        self.transform = transform
        self.batch_size = batch_size
        self.max_len = max_len

    def _evaluate(self, x, out, *args, **kwargs):
        raise NotImplementedError

    def score(self, str_array):
        raise NotImplementedError