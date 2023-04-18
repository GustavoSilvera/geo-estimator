## Begin model

import torch
import numpy as np

from utils import device


class GeoGuesser(torch.nn.Module):
    def __init__(self, dataset: torch.utils.data.Dataset, train_idxs: torch.Tensor, test_idxs: torch.Tensor):
        super().__init__()
        self.dataset = dataset
        self.train_idxs = train_idxs
        self.test_idxs = test_idxs

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # TODO
        return x
