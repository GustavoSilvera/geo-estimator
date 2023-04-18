import torch

try:
    from utils import device
except:
    pass  # already in global context


class GeoGuesser(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # TODO

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # TODO
        return x
