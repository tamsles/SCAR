"""Minimal batch adaptation isolated from model and training logic."""

from typing import Any, Mapping, Tuple

import torch


def unpack_batch(
    batch: Any, x_key: str = "x", bar_y_key: str = "bar_y"
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Extract ``(x, bar_y)`` from tuple/list or mapping batches."""
    if isinstance(batch, (tuple, list)):
        if len(batch) < 2:
            raise ValueError("tuple/list batches must contain x and bar_y")
        x, bar_y = batch[0], batch[1]
    elif isinstance(batch, Mapping):
        if x_key not in batch or bar_y_key not in batch:
            raise KeyError(
                "mapping batch must contain {!r} and {!r}".format(x_key, bar_y_key)
            )
        x, bar_y = batch[x_key], batch[bar_y_key]
    else:
        raise TypeError("batch must be a tuple/list or mapping")
    if not torch.is_tensor(x) or not torch.is_tensor(bar_y):
        raise TypeError("x and bar_y must be torch tensors")
    return x, bar_y

