from __future__ import annotations

from typing import Optional, Union

import numpy as np
import torch


def sigmoid_normalize(x: np.ndarray, estimated_max: float = 9.0) -> np.ndarray:
    """Sigmoid function with a fixed maximum value.

    Args:
        x: Input array
        estimated_max: Estimated maximum value of the input

    Returns:
        Values between 0 and 1
    """
    x = x - estimated_max / 2
    return 1 / (1 + np.exp(-x))


def top_k_numpy(scores: np.ndarray, k: int, *, sort: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """Get the top k scores and their indices from a numpy array.

    Args:
        scores: Input array of scores.
        k: Number of top elements to return.
        sort: Whether to sort the results by score. Otherwise they will be sorted by index.

    Returns:
        Top k scores and their indices.

    Raises:
        ValueError: If input is not a numpy array or k is not a positive integer.

    """
    if k <= 0:
        msg = "Input 'k' must be positive. Got: %s"
        raise ValueError(msg, k)

    k = min(k, len(scores))
    top_k_indices = np.argpartition(scores, -k)[-k:]

    if sort:
        top_k_scores = scores[top_k_indices]
        # Sort by the negative of the scores (descending) primarily,
        # and index (ascending) secondarily to maintain original order for ties.
        sorted_indices = np.lexsort((top_k_indices, -top_k_scores))
        top_k_scores = top_k_scores[sorted_indices]
        top_k_indices = top_k_indices[sorted_indices]
    else:
        top_k_indices = np.sort(top_k_indices)
        top_k_scores = scores[top_k_indices]

    return top_k_scores, top_k_indices


def auto_device() -> str:
    """Determine the appropriate device type for PyTorch operations.

    Returns:
        The name of the device to use ('cuda', or 'cpu').
    """

    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def ensure_multiple_of_8(x: int, max_value: Optional[int] = None) -> int:
    """Ensure a number is a multiple of 8.

    Args:
        x: The number to ensure is a multiple of 8.
        max_value: The maximum value that the result should not exceed.

    Returns:
        The nearest multiple of 8 that is greater than or equal to x,
        unless that would exceed max_value, in which case returns the
        largest multiple of 8 that is less than or equal to max_value.
    """
    if max_value is not None:
        max_value -= max_value % 8

    remainder = x % 8
    if remainder == 0:
        return x

    # Round up to next multiple of 8
    next_multiple = x + (8 - remainder)

    # Check if the next multiple exceeds max_value
    if max_value is not None and next_multiple > max_value:
        return max_value

    return next_multiple


CPU_INCOMPATIBLE_DTYPE = [torch.float16, torch.bfloat16, torch.half]


class TorchModule(torch.nn.Module):
    """A base class for PyTorch modules with additional utility methods, including multi-processing support."""

    def __init__(self):
        super().__init__()
        self.register_buffer("_dummy", torch.empty(0), persistent=False)

    def to(self, device: Optional[Union[str, torch.device, int]] = None, dtype: Optional[torch.dtype] = None, **kwargs):
        """Move the module to a specific device and/or dtype.

        Args:
            device: The device to move the module to.
            dtype: The dtype to move the module to.
        """
        if torch.device(device).type == "cpu":
            dtype = dtype or self.dtype
            dtype = torch.float32 if dtype in CPU_INCOMPATIBLE_DTYPE else dtype

        return super().to(device=device, dtype=dtype, **kwargs)

    @property
    def device(self) -> torch.device:
        """Get the current device of the module."""
        return self._dummy.device

    @property
    def dtype(self) -> torch.dtype:
        """Get the current dtype of the module."""
        return self._dummy.dtype

    def mps(self):
        """Move the module to a MPS device."""
        if not torch.backends.mps.is_available():
            msg = "MPS is not available"
            raise RuntimeError(msg)
        return self.to(torch.device("mps"))
