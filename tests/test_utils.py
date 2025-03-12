import numpy as np
import pytest
import torch

from mxbai_rerank.utils import TorchModule, auto_device, ensure_multiple_of_8, top_k_numpy


def test_top_k_numpy():
    # Test basic functionality
    scores = np.array([0.1, 0.5, 0.3, 0.8, 0.2])
    top_scores, top_indices = top_k_numpy(scores, k=3, sort=True)
    assert len(top_scores) == 3
    assert len(top_indices) == 3
    assert np.array_equal(top_scores, np.array([0.8, 0.5, 0.3]))
    assert np.array_equal(top_indices, np.array([3, 1, 2]))

    # Test with k larger than array size
    top_scores, top_indices = top_k_numpy(scores, k=10, sort=True)
    assert len(top_scores) == 5
    assert len(top_indices) == 5

    # Test without sorting
    top_scores, top_indices = top_k_numpy(scores, k=3, sort=False)
    assert len(top_scores) == 3
    assert len(top_indices) == 3
    assert np.all(np.sort(top_indices) == top_indices)  # Should be sorted by index

    # Test error cases
    with pytest.raises(ValueError):
        top_k_numpy(scores, k=0)
    with pytest.raises(ValueError):
        top_k_numpy(scores, k=-1)


def test_auto_device():
    device = auto_device()
    assert isinstance(device, str)
    assert device in ["cuda", "npu", "mps", "cpu"]

def test_ensure_multiple_of_8():
    assert ensure_multiple_of_8(0) == 0
    assert ensure_multiple_of_8(1) == 8
    assert ensure_multiple_of_8(8) == 8
    assert ensure_multiple_of_8(9) == 16

class TestTorchModule:
    def test_device_property(self):
        module = TorchModule()
        assert isinstance(module.device, torch.device)
        assert module.device.type == "cpu"  # Default device should be CPU

    def test_dtype_property(self):
        module = TorchModule()
        assert isinstance(module.dtype, torch.dtype)
        assert module.dtype == torch.float32  # Default dtype

    def test_cpu_method(self):
        module = TorchModule()
        module_cpu = module.cpu()
        assert module_cpu.device.type == "cpu"

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_cuda_method(self):
        module = TorchModule()
        module_cuda = module.cuda()
        assert module_cuda.device.type == "cuda"
