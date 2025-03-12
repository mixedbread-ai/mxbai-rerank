import pytest
import torch


@pytest.fixture(autouse=True)
def no_torch_grad():
    """Disable gradient computation for all tests."""
    with torch.no_grad():
        yield


@pytest.fixture
def device():
    """Return the default device for testing."""
    return torch.device("cpu")


@pytest.fixture
def sample_queries():
    """Return a list of sample queries."""
    return [
        "what is machine learning",
        "how to make pizza",
        "best programming languages",
    ]


@pytest.fixture
def sample_documents():
    """Return a list of sample documents."""
    return [
        "Machine learning is a subset of artificial intelligence that focuses on data and algorithms.",
        "To make pizza, you need flour, water, yeast, and toppings of your choice.",
        "Popular programming languages include Python, JavaScript, and Java.",
    ]


@pytest.fixture
def sample_scores():
    """Return sample relevance scores."""
    return torch.tensor([0.8, 0.5, 0.9]) 