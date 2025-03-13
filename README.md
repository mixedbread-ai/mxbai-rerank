# Mixedbread Reranking Models

[![PyPI version](https://badge.fury.io/py/mxbai-rerank.svg)](https://badge.fury.io/py/mxbai-rerank)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

Crispy reranking models from [Mixedbread](https://mixedbread.com). State-of-the-art models for search relevance, powered by reinforcement learning.

## Features

- **State-of-the-art performance** - Outperforms leading open and closed-source rerankers on major benchmarks
- **100+ languages** - Strong multilingual support out of the box
- **Long context** - Handle up to 8k tokens (32k-compatible)
- **Code & SQL** - Excellent at ranking code snippets and technical content
- **Function Call Ranking** - Supports reranking of function calls for multi-tool agents
- **Fast inference** - 8x faster than comparable models
- **Easy integration** - Drop-in improvement for existing search systems
- **Open source** - Apache 2.0-licensed, easy to customize
- **Managed API** - For production use with additional features. We support embeddings, reranking, and an end-to-end multi-modal retrieval solution.

## Installation

```bash
pip install -U mxbai-rerank
```

## Quick Start

```python
from mxbai_rerank import MxbaiRerankV2

# Initialize the reranker
reranker = MxbaiRerankV2("mixedbread-ai/mxbai-rerank-base-v2")  # or large-v2

# Example query and documents
query = "Who wrote 'To Kill a Mockingbird'?"
documents = [
    "'To Kill a Mockingbird' is a novel by Harper Lee published in 1960.",
    "The novel 'Moby-Dick' was written by Herman Melville.",
    "Harper Lee was born in 1926 in Monroeville, Alabama."
]

results = reranker.rank(query=query, documents=documents)

print(results)
```

## Models

We offer multiple model variants. For more details, see our [mxbai-rerank-v2 technical blog post](https://mixedbread.com/blog/mxbai-rerank-v2).

- **mxbai-rerank-base-v2** (0.5B) - Best balance of speed and accuracy
- **mxbai-rerank-large-v2** (1.5B) - Highest accuracy, still with excellent speed

### Legacy Models

For more details, see our [mxbai-rerank-v1 technical blog post](https://mixedbread.com/blog/mxbai-rerank-v1).

- **mxbai-rerank-xsmall-v1** (0.1B) - Fastest inference, lower accuracy
- **mxbai-rerank-base-v1** (0.2B) - Smaller, faster model
- **mxbai-rerank-large-v1** (1.5B) - Large model with highest accuracy

## Performance

### Benchmark Results

| Model | BEIR Avg | Multilingual | Chinese | Code Search | Latency (s) |
|-------|----------|----------|----------|--------------|-------------|
| mxbai-rerank-large-v2 | 57.49 | 29.79 | 84.16 | 32.05 | 0.89 |
| mxbai-rerank-base-v2 | 55.57 | 28.56 | 83.70 | 31.73 | 0.67 |
| mxbai-rerank-large-v1 | 49.32 | 21.88 | 72.53 | 30.72 | 2.24 |

*Latency measured on A100 GPU

## Advanced Usage

### Flash Attention Support

The v2 models automatically use Flash Attention 2 when available for faster inference:

```bash
pip install flash-attn --no-build-isolation
```

### Long Context Support

```python
reranker = MxbaiRerankV2(
    "mixedbread-ai/mxbai-rerank-base-v2",
    max_length=8192  # Default, can be adjusted up to model limits (32k for v2 models)
)
```

### Instruction Support

```python
results = reranker.rank(query=query, documents=documents, instruction="Figure out the best code snippet for the user query.")
```

## API Access

For managed API access with additional features, such as object reranking and instructions:

```python
from mixedbread import Mixedbread

mxbai = Mixedbread(api_key="YOUR_API_KEY")

results = mxbai.rerank(
    model="mixedbread-ai/mxbai-rerank-large-v2",
    query="your query",
    input=["doc1", "doc2", "doc3"]
)
```

## Training Details

The models were trained using a three-step process:

1. **GRPO (Guided Reinforcement Prompt Optimization)**
2. **Contrastive Learning**
3. **Preference Learning**

For more details, check our [technical blog post](https://mixedbread.com/blog/mxbai-rerank-v2).

Paper following soon.

## Citation

If you use this work, please cite:

```bibtex
@online{v2rerank2025mxbai,
  title={Baked-in Brilliance: Reranking Meets RL with mxbai-rerank-v2},
  author={Sean Lee and Rui Huang and Aamir Shakir and Julius Lipp},
  year={2024},
  url={https://www.mixedbread.com/blog/mxbai-rerank-v2},
}
```

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Please feel free to submit a pull request or report an issue on GitHub.

## Community & Support

- [Discord Community](https://mixedbread.com/redirects/discord)
- [X/Twitter](https://mixedbread.com/redirects/twitter)
- [LinkedIn](https://mixedbread.com/redirects/linked-in)
