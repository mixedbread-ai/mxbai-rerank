from __future__ import annotations

from typing import Optional

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from mxbai_rerank.base import BaseReranker
from mxbai_rerank.utils import TorchModule, auto_device


class MxbaiRerankV1(BaseReranker, TorchModule):
    """Reranker model using a pre-trained sequence classification model."""

    def __init__(
        self,
        model_name_or_path: str,
        *,
        device: str = auto_device(),
        max_length: Optional[int] = None,
        tokenizer_kwargs: Optional[dict] = None,
        torch_dtype: str | torch.dtype = "auto",
        **kwargs,
    ):
        TorchModule.__init__(self)
        tokenizer_kwargs = tokenizer_kwargs or {}

        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name_or_path,
            **{
                "torch_dtype": torch_dtype,
                "device_map": device,
                **kwargs,
            },
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, **tokenizer_kwargs)
        self.max_length = max_length or self.tokenizer.model_max_length

        self.to(self.model.device, dtype=self.model.dtype)

    def predict(self, queries: list[str], documents: list[str]) -> torch.Tensor:
        features = self.tokenizer.batch_encode_plus(
            batch_text_or_text_pairs=list(zip(queries, documents)),
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        with torch.inference_mode():
            outputs = self.model(**{k: v.to(self.device) for k, v in features.items()})
            return torch.sigmoid(outputs.logits).squeeze(dim=-1).cpu()
