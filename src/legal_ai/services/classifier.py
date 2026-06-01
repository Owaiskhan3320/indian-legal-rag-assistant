from __future__ import annotations

import logging
from contextlib import nullcontext

import torch
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from legal_ai.config import Settings
from legal_ai.services.labels import LABEL_ALIASES, LABEL_ID_TO_NAME
from legal_ai.utils.text import normalize_whitespace


LOGGER = logging.getLogger(__name__)


class LegalClassifier:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.demo_mode = bool(settings.demo_mode)
        if self.demo_mode:
            LOGGER.info("Using deterministic demo classifier; NyayaAnumana model loading is skipped.")
            self.tokenizer = None
            self.model = None
            return
        LOGGER.info("Loading classifier on device=%s", self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(
            settings.classifier_repo_id,
            subfolder=settings.classifier_subfolder,
            use_fast=False,
        )
        self.model = AutoModelForSequenceClassification.from_pretrained(
            settings.classifier_repo_id,
            subfolder=settings.classifier_subfolder,
        ).to(self.device)
        self.model.eval()

    def predict(
        self,
        text: str,
        *,
        mode: str = "mean_all",
        positive_threshold: float | None = None,
    ) -> dict:
        cleaned = normalize_whitespace(text)
        if not cleaned:
            raise ValueError("Input text must be non-empty.")
        if self.demo_mode:
            return self._predict_demo(cleaned)

        encoded = self.tokenizer(
            cleaned,
            return_tensors="pt",
            truncation=True,
            stride=self.settings.chunk_stride,
            max_length=self.settings.max_input_tokens,
            padding="max_length",
            return_overflowing_tokens=True,
        )
        chunk_count = int(encoded["input_ids"].shape[0])

        logits_batches = []
        autocast_ctx = (
            torch.autocast(device_type="cuda", dtype=torch.float16)
            if self.device == "cuda"
            else nullcontext()
        )

        with torch.inference_mode():
            with autocast_ctx:
                for start in range(0, chunk_count, self.settings.inference_batch_size):
                    end = start + self.settings.inference_batch_size
                    batch = {
                        key: value[start:end].to(self.device)
                        for key, value in encoded.items()
                        if key in {"input_ids", "attention_mask", "token_type_ids"}
                    }
                    outputs = self.model(**batch)
                    logits_batches.append(outputs.logits.detach().cpu())

        chunk_logits = torch.cat(logits_batches, dim=0)
        final_logits = self._aggregate_chunk_logits(chunk_logits, mode=mode)
        probs = F.softmax(final_logits, dim=-1)

        pred_idx = int(torch.argmax(probs).item())
        raw_label = self.model.config.id2label[pred_idx]
        predicted_label = LABEL_ALIASES.get(raw_label, pred_idx)
        predicted_name = LABEL_ID_TO_NAME.get(predicted_label, raw_label)
        confidence = float(probs[pred_idx].item())
        positive_probability = float(probs[1:].sum().item()) if probs.shape[0] >= 3 else confidence
        if positive_threshold is None:
            binary_pred = 1 if predicted_label in {1, 2} else 0
        else:
            binary_pred = 1 if positive_probability >= positive_threshold else 0

        probabilities = {}
        for idx, value in enumerate(probs.tolist()):
            raw_name = self.model.config.id2label[idx]
            mapped_label = LABEL_ALIASES.get(raw_name, idx)
            mapped_name = LABEL_ID_TO_NAME.get(mapped_label, raw_name)
            probabilities[mapped_name] = round(float(value) * 100, 2)

        return {
            "predicted_label": predicted_label,
            "predicted_name": predicted_name,
            "confidence_score": round(confidence * 100, 2),
            "confidence_band": self._confidence_band(confidence),
            "probabilities": probabilities,
            "chunk_count": chunk_count,
            "positive_probability": round(positive_probability * 100, 2),
            "positive_probability_value": positive_probability,
            "binary_prediction": binary_pred,
            "aggregation_mode": mode,
        }

    @staticmethod
    def _predict_demo(cleaned: str) -> dict:
        lowered = cleaned.lower()
        label = 1
        confidence = 0.64
        probabilities = {"Rejected": 20.0, "Accepted": 64.0, "Partially Accepted": 16.0}

        if any(marker in lowered for marker in ("partial", "inspection only", "limited relief")):
            label = 2
            confidence = 0.58
            probabilities = {"Rejected": 22.0, "Accepted": 20.0, "Partially Accepted": 58.0}
        elif any(marker in lowered for marker in ("delay explained", "alternative remedy exhausted", "weak evidence")):
            label = 0
            confidence = 0.57
            probabilities = {"Rejected": 57.0, "Accepted": 28.0, "Partially Accepted": 15.0}
        elif any(
            marker in lowered
            for marker in (
                "no reply",
                "no response",
                "not answered",
                "without hearing",
                "no show-cause",
                "defective",
                "refund",
                "without authority of law",
                "without issuing acquisition notice",
            )
        ):
            label = 1
            confidence = 0.66
            probabilities = {"Rejected": 18.0, "Accepted": 66.0, "Partially Accepted": 16.0}

        positive_probability = (probabilities["Accepted"] + probabilities["Partially Accepted"]) / 100
        return {
            "predicted_label": label,
            "predicted_name": LABEL_ID_TO_NAME[label],
            "confidence_score": round(confidence * 100, 2),
            "confidence_band": LegalClassifier._confidence_band(confidence),
            "probabilities": probabilities,
            "chunk_count": 1,
            "positive_probability": round(positive_probability * 100, 2),
            "positive_probability_value": positive_probability,
            "binary_prediction": 1 if label in {1, 2} else 0,
            "aggregation_mode": "demo_keyword_rules",
        }

    @staticmethod
    def _confidence_band(confidence: float) -> str:
        if confidence >= 0.8:
            return "High"
        if confidence >= 0.6:
            return "Moderate"
        return "Low"

    @staticmethod
    def _aggregate_chunk_logits(chunk_logits: torch.Tensor, *, mode: str) -> torch.Tensor:
        if chunk_logits.ndim != 2 or chunk_logits.shape[0] == 0:
            raise ValueError("Expected chunk logits with shape [num_chunks, num_labels].")

        if mode == "mean_all":
            return chunk_logits.mean(dim=0)
        if mode == "last_chunk":
            return chunk_logits[-1]
        if mode == "last_two_mean":
            return chunk_logits[-min(2, chunk_logits.shape[0]) :].mean(dim=0)
        if mode == "head_tail_mean":
            if chunk_logits.shape[0] == 1:
                return chunk_logits[0]
            return torch.stack((chunk_logits[0], chunk_logits[-1]), dim=0).mean(dim=0)
        if mode == "max_confidence":
            per_chunk_conf = F.softmax(chunk_logits, dim=-1).max(dim=-1).values
            return chunk_logits[int(torch.argmax(per_chunk_conf).item())]
        if mode == "tail_weighted":
            weights = torch.arange(
                1,
                chunk_logits.shape[0] + 1,
                dtype=chunk_logits.dtype,
            )
            weights = weights / weights.sum()
            return (chunk_logits * weights.unsqueeze(1)).sum(dim=0)
        raise ValueError(f"Unknown classifier aggregation mode: {mode}")
