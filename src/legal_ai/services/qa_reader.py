from __future__ import annotations

import logging
from contextlib import nullcontext
import re
from typing import Any

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from legal_ai.config import Settings
from legal_ai.utils.text import normalize_whitespace, shorten_text


LOGGER = logging.getLogger(__name__)


class LegalQAReader:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._english_tokenizer = None
        self._english_model = None
        self._multilingual_tokenizer = None
        self._multilingual_model = None

    def answer(
        self,
        *,
        question: str,
        supporting_cases: list[dict[str, Any]],
        rag_context: dict[str, Any] | None,
        detected_language: str,
        answer_language: str,
        scope: str,
    ) -> dict[str, str]:
        if not supporting_cases:
            return {
                "text": self._fallback(
                    question=question,
                    scope=scope,
                    answer_language=answer_language,
                    reason="No retrieved authorities were available.",
                ),
                "source": "fallback",
            }

        use_multilingual_reader = detected_language != "English" or answer_language != "English"
        try:
            tokenizer, model, source_name = self._load_reader(multilingual=use_multilingual_reader)
            prompt = self._build_prompt(
                question=question,
                supporting_cases=supporting_cases,
                rag_context=rag_context,
                answer_language=answer_language,
                scope=scope,
            )
            encoded = tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=min(getattr(tokenizer, "model_max_length", 512), 1024),
            )
            encoded = {key: value.to(self.device) for key, value in encoded.items()}
            autocast_ctx = (
                torch.autocast(device_type="cuda", dtype=torch.float16)
                if self.device == "cuda"
                else nullcontext()
            )
            with torch.inference_mode():
                with autocast_ctx:
                    generated = model.generate(
                        **encoded,
                        max_new_tokens=self.settings.qa_reader_max_new_tokens,
                        num_beams=self.settings.qa_reader_num_beams,
                        do_sample=False,
                        early_stopping=True,
                    )
            decoded = tokenizer.decode(generated[0], skip_special_tokens=True).strip()
            answer_text = self._normalize_answer_text(decoded)
            if not answer_text:
                raise RuntimeError("The QA reader returned an empty answer.")
            return {"text": answer_text, "source": source_name}
        except Exception as exc:  # pragma: no cover
            LOGGER.warning("QA reader failed, using fallback. error=%s", exc)
            return {
                "text": self._fallback(
                    question=question,
                    scope=scope,
                    answer_language=answer_language,
                    reason="A model-written answer was unavailable for this run.",
                ),
                "source": "fallback",
            }

    def _load_reader(self, *, multilingual: bool):
        if multilingual:
            if self._multilingual_model is None or self._multilingual_tokenizer is None:
                LOGGER.info(
                    "Loading multilingual QA reader model=%s on device=%s",
                    self.settings.qa_multilingual_reader_model_name,
                    self.device,
                )
                self._multilingual_tokenizer = AutoTokenizer.from_pretrained(
                    self.settings.qa_multilingual_reader_model_name,
                    use_fast=False,
                )
                self._multilingual_model = AutoModelForSeq2SeqLM.from_pretrained(
                    self.settings.qa_multilingual_reader_model_name
                ).to(self.device)
                self._multilingual_model.eval()
            return (
                self._multilingual_tokenizer,
                self._multilingual_model,
                self.settings.qa_multilingual_reader_model_name,
            )

        if self._english_model is None or self._english_tokenizer is None:
            LOGGER.info(
                "Loading English QA reader model=%s on device=%s",
                self.settings.qa_reader_model_name,
                self.device,
            )
            self._english_tokenizer = AutoTokenizer.from_pretrained(
                self.settings.qa_reader_model_name,
                use_fast=False,
            )
            self._english_model = AutoModelForSeq2SeqLM.from_pretrained(
                self.settings.qa_reader_model_name
            ).to(self.device)
            self._english_model.eval()
        return self._english_tokenizer, self._english_model, self.settings.qa_reader_model_name

    def _build_prompt(
        self,
        *,
        question: str,
        supporting_cases: list[dict[str, Any]],
        rag_context: dict[str, Any] | None,
        answer_language: str,
        scope: str,
    ) -> str:
        context_text = (rag_context or {}).get("context_text") or ""
        if not context_text:
            authorities: list[str] = []
            remaining_chars = self.settings.qa_reader_max_context_chars
            for idx, item in enumerate(supporting_cases[:4], start=1):
                block = (
                    f"Authority {idx}\n"
                    f"Case ID: {item['case_id']}\n"
                    f"Outcome: {item.get('label_name') or 'Unknown'}\n"
                    f"Passage: {item.get('excerpt') or item.get('summary') or 'No passage provided.'}\n"
                )
                if remaining_chars - len(block) < 0:
                    break
                authorities.append(block)
                remaining_chars -= len(block)
            context_text = "".join(authorities)

        scope_label = "the full corpus" if scope == "corpus" else "the currently selected cases"
        return (
            f"Answer the legal question in {answer_language}. "
            "If English is requested, use English only. "
            "Use only the authorities below. "
            "If support is weak or conflicting, say so clearly. "
            "Cite case IDs in parentheses. "
            f"The retrieval scope was {scope_label}. "
            "Return only the final answer text for the user. "
            "Unless the user explicitly asks for points, numbering, or headings, avoid headings, bullet labels, and section titles. "
            "Write in clean prose with short paragraphs. "
            "If the user asks about a specific case, explain the facts, outcome, and reasoning from the retrieved passage. "
            "If the question asks for detail, explanation, or comparison, give a fuller answer in 2 to 4 short paragraphs.\n\n"
            f"Question: {question}\n\n"
            f"Authorities:\n{context_text}\n"
            "Answer:"
        )

    @staticmethod
    def _normalize_answer_text(text: str) -> str:
        cleaned = (text or "").replace("\r\n", "\n").strip()
        if not cleaned:
            return ""

        paragraphs = [
            normalize_whitespace(block)
            for block in re.split(r"\n\s*\n", cleaned)
            if normalize_whitespace(block)
        ]
        if paragraphs:
            return "\n\n".join(paragraphs)
        return normalize_whitespace(cleaned)

    @staticmethod
    def _fallback(
        *,
        question: str,
        scope: str,
        answer_language: str,
        reason: str,
    ) -> str:
        scope_label = "whole corpus retrieval" if scope == "corpus" else "current-result retrieval"
        if answer_language != "English":
            return (
                f"A model-written {answer_language.lower()} answer was unavailable for "
                f"'{shorten_text(question, 120)}'. {reason} The result below remains an English evidence summary "
                f"grounded in {scope_label}."
            )
        return (
            f"A model-written answer was unavailable for '{shorten_text(question, 120)}'. "
            f"{reason} The result below remains an evidence summary grounded in {scope_label}."
        )
