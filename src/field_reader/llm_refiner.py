"""
FormFlow OCR - Local LLM Post-Correction Engine (Tier 2)
Integrates Qwen 2.5 (7B) via local Ollama daemon for deep semantic OCR correction,
contextual error repair, and visual confusion disambiguation.
"""

from __future__ import annotations

import os
import json
import time
import requests
from typing import Dict, Any, Optional, Tuple

OLLAMA_BASE_URL = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434")
DEFAULT_MODEL = os.environ.get("OLLAMA_OCR_MODEL", "qwen2.5:7b")


class LocalLLMRefiner:
    """
    Tier-2 Semantic LLM Post-Correction Engine.
    Uses local Qwen 2.5 7B to repair complex OCR misrecognitions, broken syntax,
    and ambiguous handwriting tokens.
    """

    def __init__(self, base_url: str = OLLAMA_BASE_URL, model: str = DEFAULT_MODEL):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self._is_available: Optional[bool] = None
        self._last_check_time: float = 0.0

    def check_health(self, force: bool = False) -> Dict[str, Any]:
        """Checks if the local Ollama server is reachable and lists available models."""
        now = time.time()
        if not force and self._is_available is not None and (now - self._last_check_time) < 15.0:
            return {"available": self._is_available, "model": self.model}

        self._last_check_time = now
        try:
            resp = requests.get(f"{self.base_url}/api/tags", timeout=1.5)
            if resp.status_code == 200:
                data = resp.json()
                models = [m.get("name") for m in data.get("models", [])]
                
                # Check if desired model or any compatible model is ready
                matched_model = None
                for m in models:
                    if self.model in m:
                        matched_model = m
                        break
                if not matched_model and models:
                    # Pick first available model
                    matched_model = models[0]

                if matched_model:
                    self.model = matched_model
                self._is_available = True
                return {
                    "available": True,
                    "model": self.model,
                    "models_installed": models,
                    "base_url": self.base_url
                }
        except Exception:
            pass

        self._is_available = False
        return {"available": False, "model": self.model, "error": "Ollama service not running"}

    def refine_ocr(
        self,
        raw_text: str,
        field_type: str = "general",
        confidence: float = 0.0,
        expected_cells: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Submits raw OCR text to local Qwen 2.5 for context-aware post-correction.
        Returns structured dictionary with corrected text, reasoning, and latency.
        """
        start_t = time.perf_counter()

        # Check availability
        health = self.check_health()
        if not health["available"]:
            return {
                "success": False,
                "original_text": raw_text,
                "corrected_text": raw_text,
                "reasoning": "Local LLM daemon offline (using Tier-1 dictionary fallback).",
                "latency_ms": round((time.perf_counter() - start_t) * 1000, 1),
                "source": "fallback"
            }

        # Build prompt based on field type
        f_type_lower = (field_type or "general").lower()

        if "date" in f_type_lower:
            prompt = (
                f"You are a specialized OCR post-correction system for handwritten government and financial forms.\n"
                f"The OCR engine transcribed a date field as: \"{raw_text}\"\n"
                f"Expected format: strictly DD/MM/YYYY or DD-MM-YYYY (valid calendar day 01-31, month 01-12, year 1900-2099).\n"
                f"Common OCR mistakes: 'l', '1', '/' swapped; '0' and 'O'; '2' and 'Z'; '9' and 'g'; extra punctuation.\n"
                f"Respond with a JSON object matching this schema:\n"
                f'{{"corrected_text": "<date>", "reasoning": "<brief 1-sentence explanation of what was fixed>"}}\n'
                f"Output strictly valid JSON with no markdown wrapping or preamble."
            )
        elif "pin" in f_type_lower or "zip" in f_type_lower:
            prompt = (
                f"You are a specialized OCR post-correction system for handwritten postal codes.\n"
                f"The OCR engine transcribed a postal PIN as: \"{raw_text}\"\n"
                f"Expected format: strictly 6 digits (0-9).\n"
                f"Common OCR mistakes: 'O'->'0', 'I'->'1', 'S'->'5', 'B'->'8', 'Z'->'2'.\n"
                f"Respond with a JSON object matching this schema:\n"
                f'{{"corrected_text": "<6 digits>", "reasoning": "<brief 1-sentence explanation>"}}\n'
                f"Output strictly valid JSON with no markdown wrapping or preamble."
            )
        elif "code" in f_type_lower:
            prompt = (
                f"You are a specialized OCR post-correction system for alphanumeric department codes.\n"
                f"The OCR engine transcribed: \"{raw_text}\"\n"
                f"Expected format: 2 or 3 uppercase letters, hyphen, 4 digits (e.g. ABC-1234 or XY-5678).\n"
                f"Respond with a JSON object matching this schema:\n"
                f'{{"corrected_text": "<code>", "reasoning": "<brief explanation>"}}\n'
                f"Output strictly valid JSON with no markdown wrapping or preamble."
            )
        else:
            # Free text / names / sentences
            prompt = (
                f"You are an expert OCR correction system for handwritten English and multilingual text.\n"
                f"The OCR engine transcribed handwritten text with visual noise: \"{raw_text}\"\n"
                f"Fix character confusions (e.g., rn -> m, cl -> d), broken words, and spelling mistakes while strictly preserving the author's intent.\n"
                f"Respond with a JSON object matching this schema:\n"
                f'{{"corrected_text": "<repaired text>", "reasoning": "<brief explanation>"}}\n'
                f"Output strictly valid JSON with no markdown wrapping or preamble."
            )

        try:
            payload = {
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.1,  # low temperature for deterministic correction
                    "num_predict": 120
                }
            }

            resp = requests.post(f"{self.base_url}/api/generate", json=payload, timeout=25.0)
            latency_ms = round((time.perf_counter() - start_t) * 1000, 1)

            if resp.status_code == 200:
                raw_response = resp.json().get("response", "").strip()
                # Clean code blocks
                clean_json = raw_response
                if "```json" in clean_json:
                    clean_json = clean_json.split("```json", 1)[1].split("```", 1)[0].strip()
                elif "```" in clean_json:
                    clean_json = clean_json.split("```", 1)[1].split("```", 1)[0].strip()

                try:
                    parsed = json.loads(clean_json)
                    corrected = parsed.get("corrected_text", raw_text).strip()
                    reasoning = parsed.get("reasoning", "Semantic correction applied by Local Qwen 2.5.")
                    return {
                        "success": True,
                        "original_text": raw_text,
                        "corrected_text": corrected,
                        "reasoning": reasoning,
                        "latency_ms": latency_ms,
                        "source": "llm_qwen2.5",
                        "model": self.model
                    }
                except json.JSONDecodeError:
                    # Clean fallback if raw text was returned directly
                    cleaned_direct = raw_response.replace('"', '').strip()
                    return {
                        "success": True,
                        "original_text": raw_text,
                        "corrected_text": cleaned_direct if len(cleaned_direct) < len(raw_text) * 2 else raw_text,
                        "reasoning": "Corrected via Qwen 2.5 text generation.",
                        "latency_ms": latency_ms,
                        "source": "llm_qwen2.5",
                        "model": self.model
                    }
            else:
                return {
                    "success": False,
                    "original_text": raw_text,
                    "corrected_text": raw_text,
                    "reasoning": f"Ollama HTTP error {resp.status_code}",
                    "latency_ms": round((time.perf_counter() - start_t) * 1000, 1),
                    "source": "fallback"
                }

        except Exception as e:
            return {
                "success": False,
                "original_text": raw_text,
                "corrected_text": raw_text,
                "reasoning": f"Local LLM call error: {str(e)}",
                "latency_ms": round((time.perf_counter() - start_t) * 1000, 1),
                "source": "fallback"
            }


# Singleton refiner instance
_llm_refiner_instance: Optional[LocalLLMRefiner] = None

def get_llm_refiner() -> LocalLLMRefiner:
    global _llm_refiner_instance
    if _llm_refiner_instance is None:
        _llm_refiner_instance = LocalLLMRefiner()
    return _llm_refiner_instance
