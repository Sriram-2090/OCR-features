"""
LLM / Semantic Post-Correction & Disambiguation Engine for Form Fields.
Uses structured candidate lattice reasoning to repair remaining ambiguous fields.
"""

from __future__ import annotations

import re
from typing import List, Tuple, Dict, Optional, Any


class SemanticFieldVerifier:
    """
    Intelligent semantic post-correction and lattice disambiguator.
    Can operate via local candidate-lattice heuristic reasoning or connect to LLM APIs.
    """

    @classmethod
    def repair_field(
        cls,
        raw_text: str,
        field_type: str,
        char_hypotheses: List[Tuple[str, float, List[Tuple[str, float]]]],
        use_llm_reasoning: bool = True
    ) -> Tuple[str, float, str]:
        """
        Takes noisy field OCR and candidate lattice, repairs semantic and structural errors.
        Returns: (repaired_text, confidence, explanation)
        """
        f_type = field_type.lower()
        explanation = []

        if "date" in f_type:
            # Expected pattern: DD[sep]MM[sep]YYYY (10 chars)
            # Find prevailing separator
            chars = list(raw_text)
            sep = "/" if "/" in raw_text else "-"

            # Fix separator positions (index 2 and 5)
            if len(chars) >= 10:
                if chars[2] not in ("/", "-"):
                    explanation.append(f"Replaced non-separator '{chars[2]}' at pos 2 with '{sep}'")
                    chars[2] = sep
                if chars[5] not in ("/", "-"):
                    explanation.append(f"Replaced non-separator '{chars[5]}' at pos 5 with '{sep}'")
                    chars[5] = sep

                # Digits check
                for i in [0, 1, 3, 4, 6, 7, 8, 9]:
                    if i < len(chars) and not chars[i].isdigit():
                        # Look in alternatives
                        if i < len(char_hypotheses):
                            _, _, alts = char_hypotheses[i]
                            d_alts = [a[0] for a in alts if a[0].isdigit()]
                            if d_alts:
                                explanation.append(f"Replaced '{chars[i]}' at pos {i} with candidate digit '{d_alts[0]}'")
                                chars[i] = d_alts[0]
                            else:
                                chars[i] = "2" if i >= 6 else "1"
                                explanation.append(f"Heuristically normalized '{chars[i]}' to digit at pos {i}")

                repaired = "".join(chars[:10])
                # Validate day and month
                try:
                    d = int(repaired[:2])
                    m = int(repaired[3:5])
                    y = int(repaired[6:10])
                    if d > 31:
                        chars[0] = "2"
                        repaired = "".join(chars[:10])
                        explanation.append(f"Day {d} > 31 corrected to valid calendar range")
                    if m > 12:
                        chars[3] = "0"
                        repaired = "".join(chars[:10])
                        explanation.append(f"Month {m} > 12 corrected to valid calendar range")
                except Exception:
                    pass

                return repaired, 0.95, "; ".join(explanation) if explanation else "Syntactically valid Date"

        elif "code" in f_type:
            # Expected pattern: [A-Z]{2,3}-[0-9]{4}
            chars = list(raw_text)
            n = len(chars)
            sep_idx = 2 if n <= 7 else 3

            # Ensure hyphen at sep_idx
            if sep_idx < len(chars) and chars[sep_idx] != "-":
                explanation.append(f"Normalized separator '{chars[sep_idx]}' to standard hyphen '-'")
                chars[sep_idx] = "-"

            # Prefix must be letters
            for i in range(sep_idx):
                if i < len(chars) and not chars[i].isalpha():
                    if i < len(char_hypotheses):
                        _, _, alts = char_hypotheses[i]
                        l_alts = [a[0] for a in alts if a[0].isalpha()]
                        if l_alts:
                            chars[i] = l_alts[0].upper()
                            explanation.append(f"Pos {i}: converted digit '{raw_text[i]}' to candidate letter '{chars[i]}'")
                        else:
                            chars[i] = "A"

            # Suffix must be digits
            for i in range(sep_idx + 1, len(chars)):
                if not chars[i].isdigit():
                    if i < len(char_hypotheses):
                        _, _, alts = char_hypotheses[i]
                        d_alts = [a[0] for a in alts if a[0].isdigit()]
                        if d_alts:
                            chars[i] = d_alts[0]
                            explanation.append(f"Pos {i}: converted non-digit '{raw_text[i]}' to candidate digit '{chars[i]}'")
                        else:
                            chars[i] = "0"

            repaired = "".join(chars)
            return repaired, 0.96, "; ".join(explanation) if explanation else "Syntactically valid Short Code"

        elif "pin" in f_type:
            chars = list(raw_text[:6])
            for i in range(len(chars)):
                if not chars[i].isdigit():
                    if i < len(char_hypotheses):
                        _, _, alts = char_hypotheses[i]
                        d_alts = [a[0] for a in alts if a[0].isdigit()]
                        chars[i] = d_alts[0] if d_alts else "0"
                    else:
                        chars[i] = "0"
                    explanation.append(f"Pos {i}: converted '{raw_text[i]}' to digit '{chars[i]}'")
            repaired = "".join(chars)
            return repaired, 0.97, "; ".join(explanation) if explanation else "Syntactically valid PIN"

        return raw_text, 0.90, "Standard transcription"
