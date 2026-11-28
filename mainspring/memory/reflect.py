"""Rule-based and LLM importance reflection for memory consolidation (V0.3 baseline → V0.4 LLM optional).

Two modes, one interface (``score(content) -> float``):

  Rule  (default, zero-cost): keyword heuristic — preference terms, length,
        first-person pronouns. V0.4 fix: CJK-aware char-count length + first-person
        no longer uses ``\\b`` (which never matches Chinese).  The V0.3-only version
        showed 90% kill-rate on real Chinese tech-chat.

  LLM   (optional, async): passes the candidate fragment to an LLM via the
        framework's ``LLMGateway`` port.  Accurate and language-agnostic, but costs
        one model call per candidate — for a session with 20 candidates that is 20
        calls.

Host chooses at construction time: pass an ``LLMGateway`` to enable LLM mode;
omit it for the free rule engine.
"""
from __future__ import annotations

import logging
import re

from mainspring.ports import LLMGateway

logger = logging.getLogger(__name__)

_PREFERENCE = re.compile(
    r"(喜欢|偏好|想要|希望|认为|觉得|必须|不要|总是|从不|prefer|like|want|should|must|always|never|don't)",
    re.IGNORECASE,
)
# CJK first-person terms *without* \b — they'd never match otherwise since all CJK chars are \w
_FIRST_PERSON = re.compile(r"\b(i|me|my|mine|we|our)\b|我|我们|我的|咱们", re.IGNORECASE)


class Reflector:
    """Scores a memory candidate 0..1.

    Rule mode (no gateway) scores are deterministic and cost nothing.
    LLM mode calls the framework's ``LLMGateway`` for a semantic rating.

    ``score()`` is always ``async`` for uniform calling; rule mode returns immediately.
    """

    def __init__(
        self,
        min_words: int = 4,
        preference_bonus: float = 0.3,
        first_person_bonus: float = 0.15,
        llm_gateway: LLMGateway | None = None,
        llm_model: str | None = None,
    ) -> None:
        self._min_words = min_words
        self._preference_bonus = preference_bonus
        self._first_person_bonus = first_person_bonus
        self._llm = llm_gateway
        self._llm_model = llm_model

    async def score(self, content: str) -> float:
        if self._llm is not None:
            return await self._score_via_llm(content)
        return self._score_rule(content)

    # -- rule engine (V0.3 baseline + CJK fix) ---------------------------------

    def _score_rule(self, content: str) -> float:
        """Rules: base 0.3 + length + preference keywords + first-person.

        CJK-aware length: 4 whitespace words *or* ≥16 CJK characters = "long".
        First-person no longer uses \\b for CJK terms (they never matched).
        """
        text = (content or "").strip()
        if not text:
            return 0.0
        score = 0.3
        words = len(text.split())
        cjk_chars = sum(1 for c in text if "\u4e00" <= c <= "\u9fff")
        if words >= self._min_words or cjk_chars >= 4 * self._min_words:
            score += 0.1
        if _PREFERENCE.search(text):
            score += self._preference_bonus
        if _FIRST_PERSON.search(text):
            score += self._first_person_bonus
        return round(min(1.0, score), 3)

    # -- LLM mode (async, one call per candidate) ------------------------------

    async def _score_via_llm(self, content: str) -> float:
        prompt = (
            "Rate importance of this message as a memory candidate: "
            "0.1=chatty/noise, 0.4=generic work, 0.7=fact/preference, 1.0=critical. "
            "Reply with a single float.\n\nMessage: " + content
        )
        try:
            resp = await self._llm.chat(
                messages=[
                    {"role": "system", "content": "Score memory candidates. Reply float only."},
                    {"role": "user", "content": prompt},
                ],
                model=self._llm_model,
            )
            return round(max(0.0, min(1.0, float((resp.content or "0.3").strip()))), 3)
        except Exception:
            logger.warning("LLM memory scoring failed, falling back to 0.3", exc_info=True)
            return 0.3


__all__ = ["Reflector"]