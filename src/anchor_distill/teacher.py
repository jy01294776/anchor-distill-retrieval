from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any, cast

from openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam
from openai.types.shared_params import ResponseFormatJSONSchema

from anchor_distill.config import Settings
from anchor_distill.schemas import (
    TeacherRecord,
    TokenAlternative,
    TokenPosition,
)

RUBRIC_ID = "banking77_pair_relevance_v1"
RUBRIC = """You evaluate whether a candidate support intent is relevant to a
customer query. Return one integer relevance score:
1 = unrelated
2 = weakly related
3 = ambiguous or partially related
4 = strongly related
5 = exact intent match
Judge only the supplied query and candidate intent."""

LabelDistribution = tuple[float, float, float, float, float]

RESPONSE_FORMAT = cast(
    ResponseFormatJSONSchema,
    {
        "type": "json_schema",
        "json_schema": {
            "name": "pair_relevance",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    "relevance": {
                        "type": "integer",
                        "enum": [1, 2, 3, 4, 5],
                    }
                },
                "required": ["relevance"],
                "additionalProperties": False,
            },
        },
    },
)


@dataclass(frozen=True)
class TeacherProbeResult:
    passed: bool
    model_requested: str
    model_resolved: str | None
    selected_rating: int | None
    recognized_mass: float
    accepted: bool
    system_fingerprint: str | None
    error: str | None = None


def prompt_hash() -> str:
    payload = {
        "rubric_id": RUBRIC_ID,
        "rubric": RUBRIC,
        "response_format": RESPONSE_FORMAT,
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def build_messages(query: str, anchor: str) -> list[ChatCompletionMessageParam]:
    return [
        cast(
            ChatCompletionMessageParam,
            {"role": "developer", "content": RUBRIC},
        ),
        cast(
            ChatCompletionMessageParam,
            {
                "role": "user",
                "content": (
                    f"Customer query:\n{query}\n\n"
                    f"Candidate intent:\n{anchor}\n\n"
                    "Return the relevance score."
                ),
            },
        ),
    ]


def _five(values: Iterable[float]) -> LabelDistribution:
    materialized = tuple(float(value) for value in values)
    if len(materialized) != 5:
        raise ValueError("expected exactly five label probabilities")
    return (
        materialized[0],
        materialized[1],
        materialized[2],
        materialized[3],
        materialized[4],
    )


def _token_text(token: str, raw_bytes: list[int] | None) -> str:
    if raw_bytes:
        try:
            return bytes(raw_bytes).decode("utf-8")
        except (UnicodeDecodeError, ValueError):
            pass
    return token


def _rating_from_token(token: str, raw_bytes: list[int] | None) -> int | None:
    text = _token_text(token, raw_bytes).strip().strip("\"'")
    return int(text) if text in {"1", "2", "3", "4", "5"} else None


def extract_soft_distribution(
    positions: Iterable[TokenPosition],
    *,
    selected_rating: int,
    minimum_mass: float,
) -> tuple[LabelDistribution, float, float, bool]:
    candidate_position: TokenPosition | None = None
    for position in positions:
        if _rating_from_token(position.token, position.bytes) == selected_rating:
            candidate_position = position
    if candidate_position is None:
        hard = _five(1.0 if index == selected_rating else 0.0 for index in range(1, 6))
        return hard, 0.0, 0.0, False

    masses = [0.0] * 5
    alternatives = list(candidate_position.top_logprobs)
    alternatives.append(
        TokenAlternative(
            token=candidate_position.token,
            logprob=candidate_position.logprob,
            bytes=candidate_position.bytes,
        )
    )
    for alternative in alternatives:
        label = _rating_from_token(alternative.token, alternative.bytes)
        if label is None:
            continue
        probability = math.exp(min(0.0, alternative.logprob))
        masses[label - 1] = max(masses[label - 1], probability)
    recognized_mass = min(1.0, sum(masses))
    if recognized_mass <= 0:
        hard = _five(1.0 if index == selected_rating else 0.0 for index in range(1, 6))
        return hard, 0.0, 0.0, False
    normalized = _five(value / recognized_mass for value in masses)
    entropy = -sum(value * math.log(value) for value in normalized if value > 0)
    accepted = recognized_mass >= minimum_mass
    return normalized, recognized_mass, entropy, accepted


def _positions_from_choice(choice: Any) -> list[TokenPosition]:
    if choice.logprobs is None or choice.logprobs.content is None:
        return []
    positions: list[TokenPosition] = []
    for item in choice.logprobs.content:
        alternatives = [
            TokenAlternative(
                token=alternative.token,
                logprob=alternative.logprob,
                bytes=alternative.bytes,
            )
            for alternative in (item.top_logprobs or [])
        ]
        positions.append(
            TokenPosition(
                token=item.token,
                logprob=item.logprob,
                bytes=item.bytes,
                top_logprobs=alternatives,
            )
        )
    return positions


class TeacherClient:
    def __init__(
        self,
        settings: Settings | None = None,
        *,
        client: OpenAI | None = None,
    ) -> None:
        self.settings = settings or Settings()
        if client is None and not self.settings.openai_api_key:
            raise ValueError("OPENAI_API_KEY is required for teacher requests")
        self.client = client or OpenAI(api_key=self.settings.openai_api_key)

    def score_pair(
        self,
        *,
        query_id: str,
        query: str,
        anchor_id: str,
        anchor: str,
    ) -> TeacherRecord:
        response = self.client.chat.completions.create(
            model=self.settings.teacher_model,
            messages=build_messages(query, anchor),
            response_format=RESPONSE_FORMAT,
            logprobs=True,
            top_logprobs=20,
            temperature=0,
        )
        choice = response.choices[0]
        refusal = bool(getattr(choice.message, "refusal", None))
        if refusal:
            raise ValueError("teacher refused a benign relevance probe")
        content = choice.message.content or ""
        parsed = json.loads(content)
        selected = int(parsed["relevance"])
        distribution, mass, entropy, accepted = extract_soft_distribution(
            _positions_from_choice(choice),
            selected_rating=selected,
            minimum_mass=self.settings.teacher_minimum_label_mass,
        )
        return TeacherRecord(
            query_id=query_id,
            anchor_id=anchor_id,
            rubric_id=RUBRIC_ID,
            prompt_hash=prompt_hash(),
            model_requested=self.settings.teacher_model,
            model_resolved=response.model,
            system_fingerprint=response.system_fingerprint,
            selected_rating=selected,
            label_probabilities=distribution,
            recognized_mass=mass,
            entropy=entropy,
            accepted=accepted,
            refusal=False,
            request_id=response.id,
        )

    def probe(self) -> TeacherProbeResult:
        try:
            record = self.score_pair(
                query_id="synthetic-query",
                query="My card has not arrived after two weeks.",
                anchor_id="synthetic-anchor",
                anchor="card delivery tracking",
            )
            return TeacherProbeResult(
                passed=record.accepted,
                model_requested=record.model_requested,
                model_resolved=record.model_resolved,
                selected_rating=record.selected_rating,
                recognized_mass=record.recognized_mass,
                accepted=record.accepted,
                system_fingerprint=record.system_fingerprint,
            )
        except Exception as exc:  # probe is a diagnostic boundary
            return TeacherProbeResult(
                passed=False,
                model_requested=self.settings.teacher_model,
                model_resolved=None,
                selected_rating=None,
                recognized_mass=0.0,
                accepted=False,
                system_fingerprint=None,
                error=f"{type(exc).__name__}: {exc}",
            )


def build_batch_request(
    *,
    custom_id: str,
    model: str,
    query: str,
    anchor: str,
) -> dict[str, Any]:
    return {
        "custom_id": custom_id,
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": {
            "model": model,
            "messages": build_messages(query, anchor),
            "response_format": RESPONSE_FORMAT,
            "logprobs": True,
            "top_logprobs": 20,
            "temperature": 0,
        },
    }
