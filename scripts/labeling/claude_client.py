"""Thin Anthropic adapter with structured output and prompt caching."""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Any, Literal

from scripts.labeling.label_schema import LabelResult, SCHEMA_VERSION
from scripts.labeling.llm_token_tracker import TokenUsage

DEFAULT_MODEL = "claude-sonnet-5"
HAIKU_MODEL = "claude-haiku-4-5-20251001"
SUPPORTED_MODELS = {DEFAULT_MODEL, HAIKU_MODEL}
PROMPT_VERSION = "claude-rfp-risk-v1"

SYSTEM_PROMPT = """너는 한국 공공 AI·IT 구축 RFP를 검토하는 15년 차 수석 제안서 작성자다.

입력으로 제공한 요구사항 ID, 요구사항명, 요구사항 내용만 근거로 판단한다. 문서에 없는 사업기간, 예산, 다른 요구사항, 관행을 추측하지 않는다.

primary_action은 다음 세 값 중 하나다.
- 통상수용: 별도 추가 견적이나 계약 질의 없이 일반적인 SI 범위에서 수용 가능
- 견적반영: 부담은 있으나 범위·수량·인력·장비·기간이 명확하여 견적 산정 가능
- 계약·질의검토: 범위·책임·검수 기준이 모호하거나 무상 추가개발, 포괄 책임 등 계약 위험 존재

판단 규칙:
1. reasoning에는 선택한 조치의 핵심 이유를 간결하게 쓴다.
2. evidence에는 입력 본문에서 결정적인 원문 1~3개만 인용한다.
3. 비용이나 범위를 확정할 정보가 부족하면 missing_information.is_missing을 true로 하고 부족한 내용을 구체화한다.
4. domain_dependency에는 전문지식 필요도와 발주처 지원 상태를 기록한다.
5. risk_factors 네 항목은 위험이 없으면 '없음'으로 채운다.
6. requirement_uid는 입력값을 그대로 복사한다.
"""


class ClaudeResponseError(RuntimeError):
    """Raised when Claude returns an unusable terminal response."""


@dataclass(frozen=True)
class ClaudeSettings:
    model: str = DEFAULT_MODEL
    effort: Literal["low", "medium", "high"] = "medium"
    max_tokens: int = 4096
    cache_ttl: Literal["5m", "1h"] = "5m"
    timeout_seconds: float = 120.0
    max_retries: int = 2

    def __post_init__(self) -> None:
        if self.model not in SUPPORTED_MODELS:
            raise ValueError(f"지원하지 않는 Claude 모델: {self.model}")
        if self.max_tokens < 1:
            raise ValueError("max_tokens는 1 이상이어야 합니다.")


@dataclass(frozen=True)
class ClaudeLabelingResult:
    label: LabelResult
    metadata: dict[str, Any]


def _read_usage(response: Any) -> dict[str, int]:
    usage = TokenUsage.from_response(response, provider="anthropic")
    return {
        "input_tokens": usage.input_tokens,
        "output_tokens": usage.output_tokens,
        "total_tokens": usage.total_tokens,
        "cache_creation_input_tokens": usage.cache_creation_tokens,
        "cache_read_input_tokens": usage.cached_tokens,
    }


class ClaudeLabelingClient:
    def __init__(self, settings: ClaudeSettings | None = None, client: Any = None):
        self.settings = settings or ClaudeSettings()
        self._client = client

    def _get_client(self) -> Any:
        if self._client is None:
            api_key = os.getenv("ANTHROPIC_API_KEY")
            if not api_key:
                raise RuntimeError("ANTHROPIC_API_KEY가 설정되지 않았습니다.")
            try:
                from anthropic import Anthropic
            except ImportError as exc:
                raise RuntimeError(
                    "anthropic 패키지가 없습니다. requirements.txt를 설치하세요."
                ) from exc
            self._client = Anthropic(
                api_key=api_key,
                timeout=self.settings.timeout_seconds,
                max_retries=self.settings.max_retries,
            )
        return self._client

    def label_requirement(
        self,
        *,
        requirement_uid: str,
        requirement_name: str,
        requirement_text: str,
    ) -> ClaudeLabelingResult:
        cache_control = {"type": "ephemeral", "ttl": self.settings.cache_ttl}
        request: dict[str, Any] = {
            "model": self.settings.model,
            "max_tokens": self.settings.max_tokens,
            "system": [
                {
                    "type": "text",
                    "text": SYSTEM_PROMPT,
                    "cache_control": cache_control,
                }
            ],
            "messages": [
                {
                    "role": "user",
                    "content": (
                        f"[요구사항 ID]: {requirement_uid}\n"
                        f"[요구사항명]: {requirement_name}\n"
                        f"[요구사항 내용]:\n{requirement_text}"
                    ),
                }
            ],
            "output_format": LabelResult,
        }
        if self.settings.model == DEFAULT_MODEL:
            request["output_config"] = {"effort": self.settings.effort}

        started = time.perf_counter()
        response = self._get_client().messages.parse(**request)
        latency_seconds = time.perf_counter() - started

        if response.stop_reason != "end_turn":
            raise ClaudeResponseError(
                f"완료되지 않은 Claude 응답: stop_reason={response.stop_reason}"
            )
        parsed = getattr(response, "parsed_output", None)
        if parsed is None:
            raise ClaudeResponseError("Claude 구조화 출력이 비어 있습니다.")
        label = parsed if isinstance(parsed, LabelResult) else LabelResult.model_validate(parsed)
        if label.requirement_uid != requirement_uid:
            raise ClaudeResponseError(
                "Claude 응답의 requirement_uid가 입력과 일치하지 않습니다."
            )

        metadata: dict[str, Any] = {
            "provider": "anthropic",
            "model": getattr(response, "model", self.settings.model),
            "request_id": getattr(response, "_request_id", None),
            "stop_reason": response.stop_reason,
            "latency_seconds": round(latency_seconds, 4),
            "schema_version": SCHEMA_VERSION,
            "prompt_version": PROMPT_VERSION,
            "parameters": {
                "effort": (
                    self.settings.effort
                    if self.settings.model == DEFAULT_MODEL
                    else None
                ),
                "max_tokens": self.settings.max_tokens,
                "cache_ttl": self.settings.cache_ttl,
                "timeout_seconds": self.settings.timeout_seconds,
                "max_retries": self.settings.max_retries,
            },
        }
        metadata.update(_read_usage(response))
        return ClaudeLabelingResult(label=label, metadata=metadata)
