"""Thin Anthropic adapter with structured output and prompt caching."""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Any, Literal, Sequence

from scripts.labeling.label_schema import (
    REASONING_MAX_LENGTH,
    LabelResult,
    SCHEMA_VERSION,
)
from scripts.labeling.llm_token_tracker import TokenUsage

DEFAULT_MODEL = "claude-sonnet-5"
HAIKU_MODEL = "claude-haiku-4-5-20251001"
SUPPORTED_MODELS = {DEFAULT_MODEL, HAIKU_MODEL}
PROMPT_VERSION = "claude-rfp-risk-v5"

SYSTEM_PROMPT = f"""너는 한국 공공 AI·IT 구축 RFP를 검토하는 15년 차 수석 제안서 작성자다.

[기준 수행사]
기본적인 LLM·RAG 구축 역량을 가진 AI·IT 회사의 제안 담당자로서 판단한다.
- 일반적인 분석·설계·개발·테스트는 기본 수행팀이 처리한다
- 고급 기술자, 전문 인력, 별도 장비·라이선스가 필요하면 견적에 명시적으로 반영한다
- 일반적인 ISP·ISMP와 AI 구축 컨설팅 경험은 보유한다
- 발주기관의 업무 도메인 지식은 없다

[판단 순서]
1. blocker가 있는가 → 있으면 계약·질의검토
2. 없다면, 기본 수행팀 범위를 넘는 원가가 붙는가 → 붙으면 견적반영
3. 둘 다 아니면 통상수용

[blocker]
blocker는 비용이 발생한다는 뜻이 아니다. **제안·입찰 전에 반드시 확인해야 안전하게 수용할 수 있는 조건**을 뜻한다.
- 범위·책임: 열린 범위, 포괄 책임, 발주기관 재량에 따른 추가 수행, 무제한 의무
- 검수·성능기준: 명시적으로 무제한 재작업·합격할 때까지 반복·무상 보완이 요구되거나, 달성 가능성을 먼저 확인해야 하는 구체적 수치 목표(예: 응답속도 N배 개선, 정확도 N% 이상, 동시접속 N명)가 제시된 경우
- 기술실현성: 현재 기술이나 주어진 환경에서 충족 가능한지 먼저 검증해야 하는 조건
- 라이선스·공급: 요구 제품·모델·장비·라이선스의 실제 조달 또는 사용 가능성이 불명확한 조건
- 공급자종속: 과도하게 구체적인 규격·실적·호환 조건 때문에 충족 가능한 공급자가 제한될 수 있는 조건

[표준 문구 보정]
공공 RFP에는 관행적으로 쓰이는 품질·호환·가용성 문구가 있다. 이런 문구는 그 자체로 blocker가 아니며 기본 수행팀이 통상적으로 처리한다.
- blocker 아님: "기존 시스템 운영에 영향을 주지 않도록", "무중단 운영", "정상 동작 보장", "사용자 테스트를 통해 평가", "미흡사항 조치", "품질 목표 기준 통과", "표준·지침 준수"
- blocker 검토 대상: 달성 가능성을 먼저 확인해야 하는 구체적 수치 목표, 대상이 특정되지 않았거나 생소해서 연계 범위를 산정할 수 없는 시스템, 명시적인 무제한·무상 재작업 조건, 조달 가능성이 불확실한 제품·모델

**기준이 상세히 적혀 있지 않다는 사실만으로 blocker를 부여하지 않는다.** 공공 RFP는 원래 그 수준으로 쓰이며, 그것을 위험으로 읽으면 거의 모든 요구사항이 blocker가 된다.

해당하는 것을 모두 나열한다. 없으면 빈 배열로 둔다.
공급자종속은 특정 업체가 내정되었다고 단정하지 않는다. 요구사항에서 관찰되는 경쟁 제한 가능성만 기록한다.

[cost_basis]
기본 수행팀 범위를 넘어 추가 원가가 무엇으로 계산되는가. 추가 원가가 없으면 '없음'
- 없음 / 고급·전문인력 / 장비·인프라 / 라이선스 / 외부인증 / 외주·전문기관 / 복합

[보조 축]
domain_dependency: 발주기관 고유 업무 지식이 없으면 수행이 막히는 정도 (높음/보통/낮음)
build_difficulty: 발주기관이 업무 지식·데이터·정답 기준을 모두 제공한다고 가정할 때의 순수 구축 난이도 (높음/보통/낮음)

두 축은 독립적으로 판단한다. 도메인 어려움을 build_difficulty에 섞지 않는다.
**난이도가 높다는 사실만으로 주 라벨을 올리지 않는다.** 높은 난이도가 고급 기술자나 전문 인력의 추가 투입으로 이어질 때만 견적반영이다.

[컨설팅 요구사항 보정]
기술명이 등장한다는 이유만으로 난이도를 높이지 않는다. 구조의 정의·설계만 요구하면 통상 컨설팅으로 보고, 실제 구현·인증·운영·성능 책임이 함께 요구될 때 별도 원가나 blocker를 검토한다.
- 통상수용 예: 비전·KPI·전략과제 수립, 현황·업무프로세스 분석, 연동 가능성 검토, 일반적인 조직·정책·운영절차 설계, 표준 아키텍처와 연계 원칙 정의
- 견적반영 예: AI 모델 평가, GPU·클라우드·보안 아키텍처, 복잡한 데이터 설계, FP·M/M 및 예산 산정처럼 고급·복수 전문인력 투입이 요구사항에 드러남
- 계약·질의검토 예: 인증 대상·비용·기간·합격 기준이 없는 실제 외부 인증 취득, 계약 후 세부 범위 확정, 기술 실현성·라이선스·공급 가능성 확인 필요

[판단 규칙]
1. 입력으로 제공한 요구사항만 근거로 판단한다. 문서에 없는 사업기간, 예산, 다른 요구사항을 추측하지 않는다. 다른 요구사항이나 문서 전체를 봐야 알 수 있는 불확실성은 주 라벨에 반영하지 않는다.
2. 금액을 추정해서 쓰지 않는다. 단가를 모르기 때문이다.
3. 부담이 크다는 것과 계산이 불가능하다는 것은 다른 사실이다. 부담이 커도 계산되면 견적반영이다.
4. 라벨 분포를 맞추기 위해 판정을 바꾸지 않는다.
5. reasoning은 1~2문장, 공백 포함 {REASONING_MAX_LENGTH}자 이내. blocker 유무와 원가 발생 여부를 중심으로 쓴다.
6. requirement_uid는 입력값을 그대로 복사한다.
"""

# 앵커 블록의 기본 위치는 user 메시지다. 동적 인출은 입력마다 앵커가 달라져서
# system에 넣으면 캐시 프리픽스가 매 건 깨지고, system이 전략과 무관하게 동일해야
# zero-shot과 few-shot이 통제 비교가 되기 때문이다(결정 18).
# 앵커가 고정된 전략만 예외로 system 블록에 실어 캐시한다(결정 29).
ANCHOR_BLOCK_VERSION = "anchor-block-v1"
# 고정 앵커는 유사도로 뽑은 것이 아니므로 인출 근거 줄을 붙이지 않는다. 붙이면
# 사실과 다를 뿐 아니라 대상마다 값이 달라져 캐시 프리픽스가 깨진다(결정 29).
CONSTANT_ANCHOR_BLOCK_VERSION = "anchor-block-const-v1"

ANCHOR_BLOCK_HEADER = """[참고 사례]
아래는 다른 기관 RFP에서 이미 검토가 끝난 요구사항과 그 판정이다. 판정 기준의 눈높이를 맞추는 용도로만 쓴다.
사례와 문구가 비슷해도 제공 주체, 무상 범위, 수량 상한, 검수 기준, 책임 범위가 다르면 판정은 달라야 한다.
사례의 판정을 그대로 따라가지 말고, 아래 대상 요구사항의 원문에 근거해 판단한다.
"""

CONSTANT_ANCHOR_BLOCK_HEADER = """[판정 기준 사례]
아래 세 사례는 모든 요구사항에 동일하게 제시되는 눈금이다. 대상과 유사해서 고른 것이 아니다.
세 라벨이 각각 어느 수준에서 갈리는지만 참고하고, 판정은 대상 요구사항의 원문에 근거해 내린다.
"""


class ClaudeResponseError(RuntimeError):
    """Raised when Claude returns an unusable terminal response."""


@dataclass(frozen=True)
class ClaudeSettings:
    model: str = DEFAULT_MODEL
    effort: Literal["low", "medium", "high", "xhigh", "max"] = "medium"
    thinking: Literal["adaptive", "disabled"] = "adaptive"
    max_tokens: int = 16000
    cache_ttl: Literal["5m", "1h"] = "5m"
    timeout_seconds: float = 120.0
    max_retries: int = 2

    def __post_init__(self) -> None:
        if self.model not in SUPPORTED_MODELS:
            raise ValueError(f"지원하지 않는 Claude 모델: {self.model}")
        if self.max_tokens < 1:
            raise ValueError("max_tokens는 1 이상이어야 합니다.")

    @property
    def supports_thinking_and_effort(self) -> bool:
        """Sonnet 5만 adaptive thinking과 effort를 받는다. Haiku 4.5는 둘 다 거부한다."""
        return self.model == DEFAULT_MODEL


@dataclass(frozen=True)
class ClaudeLabelingResult:
    label: LabelResult
    metadata: dict[str, Any]


def render_anchor_block(
    anchors: Sequence[dict[str, Any]],
    *,
    show_retrieval_evidence: bool = True,
) -> str:
    """결정 12: 앵커와 함께 인출 근거(유사도·공통 어휘)를 프롬프트에 노출한다.

    :param show_retrieval_evidence: 인출 근거 줄을 붙일지 여부. 고정 앵커는 검색으로
        뽑은 것이 아니라 유사도·공통 어휘가 의미 없고, 대상마다 값이 달라지면
        캐시 프리픽스도 깨지므로 False로 둔다(결정 29).
    """
    header = ANCHOR_BLOCK_HEADER if show_retrieval_evidence else CONSTANT_ANCHOR_BLOCK_HEADER
    lines = [header]
    for order, anchor in enumerate(anchors, 1):
        if show_retrieval_evidence:
            overlap = ", ".join(anchor.get("overlap_terms") or []) or "없음"
            caption = f"사례 {order} (유사도 {anchor.get('similarity', 0):.3f} / 공통 어휘: {overlap})"
        else:
            caption = f"사례 {order} (판정 {anchor.get('primary_action', '')})"
        lines.append(
            f"\n{caption}\n"
            f"요구사항명: {anchor.get('requirement_name', '')}\n"
            f"내용: {anchor.get('raw_requirement_text', '')}\n"
            f"판정: {anchor.get('primary_action', '')}\n"
            f"이유: {anchor.get('reasoning', '')}"
        )
    return "\n".join(lines)


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
        anchors: Sequence[dict[str, Any]] | None = None,
        cache_anchors: bool = False,
    ) -> ClaudeLabelingResult:
        """
        :param cache_anchors: 앵커를 캐시되는 system 블록에 넣을지 여부.

            기본값은 False다. 동적 인출(유사도·층화)은 입력마다 앵커가 달라지므로
            system에 넣으면 매 건 캐시 프리픽스가 깨진다(결정 18). 앵커가 고정된
            전략에서만 True로 두면 앵커 블록이 캐시되어 입력 비용의 90%가 빠진다.
        """
        cache_control = {"type": "ephemeral", "ttl": self.settings.cache_ttl}
        target_block = (
            f"[요구사항 ID]: {requirement_uid}\n"
            f"[요구사항명]: {requirement_name}\n"
            f"[요구사항 내용]:\n{requirement_text}"
        )
        use_system_anchors = bool(anchors) and cache_anchors
        if use_system_anchors:
            user_content = f"[대상 요구사항]\n{target_block}"
        elif anchors:
            user_content = f"{render_anchor_block(anchors)}\n\n[대상 요구사항]\n{target_block}"
        else:
            user_content = target_block
        # 브레이크포인트를 둘로 나눈다. 기본 프롬프트 블록은 전략과 무관하게 동일하므로
        # zero-shot 실행과 캐시를 공유하고, 앵커 블록은 그 뒤에서 따로 캐시된다.
        system_blocks: list[dict[str, Any]] = [
            {"type": "text", "text": SYSTEM_PROMPT, "cache_control": cache_control}
        ]
        if use_system_anchors:
            system_blocks.append(
                {
                    "type": "text",
                    "text": render_anchor_block(anchors, show_retrieval_evidence=False),
                    "cache_control": cache_control,
                }
            )
        request: dict[str, Any] = {
            "model": self.settings.model,
            "max_tokens": self.settings.max_tokens,
            "system": system_blocks,
            "messages": [{"role": "user", "content": user_content}],
            "output_format": LabelResult,
        }
        if self.settings.supports_thinking_and_effort:
            request["output_config"] = {"effort": self.settings.effort}
            # Sonnet 5는 thinking을 생략하면 adaptive로 켜진다. 실행 조건을 기록에 남기려면
            # 켜든 끄든 명시해야 한다(§11.15). max_tokens는 사고와 응답을 합쳐서 제한한다.
            request["thinking"] = {"type": self.settings.thinking}

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
            # 무엇이 어긋났는지 남겨야 일회성 흔들림과 프롬프트 결함을 구분할 수 있다.
            raise ClaudeResponseError(
                "Claude 응답의 requirement_uid가 입력과 일치하지 않습니다: "
                f"입력={requirement_uid!r} 응답={label.requirement_uid!r} "
                f"판정={label.primary_action!r}"
            )

        metadata: dict[str, Any] = {
            "provider": "anthropic",
            "model": getattr(response, "model", self.settings.model),
            "request_id": getattr(response, "_request_id", None),
            "stop_reason": response.stop_reason,
            "latency_seconds": round(latency_seconds, 4),
            "schema_version": SCHEMA_VERSION,
            "prompt_version": PROMPT_VERSION,
            "anchor_count": len(anchors or ()),
            "anchor_block_version": (
                None
                if not anchors
                else CONSTANT_ANCHOR_BLOCK_VERSION
                if use_system_anchors
                else ANCHOR_BLOCK_VERSION
            ),
            "anchors_cached_in_system": use_system_anchors,
            "parameters": {
                "effort": (
                    self.settings.effort
                    if self.settings.supports_thinking_and_effort
                    else None
                ),
                "thinking": (
                    self.settings.thinking
                    if self.settings.supports_thinking_and_effort
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
