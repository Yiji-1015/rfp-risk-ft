#!/usr/bin/env python3
"""Provider-agnostic LLM input/output token tracking utilities."""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

ProviderName = Literal["auto", "openai", "anthropic"]


@dataclass
class TokenUsage:
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    cached_tokens: int = 0
    cache_creation_tokens: int = 0
    thoughts_tokens: int = 0

    @classmethod
    def from_response(cls, response: Any, provider: ProviderName = "auto") -> TokenUsage:
        resolved = provider if provider != "auto" else detect_provider(response)
        extractors = {
            "openai": cls._from_openai,
            "anthropic": cls._from_anthropic,
        }
        extractor = extractors.get(resolved, cls._from_generic)
        return extractor(response)

    @classmethod
    def _from_openai(cls, response: Any) -> TokenUsage:
        usage = getattr(response, "usage", None)
        if usage is None and isinstance(response, dict):
            usage = response.get("usage")
        if usage is None:
            return cls()

        input_tokens = _read_int(usage, "prompt_tokens") or _read_int(usage, "input_tokens")
        output_tokens = (
            _read_int(usage, "completion_tokens") or _read_int(usage, "output_tokens")
        )
        total_tokens = _read_int(usage, "total_tokens") or (input_tokens + output_tokens)
        cached_tokens = _read_int(usage, "prompt_tokens_details", "cached_tokens")
        if not cached_tokens and isinstance(getattr(usage, "prompt_tokens_details", None), dict):
            cached_tokens = int(usage["prompt_tokens_details"].get("cached_tokens") or 0)
        return cls(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
            cached_tokens=cached_tokens,
        )

    @classmethod
    def _from_anthropic(cls, response: Any) -> TokenUsage:
        usage = getattr(response, "usage", None)
        if usage is None and isinstance(response, dict):
            usage = response.get("usage")
        if usage is None:
            return cls()

        input_tokens = _read_int(usage, "input_tokens")
        output_tokens = _read_int(usage, "output_tokens")
        cached_tokens = _read_int(usage, "cache_read_input_tokens")
        cache_creation_tokens = _read_int(usage, "cache_creation_input_tokens")
        return cls(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=(
                input_tokens + output_tokens + cached_tokens + cache_creation_tokens
            ),
            cached_tokens=cached_tokens,
            cache_creation_tokens=cache_creation_tokens,
        )

    @classmethod
    def _from_generic(cls, response: Any) -> TokenUsage:
        for candidate in (
            cls._from_openai(response),
            cls._from_anthropic(response),
        ):
            if candidate.input_tokens or candidate.output_tokens:
                return candidate
        return cls()

    def __add__(self, other: TokenUsage) -> TokenUsage:
        return TokenUsage(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            total_tokens=self.total_tokens + other.total_tokens,
            cached_tokens=self.cached_tokens + other.cached_tokens,
            cache_creation_tokens=(
                self.cache_creation_tokens + other.cache_creation_tokens
            ),
            thoughts_tokens=self.thoughts_tokens + other.thoughts_tokens,
        )


def detect_provider(response: Any) -> str:
    usage = getattr(response, "usage", None)
    if usage is None and isinstance(response, dict):
        usage = response.get("usage")
    if usage is not None:
        if _has_attr(usage, "prompt_tokens") or _has_attr(usage, "completion_tokens"):
            return "openai"
        if _has_attr(usage, "input_tokens") and _has_attr(usage, "output_tokens"):
            return "anthropic"
    return "generic"


def _has_attr(obj: Any, attr: str) -> bool:
    if isinstance(obj, dict):
        return attr in obj
    return getattr(obj, attr, None) is not None


def _read_int(obj: Any, attr: str, nested_attr: str | None = None) -> int:
    if nested_attr is not None:
        nested = getattr(obj, attr, None)
        if nested is None and isinstance(obj, dict):
            nested = obj.get(attr)
        if nested is None:
            return 0
        return _read_int(nested, nested_attr)

    value = getattr(obj, attr, None)
    if value is None and isinstance(obj, dict):
        value = obj.get(attr)
    return int(value or 0)


def _float_env(name: str) -> float:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return 0.0
    return float(raw)


@dataclass
class PricingConfig:
    input_price_per_1m: float = 0.0
    output_price_per_1m: float = 0.0
    currency: str = "KRW"
    input_price_per_1m_usd: float = 0.0
    output_price_per_1m_usd: float = 0.0
    usd_krw_rate: float = 0.0


def load_pricing_from_env(prefix: str = "LLM") -> PricingConfig:
    """Load token pricing from environment variables.

    Preferred (KRW, direct):
      LLM_INPUT_PRICE_PER_1M_KRW
      LLM_OUTPUT_PRICE_PER_1M_KRW

    Alternative (USD + exchange rate -> shown as KRW):
      LLM_INPUT_PRICE_PER_1M_USD
      LLM_OUTPUT_PRICE_PER_1M_USD
      USD_KRW_RATE

    """
    input_krw = _float_env(f"{prefix}_INPUT_PRICE_PER_1M_KRW")
    output_krw = _float_env(f"{prefix}_OUTPUT_PRICE_PER_1M_KRW")
    if input_krw or output_krw:
        return PricingConfig(
            input_price_per_1m=input_krw,
            output_price_per_1m=output_krw,
            currency="KRW",
        )

    input_usd = _float_env(f"{prefix}_INPUT_PRICE_PER_1M_USD") or _float_env(
        f"{prefix}_INPUT_PRICE_PER_1M"
    )
    output_usd = _float_env(f"{prefix}_OUTPUT_PRICE_PER_1M_USD") or _float_env(
        f"{prefix}_OUTPUT_PRICE_PER_1M"
    )
    rate = _float_env("USD_KRW_RATE")
    if rate and (input_usd or output_usd):
        return PricingConfig(
            input_price_per_1m=input_usd * rate,
            output_price_per_1m=output_usd * rate,
            currency="KRW",
            input_price_per_1m_usd=input_usd,
            output_price_per_1m_usd=output_usd,
            usd_krw_rate=rate,
        )

    if input_usd or output_usd:
        return PricingConfig(
            input_price_per_1m=input_usd,
            output_price_per_1m=output_usd,
            currency="USD",
            input_price_per_1m_usd=input_usd,
            output_price_per_1m_usd=output_usd,
        )

    return PricingConfig()


@dataclass
class TokenTracker:
    provider: str = "auto"
    model: str = ""
    calls: int = 0
    totals: TokenUsage = field(default_factory=TokenUsage)
    by_label: dict[str, TokenUsage] = field(default_factory=dict)
    input_price_per_1m: float = 0.0
    output_price_per_1m: float = 0.0
    currency: str = "KRW"
    input_price_per_1m_usd: float = 0.0
    output_price_per_1m_usd: float = 0.0
    usd_krw_rate: float = 0.0

    @classmethod
    def from_pricing(
        cls,
        *,
        provider: str,
        model: str,
        pricing: PricingConfig,
    ) -> TokenTracker:
        return cls(
            provider=provider,
            model=model,
            input_price_per_1m=pricing.input_price_per_1m,
            output_price_per_1m=pricing.output_price_per_1m,
            currency=pricing.currency,
            input_price_per_1m_usd=pricing.input_price_per_1m_usd,
            output_price_per_1m_usd=pricing.output_price_per_1m_usd,
            usd_krw_rate=pricing.usd_krw_rate,
        )

    def record(
        self,
        response_or_usage: Any,
        label: str = "default",
        provider: ProviderName | None = None,
    ) -> TokenUsage:
        if isinstance(response_or_usage, TokenUsage):
            usage = response_or_usage
        else:
            usage = TokenUsage.from_response(
                response_or_usage,
                provider=provider or self.provider,  # type: ignore[arg-type]
            )
        self.calls += 1
        self.totals = self.totals + usage
        self.by_label.setdefault(label, TokenUsage())
        self.by_label[label] = self.by_label[label] + usage
        return usage

    def estimated_cost(self) -> float:
        if not self.input_price_per_1m and not self.output_price_per_1m:
            return 0.0
        input_cost = (self.totals.input_tokens / 1_000_000) * self.input_price_per_1m
        output_cost = (self.totals.output_tokens / 1_000_000) * self.output_price_per_1m
        return input_cost + output_cost

    def estimated_cost_usd(self) -> float:
        if self.currency == "USD":
            return self.estimated_cost()
        if self.usd_krw_rate and (self.input_price_per_1m_usd or self.output_price_per_1m_usd):
            input_cost = (self.totals.input_tokens / 1_000_000) * self.input_price_per_1m_usd
            output_cost = (self.totals.output_tokens / 1_000_000) * self.output_price_per_1m_usd
            return input_cost + output_cost
        return 0.0

    def format_cost(self) -> str:
        cost = self.estimated_cost()
        if cost <= 0:
            return ""
        if self.currency == "KRW":
            return f"{cost:,.0f}원"
        return f"${cost:.4f}"

    def format_usage(self, usage: TokenUsage | None = None) -> str:
        current = usage or self.totals
        return (
            f"input={current.input_tokens:,} "
            f"output={current.output_tokens:,} "
            f"total={current.total_tokens:,}"
        )

    def print_summary(self) -> None:
        print("\n========== LLM 토큰 사용량 요약 ==========")
        print(f"프로바이더: {self.provider}")
        print(f"모델: {self.model}")
        print(f"API 호출 수: {self.calls:,}회")
        print(f"입력 토큰:  {self.totals.input_tokens:,}")
        print(f"출력 토큰:  {self.totals.output_tokens:,}")
        print(f"합계 토큰:  {self.totals.total_tokens:,}")
        if self.totals.cached_tokens:
            print(f"캐시 읽기:  {self.totals.cached_tokens:,}")
        if self.totals.cache_creation_tokens:
            print(f"캐시 생성:  {self.totals.cache_creation_tokens:,}")
        if self.totals.thoughts_tokens:
            print(f"사고 토큰:  {self.totals.thoughts_tokens:,}")
        if self.by_label:
            print("\n[호출 유형별]")
            for label, usage in sorted(self.by_label.items()):
                print(f"  {label}: {self.format_usage(usage)}")
        if self.input_price_per_1m or self.output_price_per_1m:
            print(f"\n예상 비용: {self.format_cost()}")
            if self.currency == "KRW":
                print(
                    f"  (입력 {self.input_price_per_1m:,.2f}원/1M, "
                    f"출력 {self.output_price_per_1m:,.2f}원/1M 기준)"
                )
                usd_cost = self.estimated_cost_usd()
                if usd_cost:
                    print(f"  참고 USD: ${usd_cost:.4f} (환율 {self.usd_krw_rate:,.2f}원)")
            else:
                print(
                    f"  (입력 ${self.input_price_per_1m}/1M, "
                    f"출력 ${self.output_price_per_1m}/1M 기준)"
                )
        else:
            print(
                "\n비용 추정: .env에 LLM_INPUT_PRICE_PER_1M_KRW, "
                "LLM_OUTPUT_PRICE_PER_1M_KRW 설정 시 원화 추정치 표시"
            )
        print("==========================================\n")

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "provider": self.provider,
            "model": self.model,
            "calls": self.calls,
            "input_tokens": self.totals.input_tokens,
            "output_tokens": self.totals.output_tokens,
            "total_tokens": self.totals.total_tokens,
            "cached_tokens": self.totals.cached_tokens,
            "cache_creation_tokens": self.totals.cache_creation_tokens,
            "thoughts_tokens": self.totals.thoughts_tokens,
            "currency": self.currency,
            "estimated_cost": self.estimated_cost(),
            "by_label": {label: asdict(usage) for label, usage in self.by_label.items()},
        }
        if self.currency == "KRW":
            payload["estimated_cost_krw"] = self.estimated_cost()
        usd_cost = self.estimated_cost_usd()
        if usd_cost:
            payload["estimated_cost_usd"] = usd_cost
        return payload

    def save_json(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(self.to_dict(), handle, ensure_ascii=False, indent=2)
