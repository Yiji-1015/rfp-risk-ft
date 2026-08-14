"""Tests for provider-agnostic LLM token tracking utilities."""

import os
from unittest.mock import patch

from scripts.labeling.llm_token_tracker import (
    TokenTracker,
    TokenUsage,
    load_pricing_from_env,
)


class _GeminiUsageMeta:
    prompt_token_count = 1200
    candidates_token_count = 350
    total_token_count = 1550


class _GeminiResponse:
    usage_metadata = _GeminiUsageMeta()


class _OpenAIUsage:
    prompt_tokens = 900
    completion_tokens = 120
    total_tokens = 1020


class _OpenAIResponse:
    usage = _OpenAIUsage()


class _AnthropicUsage:
    input_tokens = 500
    output_tokens = 80


class _AnthropicResponse:
    usage = _AnthropicUsage()


def test_token_usage_from_gemini_response():
    usage = TokenUsage.from_response(_GeminiResponse(), provider="gemini")
    assert usage.input_tokens == 1200
    assert usage.output_tokens == 350


def test_token_usage_from_openai_response():
    usage = TokenUsage.from_response(_OpenAIResponse(), provider="openai")
    assert usage.input_tokens == 900
    assert usage.output_tokens == 120


def test_token_usage_from_anthropic_response():
    usage = TokenUsage.from_response(_AnthropicResponse(), provider="anthropic")
    assert usage.input_tokens == 500
    assert usage.output_tokens == 80


def test_token_tracker_krw_cost():
    tracker = TokenTracker(
        provider="openai",
        model="gpt-4.1-mini",
        input_price_per_1m=150.0,
        output_price_per_1m=600.0,
        currency="KRW",
    )
    tracker.record(_OpenAIResponse(), label="labeling")

    assert tracker.calls == 1
    cost = tracker.estimated_cost()
    expected = (900 / 1_000_000) * 150 + (120 / 1_000_000) * 600
    assert abs(cost - expected) < 1e-9
    assert tracker.format_cost() == f"{cost:,.0f}원"


def test_load_pricing_krw_from_env():
    env = {
        "LLM_INPUT_PRICE_PER_1M_KRW": "200",
        "LLM_OUTPUT_PRICE_PER_1M_KRW": "800",
    }
    with patch.dict(os.environ, env, clear=False):
        pricing = load_pricing_from_env()
    assert pricing.currency == "KRW"
    assert pricing.input_price_per_1m == 200.0
    assert pricing.output_price_per_1m == 800.0


def test_load_pricing_usd_with_rate_converts_to_krw():
    env = {
        "LLM_INPUT_PRICE_PER_1M_USD": "0.10",
        "LLM_OUTPUT_PRICE_PER_1M_USD": "0.40",
        "USD_KRW_RATE": "1400",
    }
    with patch.dict(os.environ, env, clear=True):
        pricing = load_pricing_from_env()
    assert pricing.currency == "KRW"
    assert pricing.input_price_per_1m == 140.0
    assert pricing.output_price_per_1m == 560.0
