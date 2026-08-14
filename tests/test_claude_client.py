from types import SimpleNamespace

import pytest

from scripts.labeling.claude_client import (
    ClaudeLabelingClient,
    ClaudeResponseError,
    ClaudeSettings,
)
from scripts.labeling.label_schema import LabelResult


VALID_LABEL = {
    "requirement_uid": "doc:SFR-001",
    "primary_action": "계약·질의검토",
    "confidence": "높음",
    "reasoning": "검수 기준이 불명확합니다.",
    "evidence": ["세부 기준은 추후 협의한다."],
    "missing_information": {
        "is_missing": True,
        "missing_details": "정량 검수 기준",
    },
    "domain_dependency": {
        "level": "낮음",
        "domain_name": "일반 IT",
        "support_status": "미지정",
    },
    "risk_factors": {
        "cost_driver": "재작업",
        "scope_uncertainty": "검수 범위",
        "responsibility_risk": "수행사 부담",
        "acceptance_risk": "합격 기준 불명확",
    },
}


class FakeMessages:
    def __init__(self, response):
        self.response = response
        self.kwargs = None

    def parse(self, **kwargs):
        self.kwargs = kwargs
        return self.response


def fake_response(stop_reason="end_turn"):
    usage = SimpleNamespace(
        input_tokens=100,
        output_tokens=20,
        cache_creation_input_tokens=80,
        cache_read_input_tokens=0,
    )
    return SimpleNamespace(
        parsed_output=LabelResult.model_validate(VALID_LABEL),
        stop_reason=stop_reason,
        usage=usage,
        _request_id="req_test",
        model="claude-sonnet-5",
    )


def test_sonnet_request_uses_medium_effort_and_explicit_system_cache():
    messages = FakeMessages(fake_response())
    sdk_client = SimpleNamespace(messages=messages)
    client = ClaudeLabelingClient(
        settings=ClaudeSettings(cache_ttl="5m"),
        client=sdk_client,
    )

    result = client.label_requirement(
        requirement_uid="doc:SFR-001",
        requirement_name="검수",
        requirement_text="세부 기준은 추후 협의한다.",
    )

    kwargs = messages.kwargs
    assert kwargs["model"] == "claude-sonnet-5"
    assert kwargs["max_tokens"] == 4096
    assert kwargs["output_config"] == {"effort": "medium"}
    assert kwargs["output_format"] is LabelResult
    assert kwargs["system"][0]["cache_control"] == {
        "type": "ephemeral",
        "ttl": "5m",
    }
    assert "temperature" not in kwargs
    assert "top_p" not in kwargs
    assert "top_k" not in kwargs
    assert result.metadata["cache_creation_input_tokens"] == 80
    assert result.metadata["request_id"] == "req_test"


def test_haiku_request_omits_unsupported_effort():
    response = fake_response()
    response.model = "claude-haiku-4-5-20251001"
    messages = FakeMessages(response)
    client = ClaudeLabelingClient(
        settings=ClaudeSettings(model="claude-haiku-4-5-20251001"),
        client=SimpleNamespace(messages=messages),
    )

    client.label_requirement(
        requirement_uid="doc:SFR-001",
        requirement_name="검수",
        requirement_text="세부 기준은 추후 협의한다.",
    )

    assert "output_config" not in messages.kwargs


def test_non_terminal_response_is_rejected():
    client = ClaudeLabelingClient(
        settings=ClaudeSettings(),
        client=SimpleNamespace(messages=FakeMessages(fake_response("max_tokens"))),
    )

    with pytest.raises(ClaudeResponseError, match="max_tokens"):
        client.label_requirement(
            requirement_uid="doc:SFR-001",
            requirement_name="검수",
            requirement_text="세부 기준은 추후 협의한다.",
        )
