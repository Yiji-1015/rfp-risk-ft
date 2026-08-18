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
    "blockers": ["범위·책임"],
    "cost_basis": "고급·전문인력",
    "domain_dependency": "낮음",
    "build_difficulty": "보통",
    "reasoning": "'세부 기준은 추후 협의한다'로만 적혀 있어 합격 기준이 확정되지 않았다.",
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
    assert kwargs["max_tokens"] == 16000
    assert kwargs["output_config"] == {"effort": "medium"}
    assert kwargs["thinking"] == {"type": "adaptive"}
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


def test_haiku_request_omits_unsupported_effort_and_thinking():
    """Haiku 4.5는 effort와 adaptive thinking을 모두 거부한다."""
    response = fake_response()
    response.model = "claude-haiku-4-5-20251001"
    messages = FakeMessages(response)
    client = ClaudeLabelingClient(
        settings=ClaudeSettings(model="claude-haiku-4-5-20251001"),
        client=SimpleNamespace(messages=messages),
    )

    result = client.label_requirement(
        requirement_uid="doc:SFR-001",
        requirement_name="검수",
        requirement_text="세부 기준은 추후 협의한다.",
    )

    assert "output_config" not in messages.kwargs
    assert "thinking" not in messages.kwargs
    assert result.metadata["parameters"]["thinking"] is None
    assert result.metadata["parameters"]["effort"] is None


def test_thinking_is_always_sent_explicitly_on_sonnet():
    """
    Sonnet 5는 thinking을 생략하면 adaptive로 켜진다. 우연히 켜진 상태로
    유료 실행에 들어가지 않도록, 끌 때도 명시해서 보낸다.
    """
    messages = FakeMessages(fake_response())
    client = ClaudeLabelingClient(
        settings=ClaudeSettings(thinking="disabled"),
        client=SimpleNamespace(messages=messages),
    )

    result = client.label_requirement(
        requirement_uid="doc:SFR-001",
        requirement_name="검수",
        requirement_text="세부 기준은 추후 협의한다.",
    )

    assert messages.kwargs["thinking"] == {"type": "disabled"}
    assert result.metadata["parameters"]["thinking"] == "disabled"


ANCHORS = [
    {
        "requirement_uid": "doc_b:R-1",
        "document_id": "doc_b",
        "requirement_name": "무상 추가개발",
        "raw_requirement_text": "검수 후 요구 기능은 무상으로 추가 개발한다.",
        "primary_action": "계약·질의검토",
        "reasoning": "무상 범위의 상한이 없다.",
        "similarity": 0.4213,
        "overlap_terms": ["무상", "추가 개발"],
    }
]


def test_zero_shot_user_message_has_no_anchor_block():
    messages = FakeMessages(fake_response())
    client = ClaudeLabelingClient(client=SimpleNamespace(messages=messages))

    result = client.label_requirement(
        requirement_uid="doc:SFR-001",
        requirement_name="검수",
        requirement_text="세부 기준은 추후 협의한다.",
    )

    content = messages.kwargs["messages"][0]["content"]
    assert content.startswith("[요구사항 ID]")
    assert "참고 사례" not in content
    assert result.metadata["anchor_count"] == 0
    assert result.metadata["anchor_block_version"] is None


def test_anchors_go_into_the_user_message_not_the_cached_system_block():
    """
    앵커는 입력마다 달라진다. 캐시되는 system 블록에 넣으면 매 건 캐시가 깨지고,
    system이 전략별로 달라지면 zero-shot과의 통제 비교도 성립하지 않는다.
    """
    zero_shot_messages = FakeMessages(fake_response())
    ClaudeLabelingClient(
        client=SimpleNamespace(messages=zero_shot_messages)
    ).label_requirement(
        requirement_uid="doc:SFR-001",
        requirement_name="검수",
        requirement_text="세부 기준은 추후 협의한다.",
    )

    fewshot_messages = FakeMessages(fake_response())
    result = ClaudeLabelingClient(
        client=SimpleNamespace(messages=fewshot_messages)
    ).label_requirement(
        requirement_uid="doc:SFR-001",
        requirement_name="검수",
        requirement_text="세부 기준은 추후 협의한다.",
        anchors=ANCHORS,
    )

    assert fewshot_messages.kwargs["system"] == zero_shot_messages.kwargs["system"]

    content = fewshot_messages.kwargs["messages"][0]["content"]
    assert "참고 사례" in content
    assert "계약·질의검토" in content
    assert "무상 범위의 상한이 없다." in content
    assert "0.421" in content  # 결정 12: 인출 근거를 프롬프트에 노출
    assert "무상, 추가 개발" in content
    assert content.index("참고 사례") < content.index("[대상 요구사항]")
    assert result.metadata["anchor_count"] == 1
    assert result.metadata["anchor_block_version"] == "anchor-block-v1"


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
