from scripts.evaluation.explanation_viewer import (
    _containing_phrases,
    aggregate_phrase_evidence,
    character_explanation,
    render_html,
)


def test_containing_phrases_ignores_case_and_drops_symbols():
    assert _containing_phrases("○ GPU 자원", "pu") == ["GPU"]
    assert _containing_phrases("◇ ○", "○") == []


def test_character_explanation_maps_fragment_contributions_to_text():
    explanation = character_explanation(
        "범위는 상호 협의한다", [("상호", 1.0), ("협의", 2.0)]
    )

    assert explanation["fragments"] == [
        {"text": "협의", "contribution": 2.0},
        {"text": "상호", "contribution": 1.0},
    ]
    assert max(explanation["strengths"]) == 1.0
    assert explanation["strengths"][7] > explanation["strengths"][4]


def test_render_html_embeds_records_without_closing_the_script():
    html = render_html(
        [
            {
                "requirement_uid": "demo:R-001",
                "raw_requirement_text": "문장 </script> 확인",
                "gold": "통상수용",
                "explanation_label": "통상수용",
                "explanation_runner_up": "견적반영",
                "explanation_strengths": [0.0] * 15,
                "explanation_fragments": [],
            }
        ]
    )

    assert "RFP 요구사항 설명" in html
    assert "demo:R-001" in html
    assert "<\\/script>" in html
    assert "id=\"requirement-search\"" in html


def test_aggregate_phrase_evidence_expands_ngrams_to_readable_words():
    records = [
        {
            "requirement_uid": "demo:R-001",
            "raw_requirement_text": "범위는 상호 협의하여 결정한다.",
            "explanation_label": "계약·질의검토",
            "explanation_fragments": [
                {"text": "협의하", "contribution": 2.0},
                {"text": "의하여", "contribution": 1.0},
            ],
        }
    ]

    evidence = aggregate_phrase_evidence(records)

    assert evidence["계약·질의검토"][0] == {
        "phrase": "협의하여",
        "total_contribution": 3.0,
        "requirement_count": 1,
    }
