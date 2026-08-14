import unittest

from scripts.data.preprocess_text import (
    choose_longest,
    flatten_list_text,
    make_model_text,
    normalize_bullets,
    normalize_common,
)


class PreprocessTextTests(unittest.TestCase):
    def test_normalizes_common_bullet_symbols(self):
        source = "◦ 첫 항목\n○ 둘째 항목\n■ 셋째 항목\n- 하위 항목"
        self.assertEqual(
            normalize_bullets(source),
            "- 첫 항목\n- 둘째 항목\n- 셋째 항목\n- 하위 항목",
        )

    def test_does_not_change_inline_hyphens(self):
        source = "AI-플랫폼 구축 및 1-2년 지원"
        self.assertEqual(normalize_bullets(source), source)

    def test_common_normalization_handles_unicode_and_html_breaks(self):
        source = "A\u0301  항목<br>다음\r\n줄"
        self.assertEqual(normalize_common(source), "Á 항목\n다음\n줄")

    def test_flat_variant_keeps_content_and_removes_list_boundaries(self):
        source = "◦ 첫 항목\n- 둘째 항목"
        self.assertEqual(flatten_list_text(source), "첫 항목 둘째 항목")

    def test_model_text_includes_requirement_name(self):
        self.assertEqual(
            make_model_text("AI 대화형 서비스", "◦ 실시간 응답", "normalized-list"),
            "AI 대화형 서비스\n- 실시간 응답",
        )

    def test_choose_longest_uses_body_length(self):
        records = [
            {"requirement_uid": "doc:A", "raw_requirement_text": "짧음"},
            {"requirement_uid": "doc:B", "raw_requirement_text": "가장 긴 본문"},
        ]
        self.assertEqual(choose_longest(records)[0]["requirement_uid"], "doc:B")


if __name__ == "__main__":
    unittest.main()
