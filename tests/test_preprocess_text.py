import unittest

from scripts.data.preprocess_text import (
    apply_mask,
    choose_longest,
    flatten_list_text,
    make_model_text,
    mask_subjects,
    normalize_bullets,
    normalize_common,
    normalize_endings,
    strip_particles,
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


class MaskingTests(unittest.TestCase):
    """전처리 ablation의 세 규칙. 각각 한 문장으로 정의되는 범위만 건드려야 한다."""

    def test_subject_mask_replaces_the_actor_and_its_particle(self):
        self.assertEqual(
            mask_subjects("제안사는 계약상대자가 발주기관과 협의한다"),
            "<주체> <주체> <주체> 협의한다",
        )

    def test_subject_mask_leaves_the_word_when_it_is_part_of_another_word(self):
        # 건설공사·공사비의 "공사"는 발주기관이 아니라 공사(工事)다.
        self.assertEqual(
            mask_subjects("건설공사 공사비는 공사에서 정한다"),
            "건설공사 공사비는 <주체> 정한다",
        )

    def test_endings_are_reduced_to_the_stem(self):
        self.assertEqual(
            normalize_endings("성능을 제시하여야 하고 일정을 관리하는"),
            "성능을 제시 하고 일정을 관리",
        )

    def test_ending_stays_when_stripping_would_leave_one_letter(self):
        self.assertEqual(normalize_endings("정한다"), "정한다")

    def test_particles_collapse_the_spelling_variants_of_one_word(self):
        self.assertEqual(
            strip_particles("데이터를 데이터는 데이터의 데이터"),
            "데이터 데이터 데이터 데이터",
        )

    def test_particle_stays_when_stripping_would_leave_one_letter(self):
        self.assertEqual(strip_particles("정의 회의"), "정의 회의")

    def test_masking_keeps_line_breaks_and_edge_symbols(self):
        source = "요구사항명\n- 데이터를 수집하여야 한다."
        self.assertEqual(
            strip_particles(source), "요구사항명\n- 데이터 수집하여야 한다."
        )

    def test_apply_mask_is_a_no_op_without_a_rule(self):
        for value in (None, "", "none"):
            self.assertEqual(apply_mask("제안사는 한다", value), "제안사는 한다")

    def test_apply_mask_rejects_an_unknown_rule(self):
        with self.assertRaises(ValueError):
            apply_mask("제안사는 한다", "존재하지않는규칙")

    def test_apply_mask_chains_rules_in_the_order_written(self):
        source = "제안사는 데이터를 제공하여야 한다"
        self.assertEqual(apply_mask(source, "subject"), "<주체> 데이터를 제공하여야 한다")
        self.assertEqual(apply_mask(source, "josa"), "제안사 데이터 제공하여야 한다")
        # 겹쳐 적용하면 주체 치환과 조사 제거가 함께 걸린다.
        self.assertEqual(apply_mask(source, "subject+josa"), "<주체> 데이터 제공하여야 한다")

    def test_chained_mask_rejects_an_unknown_member(self):
        with self.assertRaises(ValueError):
            apply_mask("제안사는 한다", "subject+없는규칙")


if __name__ == "__main__":
    unittest.main()
