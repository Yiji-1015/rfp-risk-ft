import numpy as np
import torch

from scripts.modeling.finetune import (
    BINARY_LABELS,
    LABELS,
    LABEL_TO_ID,
    class_weights,
    collapse,
    make_collator,
    pick_device,
    set_seed,
)

CPU = torch.device("cpu")


def _rows(counts: dict[str, int]) -> list[dict]:
    return [{"primary_action": label} for label, n in counts.items() for _ in range(n)]


def test_label_order_matches_the_project_wide_order() -> None:
    assert LABELS == ("통상수용", "견적반영", "계약·질의검토")
    assert [LABEL_TO_ID[label] for label in LABELS] == [0, 1, 2]


def test_class_weights_favour_the_minority_class() -> None:
    weights = class_weights(_rows({"통상수용": 60, "견적반영": 20, "계약·질의검토": 20}), CPU)

    assert weights.shape == (3,)
    # balanced 가중치는 n / (클래스 수 x 해당 건수)다.
    assert torch.allclose(weights, torch.tensor([100 / 180, 100 / 60, 100 / 60]), atol=1e-6)
    assert weights[0] < weights[1]


def test_missing_class_keeps_a_neutral_weight() -> None:
    # 학습 fold에 한 클래스가 없어도 길이 3을 유지해야 손실 함수가 깨지지 않는다.
    weights = class_weights(_rows({"통상수용": 10, "견적반영": 10}), CPU)

    assert weights.shape == (3,)
    assert weights[2] == 1.0


def test_seed_makes_the_run_repeatable() -> None:
    set_seed(7)
    first = (torch.randn(4).tolist(), np.random.rand(4).tolist())
    set_seed(7)
    second = (torch.randn(4).tolist(), np.random.rand(4).tolist())

    assert first == second


def test_device_choice_falls_back_to_cpu() -> None:
    assert pick_device().type in {"cuda", "xpu", "cpu"}


def test_binary_collapse_keeps_only_the_accept_class_apart() -> None:
    assert collapse("통상수용") == "통상수용"
    assert collapse("견적반영") == "검토필요"
    assert collapse("계약·질의검토") == "검토필요"
    assert BINARY_LABELS == ("통상수용", "검토필요")


def test_binary_class_weights_use_the_collapsed_distribution() -> None:
    rows = _rows({"통상수용": 50, "견적반영": 25, "계약·질의검토": 25})

    weights = class_weights(rows, CPU, BINARY_LABELS)

    # 접으면 50 대 50이라 가중치가 사라진다. 3분류에서는 소수 클래스가 밀린다.
    assert weights.shape == (2,)
    assert torch.allclose(weights, torch.tensor([1.0, 1.0]), atol=1e-6)
    assert class_weights(rows, CPU)[1] > 1.0


class _StubTokenizer:
    """`pad`만 흉내 낸다. 배치 최장 길이에 맞추고 마스크를 만든다."""

    pad_token_id = 0

    def pad(self, items, return_tensors=None):
        width = max(len(item["input_ids"]) for item in items)
        return {
            "input_ids": torch.tensor(
                [item["input_ids"] + [0] * (width - len(item["input_ids"])) for item in items]
            ),
            "attention_mask": torch.tensor(
                [[1] * len(item["input_ids"]) + [0] * (width - len(item["input_ids"])) for item in items]
            ),
        }


def test_collator_pads_to_the_longest_item_in_the_batch() -> None:
    batch = [
        {"input_ids": [1, 2, 3], "labels": 0},
        {"input_ids": [4, 5, 6, 7, 8], "labels": 2},
    ]

    result = make_collator(_StubTokenizer())(batch)

    # 배치 최장이 5이므로 5로 맞춘다. 고정 길이(max_length)로 늘리지 않는다.
    assert result["input_ids"].shape == (2, 5)
    assert result["attention_mask"][0].tolist() == [1, 1, 1, 0, 0]
    assert result["labels"].tolist() == [0, 2]
