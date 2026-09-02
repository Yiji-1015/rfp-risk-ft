import numpy as np
import torch

from scripts.modeling.finetune import LABELS, LABEL_TO_ID, class_weights, pick_device, set_seed

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
