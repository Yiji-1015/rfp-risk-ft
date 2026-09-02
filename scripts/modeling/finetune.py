#!/usr/bin/env python3
"""경량 한국어 인코더를 주 라벨 하나로 파인튜닝한다.

§9.4가 요구하는 부품을 직접 구성한다 — `Dataset`/`DataLoader`, 학습·검증 루프,
optimizer와 scheduler, class weight, best-checkpoint 선택, seed 고정.

fold 하나만 돌리는 것이 기본값이다. 처음부터 10 fold를 돌리면 무엇이 잘못됐는지 알 수
없다. 먼저 한 fold에서 학습 loss가 내려가는지, 검증 점수가 언제 꺾이는지를 눈으로
확인하고 나서 전체로 넓힌다.

TF-IDF 기준선과 달리 인코더는 입력을 **자른다**(`--max-length`). 924건의 토큰 길이는
중앙값 165, 95%가 580이라 512로도 6.8%가 잘린다. 잘린 만큼 인코더가 불리한 비교이므로
길이를 바꿀 때마다 함께 적는다.
"""

from __future__ import annotations

import argparse
import json
import os
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from sklearn.metrics import f1_score
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

from scripts.evaluation.folds import make_lodo_folds
from scripts.labeling.label_dataset import (
    DATASET_VERSION_ENV,
    DEFAULT_DATASET_KEY,
    TEXT_MASK_ENV,
    get_model_text,
    load_label_dataset,
)

ROOT = Path(__file__).resolve().parents[2]
LABELS = ("통상수용", "견적반영", "계약·질의검토")
LABEL_TO_ID = {label: index for index, label in enumerate(LABELS)}

# `--binary`는 `견적반영`과 `계약·질의검토`를 합친다. TF-IDF에서 3분류 0.638이 2분류
# 0.788로 오른 것이 전적으로 그 경계에서 나왔으므로, 파인튜닝도 같은 축으로 재본다.
ACCEPT, REVIEW = "통상수용", "검토필요"
BINARY_LABELS = (ACCEPT, REVIEW)


def collapse(label: str) -> str:
    return ACCEPT if label == ACCEPT else REVIEW


def pick_device() -> torch.device:
    """CUDA → XPU → CPU 순으로 고른다. gcube 컨테이너와 로컬에서 같은 코드가 돈다."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return torch.device("xpu")
    return torch.device("cpu")


def set_seed(seed: int) -> None:
    """재현을 위해 파이썬·numpy·torch의 난수를 함께 고정한다.

    그래도 파인튜닝은 seed마다 결과가 흔들린다. 하나의 점수를 주장하려면 seed를 여러 개
    돌려 편차를 함께 보고해야 한다.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class RequirementDataset(Dataset):
    """요구사항 텍스트를 토큰으로 바꿔 배치로 내주는 최소 구현.

    패딩은 여기서 하지 않고 배치를 만들 때 그 배치의 최장 길이에 맞춘다. 모든 건을
    `max_length`까지 채우면 중앙값 165토큰짜리 데이터를 4,096까지 늘리게 되어 긴 입력
    모델을 감당할 수 없다. `attention_mask`가 패딩을 가리므로 결과는 달라지지 않는다.
    """

    def __init__(self, rows, tokenizer, max_length: int, labels: Sequence[str]):
        collapsed = len(labels) == 2
        self.texts = [get_model_text(row) for row in rows]
        self.labels = [
            labels.index(collapse(row["primary_action"]) if collapsed else row["primary_action"])
            for row in rows
        ]
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, index: int) -> dict[str, Any]:
        encoded = self.tokenizer(
            self.texts[index], truncation=True, max_length=self.max_length
        )
        return {**encoded, "labels": self.labels[index]}


def make_collator(tokenizer):
    """배치 안에서만 패딩한다. 토크나이저가 패딩 토큰과 마스크를 함께 만든다."""

    def collate(batch: Sequence[dict[str, Any]]) -> dict[str, torch.Tensor]:
        labels = torch.tensor([item["labels"] for item in batch])
        padded = tokenizer.pad(
            [{k: v for k, v in item.items() if k != "labels"} for item in batch],
            return_tensors="pt",
        )
        return {**padded, "labels": labels}

    return collate


def class_weights(rows, device: torch.device, labels: Sequence[str] = LABELS) -> torch.Tensor:
    """학습 fold의 분포만으로 balanced 가중치를 만든다.

    TF-IDF 기준선의 `class_weight='balanced'`와 같은 개념이며, 여기서는 손실 함수에
    직접 넣는다. 평가 문서의 분포는 보지 않는다.
    """
    collapsed = len(labels) == 2
    observed = [collapse(row["primary_action"]) if collapsed else row["primary_action"] for row in rows]
    present = np.array(sorted(set(observed)), dtype=object)
    weights = dict(
        zip(present.tolist(), compute_class_weight("balanced", classes=present, y=observed))
    )
    return torch.tensor(
        [float(weights.get(label, 1.0)) for label in labels], dtype=torch.float, device=device
    )


@torch.no_grad()
def evaluate(model, loader: DataLoader, device: torch.device, label_count: int = len(LABELS)):
    """macro F1과 예측을 돌려준다. `labels`를 명시해 분모를 클래스 수로 고정한다."""
    model.eval()
    gold, pred = [], []
    for batch in loader:
        targets = batch.pop("labels")
        batch = {key: value.to(device) for key, value in batch.items()}
        logits = model(**batch).logits
        pred.extend(logits.argmax(dim=-1).cpu().tolist())
        gold.extend(targets.tolist())
    macro = f1_score(gold, pred, labels=list(range(label_count)), average="macro", zero_division=0)
    return float(macro), pred


@dataclass
class EpochLog:
    epoch: int
    train_loss: float
    validation_macro_f1: float


def train_one_fold(rows, fold, args, device) -> dict[str, Any]:
    """fold 하나를 학습하고, 검증 점수가 가장 좋았던 시점의 가중치로 평가한다."""
    labels = BINARY_LABELS if args.binary else LABELS
    fit_rows, validation_rows, test_rows = fold.split(rows)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model, num_labels=len(labels)
    ).to(device)
    collate = make_collator(tokenizer)

    def loader(subset, shuffle):
        return DataLoader(
            RequirementDataset(subset, tokenizer, args.max_length, labels),
            batch_size=args.batch_size,
            shuffle=shuffle,
            collate_fn=collate,
        )

    train_loader = loader(fit_rows, True)
    validation_loader = loader(validation_rows, False)
    test_loader = loader(test_rows, False)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    # 갱신 횟수는 배치 수가 아니라 누적 후 optimizer가 실제로 걸음을 뗀 횟수다.
    steps_per_epoch = -(-len(train_loader) // args.grad_accum)
    total_steps = steps_per_epoch * args.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, int(total_steps * args.warmup_ratio), total_steps
    )
    loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights(fit_rows, device, labels))

    history: list[EpochLog] = []
    best_score, best_state = -1.0, None
    print(
        f"  학습 {len(fit_rows)} / 검증 {len(validation_rows)} / 평가 {len(test_rows)}"
        f"  ({fold.test_document}), {total_steps} step"
        f"  (배치 {args.batch_size} x 누적 {args.grad_accum} = 유효 {args.batch_size * args.grad_accum})"
    )
    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        optimizer.zero_grad()
        for index, batch in enumerate(train_loader, start=1):
            targets = batch.pop("labels").to(device)
            batch = {key: value.to(device) for key, value in batch.items()}
            loss = loss_fn(model(**batch).logits, targets)
            running += loss.item()
            # 누적 구간의 평균이 되도록 나눠서 역전파한다. 그래야 누적 횟수를 바꿔도
            # 기울기 크기가 유지되고 learning rate를 다시 잡지 않아도 된다.
            (loss / args.grad_accum).backward()
            if index % args.grad_accum == 0 or index == len(train_loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

        train_loss = running / len(train_loader)
        validation_macro, _ = evaluate(model, validation_loader, device, len(labels))
        history.append(EpochLog(epoch, train_loss, validation_macro))
        marker = ""
        if validation_macro > best_score:
            best_score = validation_macro
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            marker = "  ← 최고"
        print(
            f"  epoch {epoch}/{args.epochs}  학습 loss {train_loss:.4f}"
            f"  검증 macro F1 {validation_macro:.3f}{marker}"
        )

    # 평가 문서는 학습 내내 보지 않는다. 검증으로 고른 시점의 가중치로 한 번만 본다.
    model.load_state_dict(best_state)
    test_macro, predictions = evaluate(model, test_loader, device, len(labels))
    print(f"  평가 문서 macro F1 {test_macro:.3f} (검증 최고 {best_score:.3f} 시점)")

    return {
        "fold_index": fold.index,
        "test_document": fold.test_document,
        "train_size": len(fit_rows),
        "validation_size": len(validation_rows),
        "test_size": len(test_rows),
        "best_validation_macro_f1": best_score,
        "test_macro_f1": test_macro,
        "history": [asdict(entry) for entry in history],
        # 건별 예측을 남긴다. 점수 하나로는 어느 라벨에서 막히는지, TF-IDF와 같은
        # 경계에서 틀리는지를 볼 수 없다. `DataLoader(shuffle=False)`라 순서가 보존된다.
        "predictions": [
            {
                "requirement_uid": row["requirement_uid"],
                "gold": collapse(row["primary_action"]) if args.binary else row["primary_action"],
                "pred": labels[index],
            }
            for row, index in zip(test_rows, predictions)
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="klue/roberta-small")
    parser.add_argument("--fold", type=int, default=0, help="-1이면 10 fold 전체")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument(
        "--grad-accum",
        type=int,
        default=1,
        help="기울기를 몇 배치 모았다 갱신할지. 유효 batch = batch-size x grad-accum",
    )
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mask", default=None, help="입력 마스킹 (예: subject+ending+josa)")
    parser.add_argument(
        "--binary",
        action="store_true",
        help="견적반영과 계약·질의검토를 검토필요로 합쳐 2분류로 학습한다",
    )
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    if args.mask:
        os.environ[TEXT_MASK_ENV] = args.mask
    device = pick_device()
    set_seed(args.seed)

    rows, meta = load_label_dataset()
    folds = make_lodo_folds(rows)
    selected = folds if args.fold < 0 else [folds[args.fold]]

    print(f"모델 {args.model} | 장치 {device} | seed {args.seed} | 마스킹 {args.mask or '없음'}"
          f" | {'2분류' if args.binary else '3분류'}")
    print(f"데이터 {meta['dataset_version']} {len(rows)}건 | max_length {args.max_length}")

    results = []
    for fold in selected:
        print(f"fold {fold.index} — {fold.test_document}")
        results.append(train_one_fold(rows, fold, args, device))

    if len(results) > 1:
        average = float(np.mean([r["test_macro_f1"] for r in results]))
        print(f"fold 평균 평가 macro F1 {average:.3f}")
        gold = [p["gold"] for r in results for p in r["predictions"]]
        pred = [p["pred"] for r in results for p in r["predictions"]]
        names = list(BINARY_LABELS if args.binary else LABELS)
        pooled = f1_score(gold, pred, labels=names, average="macro", zero_division=0)
        print(f"통합 OOF macro F1 {pooled:.3f} ({len(gold)}건)")
        for label, value in zip(
            names, f1_score(gold, pred, labels=names, average=None, zero_division=0)
        ):
            print(f"  {label:<12} F1 {value:.3f}")

    version = os.getenv(DATASET_VERSION_ENV, DEFAULT_DATASET_KEY)
    output = args.output or ROOT / "reports" / "current" / version / "finetune_runs.jsonl"
    output.parent.mkdir(parents=True, exist_ok=True)
    record = {"config": {**vars(args), "output": str(output)}, "device": str(device), "results": results}
    with output.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
    print(f"기록 추가: {output}")


if __name__ == "__main__":
    main()
