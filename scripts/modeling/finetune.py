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
    """요구사항 텍스트를 토큰으로 바꿔 배치로 내주는 최소 구현."""

    def __init__(self, rows: Sequence[dict[str, Any]], tokenizer, max_length: int):
        self.texts = [get_model_text(row) for row in rows]
        self.labels = [LABEL_TO_ID[row["primary_action"]] for row in rows]
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        encoded = self.tokenizer(
            self.texts[index],
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )
        item = {key: value.squeeze(0) for key, value in encoded.items()}
        item["labels"] = torch.tensor(self.labels[index])
        return item


def class_weights(rows: Sequence[dict[str, Any]], device: torch.device) -> torch.Tensor:
    """학습 fold의 분포만으로 balanced 가중치를 만든다.

    TF-IDF 기준선의 `class_weight='balanced'`와 같은 개념이며, 여기서는 손실 함수에
    직접 넣는다. 평가 문서의 분포는 보지 않는다.
    """
    labels = [row["primary_action"] for row in rows]
    present = np.array(sorted(set(labels)), dtype=object)
    weights = dict(
        zip(present.tolist(), compute_class_weight("balanced", classes=present, y=labels))
    )
    return torch.tensor(
        [float(weights.get(label, 1.0)) for label in LABELS], dtype=torch.float, device=device
    )


@torch.no_grad()
def evaluate(model, loader: DataLoader, device: torch.device) -> tuple[float, list[int]]:
    """macro F1과 예측을 돌려준다. `labels`를 명시해 분모를 3으로 고정한다."""
    model.eval()
    gold, pred = [], []
    for batch in loader:
        targets = batch.pop("labels")
        batch = {key: value.to(device) for key, value in batch.items()}
        logits = model(**batch).logits
        pred.extend(logits.argmax(dim=-1).cpu().tolist())
        gold.extend(targets.tolist())
    macro = f1_score(gold, pred, labels=list(range(len(LABELS))), average="macro", zero_division=0)
    return float(macro), pred


@dataclass
class EpochLog:
    epoch: int
    train_loss: float
    validation_macro_f1: float


def train_one_fold(rows, fold, args, device) -> dict[str, Any]:
    """fold 하나를 학습하고, 검증 점수가 가장 좋았던 시점의 가중치로 평가한다."""
    fit_rows, validation_rows, test_rows = fold.split(rows)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model, num_labels=len(LABELS)
    ).to(device)

    def loader(subset, shuffle):
        return DataLoader(
            RequirementDataset(subset, tokenizer, args.max_length),
            batch_size=args.batch_size,
            shuffle=shuffle,
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
    loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights(fit_rows, device))

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
        validation_macro, _ = evaluate(model, validation_loader, device)
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
    test_macro, _ = evaluate(model, test_loader, device)
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
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    if args.mask:
        os.environ[TEXT_MASK_ENV] = args.mask
    device = pick_device()
    set_seed(args.seed)

    rows, meta = load_label_dataset()
    folds = make_lodo_folds(rows)
    selected = folds if args.fold < 0 else [folds[args.fold]]

    print(f"모델 {args.model} | 장치 {device} | seed {args.seed} | 마스킹 {args.mask or '없음'}")
    print(f"데이터 {meta['dataset_version']} {len(rows)}건 | max_length {args.max_length}")

    results = []
    for fold in selected:
        print(f"fold {fold.index} — {fold.test_document}")
        results.append(train_one_fold(rows, fold, args, device))

    if len(results) > 1:
        average = float(np.mean([r["test_macro_f1"] for r in results]))
        print(f"fold 평균 평가 macro F1 {average:.3f}")

    version = os.getenv(DATASET_VERSION_ENV, DEFAULT_DATASET_KEY)
    output = args.output or ROOT / "reports" / "current" / version / "finetune_runs.jsonl"
    output.parent.mkdir(parents=True, exist_ok=True)
    record = {"config": {**vars(args), "output": str(output)}, "device": str(device), "results": results}
    with output.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
    print(f"기록 추가: {output}")


if __name__ == "__main__":
    main()
