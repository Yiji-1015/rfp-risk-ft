#!/usr/bin/env python3
"""sLLM(디코더)을 QLoRA로 파인튜닝해 주 라벨을 판정한다.

`finetune.py`(인코더 + 새 분류 헤드)와 다른 점 세 가지.
- 라벨을 **단어로 생성**한다. 헤드를 새로 만들지 않고, 모델이 이미 아는 라벨 뜻을 쓴다.
- 입력이 프롬프트라서 **라벨 정의를 같이 준다**. 인코더에는 규칙을 줄 자리가 없었다.
- 판정은 자유 생성이 아니라 **세 라벨의 로그확률 비교**다. 파싱 실패가 없고 출력은
  항상 셋 중 하나다. 학습도 라벨 토큰에서만 loss를 건다.

fold 분할·평가·기록 형식은 `finetune.py`와 같다(13문서 LODO, `finetune_runs.jsonl`).
`--smoke`로 수십 건만 돌려 코드 경로를 먼저 확인하고, `--fold 0`으로 한 fold를 본 뒤
`--fold -1`로 넓힌다.

    RFP_DATASET_VERSION=v5 python -m scripts.modeling.finetune_llm --fold 0
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score
from transformers import AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup

from scripts.evaluation.folds import make_lodo_folds
from scripts.labeling.label_dataset import DATASET_VERSION_ENV, DEFAULT_DATASET_KEY, get_model_text, load_label_dataset
from scripts.modeling.finetune import LABELS, class_weights, pick_device, set_seed
from scripts.modeling.run_id import run_id

ROOT = Path(__file__).resolve().parents[2]

# 라벨 정의. 라벨링 프롬프트(v5)의 판정 규칙을 줄인 것이며, 학습·평가 전 건에 똑같이 붙는다.
SYSTEM = """당신은 공공기관 AI·IT 사업 제안요청서(RFP)의 요구사항 조항을 검토하는 제안 실무자다.
조항 하나를 읽고 다음 셋 중 하나로 판정한다.
- 통상수용: 표준 기술·일반 관행으로 별도 원가나 협상 없이 받아들이는 조항. 상투적 보안·품질·문서 조항 포함.
- 견적반영: 추가 원가(인력, 장비, 라이선스, 기간, 외부 용역)가 계산되는 조항. 원가만 잡히면 계약 조건 문제는 아니다.
- 계약·질의검토: 계약 전에 확인·질의가 필요한 조항. 특정 벤더·제품 지정이나 단일 공급자, 발주기관이 정한 검수·성능 기준 수치, 구현 범위·책임이 열려 있음, 법·규제·외부기관 승인, 무제한 의무.
판정 라벨만 답한다."""


def build_prompt(tokenizer, text: str, max_length: int) -> list[int]:
    ids = tokenizer(text, add_special_tokens=False, truncation=True, max_length=max_length).input_ids
    clause = tokenizer.decode(ids)
    messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": f"[조항]\n{clause}\n\n판정:"}]
    text = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    return list(tokenizer(text, add_special_tokens=False).input_ids)


def label_ids(tokenizer) -> list[list[int]]:
    return [tokenizer(label, add_special_tokens=False).input_ids + [tokenizer.eos_token_id] for label in LABELS]


def pad_batch(seqs: Sequence[list[int]], pad_id: int, device) -> tuple[torch.Tensor, torch.Tensor]:
    width = max(len(s) for s in seqs)
    ids = torch.full((len(seqs), width), pad_id, dtype=torch.long)
    mask = torch.zeros((len(seqs), width), dtype=torch.long)
    for i, s in enumerate(seqs):
        ids[i, : len(s)] = torch.tensor(s)
        mask[i, : len(s)] = 1
    return ids.to(device), mask.to(device)


@torch.no_grad()
def predict(model, prompts: Sequence[list[int]], cands: list[list[int]], pad_id: int, device, batch_size: int) -> list[int]:
    """건마다 세 라벨의 로그확률 합을 비교한다. 출력은 항상 LABELS의 인덱스다."""
    model.eval()
    out: list[int] = []
    for start in range(0, len(prompts), batch_size):
        chunk = prompts[start : start + batch_size]
        seqs = [p + c for p in chunk for c in cands]
        ids, mask = pad_batch(seqs, pad_id, device)
        logits = model(input_ids=ids, attention_mask=mask).logits
        scores = []
        for i, seq in enumerate(seqs):
            lp = len(chunk[i // len(cands)])
            cand = cands[i % len(cands)]
            step = logits[i, lp - 1 : lp - 1 + len(cand)].float().log_softmax(-1)
            scores.append(step.gather(1, torch.tensor(cand, device=device).unsqueeze(1)).sum().item())
        for i in range(len(chunk)):
            block = scores[i * len(cands) : (i + 1) * len(cands)]
            out.append(int(np.argmax(block)))
    return out


def macro(gold: Sequence[int], pred: Sequence[int]) -> float:
    return float(f1_score(gold, pred, labels=list(range(len(LABELS))), average="macro", zero_division=0))


def patch_embeddings(model) -> None:
    """trust_remote_code 모델(EXAONE 3.5)이 transformers 5.x에서 `get_input_embeddings`를
    구현하지 않아 peft가 죽는다. 어휘 크기의 Embedding 층을 찾아 직접 답하게 한다."""
    try:
        model.get_input_embeddings()
        return
    except NotImplementedError:
        pass
    embedding = max((m for m in model.modules() if isinstance(m, torch.nn.Embedding)), key=lambda m: m.num_embeddings)
    inner = getattr(model, getattr(model, "base_model_prefix", ""), model)
    for cls in {type(model), type(inner)}:
        cls.get_input_embeddings = lambda self, _e=embedding: _e
    if not hasattr(type(model), "get_output_embeddings") or model.get_output_embeddings() is None:
        head = getattr(model, "lm_head", None)
        if head is not None:
            type(model).get_output_embeddings = lambda self, _h=head: _h
    print(f"  임베딩 층 패치: {type(embedding).__name__}({embedding.num_embeddings})")


def load_model(args, device):
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

    kwargs: dict[str, Any] = {"trust_remote_code": True}
    if args.no_quant:
        kwargs["torch_dtype"] = torch.bfloat16 if device.type == "cuda" else torch.float32
    else:
        from transformers import BitsAndBytesConfig

        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        kwargs["device_map"] = {"": 0}
    model = AutoModelForCausalLM.from_pretrained(args.model, **kwargs)
    patch_embeddings(model)
    if args.no_quant:
        model.to(device)
    if not args.no_quant:
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    else:
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()
    model.config.use_cache = False
    config = LoraConfig(r=args.lora_r, lora_alpha=args.lora_r * 2, lora_dropout=0.05,
                        target_modules="all-linear", task_type="CAUSAL_LM")
    return get_peft_model(model, config)


def train_one_fold(rows, fold, args, device) -> dict[str, Any]:
    from peft import get_peft_model_state_dict, set_peft_model_state_dict

    fit_rows, validation_rows, test_rows = fold.split(rows)
    if args.smoke:
        fit_rows, validation_rows, test_rows = fit_rows[:32], validation_rows[:12], test_rows[:12]
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    cands = label_ids(tokenizer)
    model = load_model(args, device)
    model.print_trainable_parameters()

    def encode(subset):
        prompts = [build_prompt(tokenizer, get_model_text(r), args.max_length) for r in subset]
        golds = [LABELS.index(r["primary_action"]) for r in subset]
        return prompts, golds

    fit_prompts, fit_gold = encode(fit_rows)
    val_prompts, val_gold = encode(validation_rows)
    test_prompts, test_gold = encode(test_rows)
    weights = class_weights(fit_rows, device)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0)
    steps_per_epoch = math.ceil(math.ceil(len(fit_rows) / args.batch_size) / args.grad_accum)
    total_steps = steps_per_epoch * args.epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, int(total_steps * args.warmup_ratio), total_steps)
    print(f"  학습 {len(fit_rows)} / 검증 {len(validation_rows)} / 평가 {len(test_rows)}  ({fold.test_document}), {total_steps} step")

    rng = np.random.default_rng(args.seed + fold.index)
    history, best_score, best_state = [], -1.0, None
    for epoch in range(1, args.epochs + 1):
        model.train()
        order = rng.permutation(len(fit_rows))
        running, batches = 0.0, 0
        optimizer.zero_grad()
        for b, start in enumerate(range(0, len(order), args.batch_size), start=1):
            idx = order[start : start + args.batch_size]
            seqs = [fit_prompts[i] + cands[fit_gold[i]] for i in idx]
            ids, mask = pad_batch(seqs, pad_id, device)
            targets = torch.full_like(ids, -100)
            for j, i in enumerate(idx):
                lp = len(fit_prompts[i])
                targets[j, lp : lp + len(cands[fit_gold[i]])] = torch.tensor(cands[fit_gold[i]], device=device)
            logits = model(input_ids=ids, attention_mask=mask).logits[:, :-1]
            tgt = targets[:, 1:]
            sel = tgt != -100
            tok_loss = F.cross_entropy(logits[sel].float(), tgt[sel], reduction="none")
            ex = torch.arange(len(idx), device=device).unsqueeze(1).expand_as(sel)[sel]
            per_ex = torch.zeros(len(idx), device=device).index_add_(0, ex, tok_loss) / sel.sum(1).clamp(min=1)
            w = weights[torch.tensor([fit_gold[i] for i in idx], device=device)]
            loss = (per_ex * w).sum() / w.sum()
            running += loss.item(); batches += 1
            (loss / args.grad_accum).backward()
            if b % args.grad_accum == 0 or start + args.batch_size >= len(order):
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                optimizer.step(); scheduler.step(); optimizer.zero_grad()
        val_pred = predict(model, val_prompts, cands, pad_id, device, args.eval_batch_size)
        val_macro = macro(val_gold, val_pred)
        history.append({"epoch": epoch, "train_loss": running / max(batches, 1), "validation_macro_f1": val_macro})
        marker = ""
        if val_macro > best_score:
            best_score = val_macro
            best_state = {k: v.detach().cpu().clone() for k, v in get_peft_model_state_dict(model).items()}
            marker = "  ← 최고"
        print(f"  epoch {epoch}/{args.epochs}  학습 loss {running / max(batches, 1):.4f}  검증 macro F1 {val_macro:.3f}{marker}")

    set_peft_model_state_dict(model, best_state)
    test_pred = predict(model, test_prompts, cands, pad_id, device, args.eval_batch_size)
    test_macro = macro(test_gold, test_pred)
    print(f"  평가 문서 macro F1 {test_macro:.3f} (검증 최고 {best_score:.3f} 시점)")
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return {
        "fold_index": fold.index, "test_document": fold.test_document,
        "train_size": len(fit_rows), "validation_size": len(validation_rows), "test_size": len(test_rows),
        "best_validation_macro_f1": best_score, "test_macro_f1": test_macro, "history": history,
        "predictions": [{"requirement_uid": r["requirement_uid"], "gold": r["primary_action"], "pred": LABELS[p]}
                        for r, p in zip(test_rows, test_pred)],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct")
    parser.add_argument("--fold", type=int, default=0, help="-1이면 전체 fold")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--eval-batch-size", type=int, default=4, help="건 수. 실제 시퀀스는 3배(라벨 후보마다 하나)")
    parser.add_argument("--max-length", type=int, default=512, help="조항 본문 토큰 상한(프롬프트 제외)")
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-quant", action="store_true", help="4비트 양자화 없이 로드(CPU 연기 테스트용)")
    parser.add_argument("--smoke", action="store_true", help="fold당 수십 건만 써서 코드 경로 확인")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    device = pick_device()
    set_seed(args.seed)
    rows, meta = load_label_dataset()
    folds = make_lodo_folds(rows)
    selected = folds if args.fold < 0 else [folds[args.fold]]
    print(f"모델 {args.model} | 장치 {device} | seed {args.seed} | {'양자화 없음' if args.no_quant else 'QLoRA 4bit'} | r={args.lora_r}")
    print(f"데이터 {meta['dataset_version']} {len(rows)}건 | max_length {args.max_length}")

    results = [train_one_fold(rows, fold, args, device) for fold in selected]
    if len(results) > 1:
        print(f"fold 평균 평가 macro F1 {np.mean([r['test_macro_f1'] for r in results]):.3f}")
        gold = [p["gold"] for r in results for p in r["predictions"]]
        pred = [p["pred"] for r in results for p in r["predictions"]]
        print(f"통합 OOF macro F1 {f1_score(gold, pred, labels=list(LABELS), average='macro', zero_division=0):.3f} ({len(gold)}건)")

    if args.smoke:
        print("연기 테스트라 기록하지 않는다.")
        return
    version = os.getenv(DATASET_VERSION_ENV, DEFAULT_DATASET_KEY)
    output = args.output or ROOT / "reports" / "current" / version / "finetune_runs.jsonl"
    output.parent.mkdir(parents=True, exist_ok=True)
    # `tag`는 앙상블 멤버 이름(`llm42`)이 된다. mask/binary는 인코더 기록과 형식을 맞추기 위한 자리다.
    config = {**vars(args), "output": str(output), "mask": None, "binary": False, "tag": "llm", "system_prompt": SYSTEM}
    record = {"run_id": run_id(config, version), "config": config, "device": str(device), "results": results}
    with output.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
    print(f"기록 추가: {output}  (run_id={record['run_id']})")


if __name__ == "__main__":
    main()
