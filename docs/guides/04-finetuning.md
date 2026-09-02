# 가이드 04 — 파인튜닝은 무엇이 다르고, 어떻게 돌리는가

> 학습용 요약본이다. 시간순 기록은 [`../history/`](../history/)와 [`../WORKLOG.md`](../WORKLOG.md)에,
> 실행 절차는 [`../PIPELINE.md`](../PIPELINE.md)에 있다. **숫자가 어긋나면 그쪽이 정본이다.**
>
> [가이드 02](02-model-comparison.md)가 "무엇을 더해봤고 왜 채택하지 않았는가"라면,
> 이 문서는 **로드맵(§9)의 마지막 칸**인 경량 인코더 파인튜닝을 다룬다.
> 코드는 [`scripts/modeling/finetune.py`](../../scripts/modeling/finetune.py)다.

## 1. TF-IDF 기준선과 무엇이 다른가

| | TF-IDF + Logistic | 파인튜닝 |
|---|---|---|
| 텍스트 → 숫자 | **고정 규칙** (문자 n-gram 세기) | **학습됨** (인코더가 표현을 바꿔감) |
| 학습되는 것 | 계수 벡터 3개 | 인코더 전체 + 새 분류 헤드 |
| 학습 방식 | `fit()` 한 번 | 데이터를 여러 바퀴, 배치로 나눠서 |
| 입력 길이 | 제한 없음 | **잘린다** (`--max-length`) |
| 같은 코드 재실행 | 같은 결과 | **다른 결과** (랜덤 초기화·셔플) |

마지막 두 줄이 보고 방식을 바꾼다.

**입력이 잘린다.** 924건의 토큰 길이는 중앙값 165, 90%가 444, 95%가 580, 최대 2,261이다.
`max_length=256`이면 71.6%, `512`면 93.2%만 온전히 들어간다. RoBERTa 계열은 512가 한계다.
TF-IDF는 본문 전체를 보므로 **이 비교는 인코더가 불리한 조건에서 출발한다.** 결과를 적을
때 "인코더가 졌다"가 아니라 "**N토큰으로 자른 인코더가** 졌다"로 적는다.

**seed마다 결과가 다르다.** 점수 하나를 주장하려면 seed를 여러 개 돌려 편차를 함께 낸다.
`--seed`를 바꿔 최소 3회 돌리고 범위를 보고한다.

## 2. 부품 여섯 개

§9.4가 직접 구성하라고 정한 목록이며, 코드에서 각각 한 곳에 모여 있다.

| 부품 | 위치 | 메모 |
|---|---|---|
| `Dataset` / `DataLoader` | `RequirementDataset` | 텍스트 → 토큰 → 배치. 입력은 `get_model_text()`라 마스킹 스위치가 그대로 걸린다 |
| class weight | `class_weights()` | TF-IDF의 `class_weight='balanced'`와 같은 개념. **학습 fold 분포만** 보고 손실 함수에 넣는다 |
| optimizer + scheduler | `train_one_fold()` | AdamW + linear warmup. gradient clipping 1.0 |
| 학습·검증 루프 | `train_one_fold()` | epoch마다 학습 loss와 검증 macro F1을 남긴다 |
| best-checkpoint | `train_one_fold()` | 검증 최고 시점의 가중치를 보관하고, **평가 문서는 그 시점으로 딱 한 번** 본다 |
| seed / 장치 | `set_seed()`, `pick_device()` | `cuda → xpu → cpu` 자동 감지 |

**평가 문서는 학습 내내 보지 않는다.** 멈출 시점은 검증 문서(8/1/1의 그 1)로만 고른다.
§9.3이 이 자리를 파인튜닝을 위해 비워둔 것이다.

## 3. 실행

### 로컬 — 연기 테스트용

```powershell
$env:RFP_DATASET_VERSION='v4'
python -m scripts.modeling.finetune --epochs 6 --max-length 192 --batch-size 8 --grad-accum 2
```

**로컬 메모리 상한이 있다.** 실측 결과 `토큰 x batch`가 2,048 이상이면 Segmentation
fault로 죽는다.

| 토큰 x batch | 곱 | 결과 |
|---|---:|---|
| 128 x 8 | 1,024 | 통과 |
| 192 x 8 | 1,536 | 통과 |
| 128 x 16 | 2,048 | **Segfault** |
| 256 x 8 | 2,048 | **Segfault** |

어느 축을 줄이든 상관없다. `OMP_NUM_THREADS=1`로도 해결되지 않으므로 스레드가 아니라
할당 크기 문제다. 유효 batch는 `--grad-accum`으로 메모리와 분리해 키운다. 누적할 때 loss를
누적 횟수로 나누므로 `--grad-accum`을 바꿔도 learning rate를 다시 잡을 필요는 없다.

### gcube — 본 실험

컨테이너에서 저장소를 받아 그대로 돌린다. 학습에 필요한 것은 동결 라벨 파일 하나이고
저장소에 들어 있다. 원본 RFP는 필요 없다.

```bash
git clone https://github.com/Yiji-1015/rfp-risk-ft.git && cd rfp-risk-ft
pip install -r requirements.txt          # 리눅스라 CUDA 빌드 torch가 잡힌다
export RFP_DATASET_VERSION=v4
python -m scripts.modeling.finetune --model klue/roberta-base --fold -1 \
  --epochs 4 --max-length 512 --batch-size 32
```

`--fold -1`이 10 fold 전체다. GPU에서는 위 메모리 상한을 신경 쓰지 않아도 된다.

결과는 `reports/current/<버전>/finetune_runs.jsonl`에 **한 줄씩 덧붙는다.** 설정과 fold별
학습 곡선이 함께 남으므로 나중에 실행끼리 비교할 수 있다. 모델 가중치는 Git에 넣지 않는다.

## 4. 사전 등록해 둔 비교

`--mask subject+ending+josa`는 2026-09-02 14:04 결정이 등록한 입력 변형이다. TF-IDF에서
관찰한 이득(word+char +0.018)이 **다른 모델 계열에서도 재현되는지**가 질문이다. 같은 924건을
보고 고른 조합이라 TF-IDF에서 다시 재는 것으로는 확인되지 않는다.

```bash
python -m scripts.modeling.finetune --model klue/roberta-base --fold -1 --max-length 512   # 원문
python -m scripts.modeling.finetune --model klue/roberta-base --fold -1 --max-length 512 \
  --mask subject+ending+josa                                                               # 변형
```

재현되면 입력 표현 효과로, 재현되지 않으면 희소 TF-IDF 특유의 우연으로 보고한다.

## 5. 1단계에서 관측한 것 (2026-09-02)

`klue/roberta-small`, fold 0(ccrs), 192토큰, 유효 batch 16, CPU.

```
epoch   학습 loss   검증 macro F1
  1      1.0661        0.437
  2      0.8834        0.603
  3      0.7058        0.642
  4      0.5641        0.625
  5      0.4613        0.649  ← 최고
  6      0.3660        0.622
평가 문서 macro F1 0.582
```

- **학습은 정상이다.** loss가 단조 감소하므로 `lr=2e-5`를 손댈 이유가 없다.
- **epoch 3에서 이미 상한이다.** 검증은 0.62~0.65에서 오르내리는데 학습 loss는 계속
  내려간다. 이후 epoch은 일반화가 아니라 학습 데이터 암기에 쓰인다.
- **최고 epoch 선택 자체가 잡음이다.** 3~6 epoch 검증값이 ±0.02 안에서 흔들린다. 검증
  문서가 156건뿐이라 epoch 5가 3보다 낫다고 말할 근거가 없다. 한계로 함께 보고한다.

같은 fold의 기준선과만 비교한다. 10 fold 평균(0.614)과 비교하면 안 된다.

| fold 0 (ccrs, 평가 88건) | macro F1 |
|---|---:|
| Dummy | 0.152 |
| char TF-IDF | **0.680** |
| word+char TF-IDF | 0.663 |
| 파인튜닝 (small, 192토큰, seed 1개) | 0.582 |

ccrs는 TF-IDF가 잘하는 fold다(평균 0.608 대비 0.680). 현재 조건은 **제일 작은 모델 ·
잘린 입력 · fold 하나 · seed 하나**이므로 이 숫자로 결론을 내리지 않는다.
