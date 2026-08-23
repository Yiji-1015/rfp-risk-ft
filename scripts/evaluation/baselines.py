"""TF-IDF + 선형 모델 기준선을 LODO로 평가한다.

`folds.py`가 "어떻게 나눌 것인가"를 정했다면 이 파일은 "나눈 뒤 무엇을 재고 어떻게
읽을 것인가"를 담당한다.

## 왜 지표를 여러 개 내는가

한 숫자로는 이 문제를 읽을 수 없다. 같은 예측이 지표에 따라 반대로 보인다.

mfds 문서(192건)에서 실제로 관측한 값이다.

    통상수용   137건  F1 0.823
    견적반영    25건  F1 0.196
    계약·질의검토 30건  F1 0.510

    macro F1 = (0.823 + 0.196 + 0.510) / 3 = 0.510   <- 클래스마다 1/3씩
    accuracy = 0.698                                  <- 137건짜리가 지배

**macro F1은 건수를 무시한다.** 137건짜리 클래스와 25건짜리 클래스가 똑같은 무게다.
그래서 전체의 13%인 `견적반영`을 놓치면 그것만으로 0.19가 깎인다. accuracy로 보면
같은 예측이 0.698로 준수해 보인다. §10.2가 accuracy를 보조 지표로만 쓰겠다고 한 이유다.

micro F1은 단일 라벨 다중분류에서 accuracy와 항상 같으므로 따로 내지 않는다.

`계약·질의검토` recall을 따로 내는 것도 §10.2의 요구다. 이 라벨을 놓치면 실무에서
검토 없이 넘어가는 조항이 생긴다. 다른 두 라벨의 오류와 비용이 다르다.

## 왜 점수 옆에 항상 기준선과 노출률을 붙이는가

fold 점수는 그 자체로 비교할 수 없다. 두 가지가 fold마다 다르기 때문이다.

1. **난이도** — 다수 클래스만 찍어도 koen은 79.2%, defense는 34.3%다(issues/003)
2. **점수 구성** — 표준 문구 반복 노출이 mfds 22.9%, ccrs 0.0%다(issues/006)

그리고 이 둘은 **독립이 아니다.** 상관이 r = 0.856이라 쉬운 fold가 공짜 점수까지 더
받는다(2026-08-23 결정). 하나만 할인하면 과대평가, 각각 따로 할인하면 이중 할인이 된다.
그래서 `FoldResult`는 점수와 두 진단값을 **한 덩어리로** 들고 다닌다. 점수만 떼어내
비교하는 일이 실수로라도 일어나지 않게 하기 위해서다.

## 왜 세 갈래 점수를 내는가

결정 34가 정한 보고 방식이다. 표준 문구 반복은 누수가 아니지만(배포에서도 같은 조항이
나온다) 점수에 **두 능력이 섞인다** — 반복 문구를 알아보는 능력과 처음 보는 요구사항을
판단하는 능력. 섞이는 비율이 fold마다 달라서, 전체 점수만 보면 능력이 아니라 구성을
보게 된다. 그래서 전체 / 반복 제외 / 반복만으로 나눠 낸다.

## 왜 한 번에 하나만 바꾸는가

`compare()`는 설정 두 개를 받아 fold별 차이를 낸다. 그리드서치로 최고 조합을 뽑는
방식을 쓰지 않는 이유는, 이 프로젝트의 질문이 "최고 점수가 몇 점인가"가 아니라
**"복잡하게 만든 값을 했는가"**이기 때문이다(§9.3의 통제 비교). 그리드서치는 점수를
주지만 어느 선택이 효과를 냈는지는 설명하지 못한다.

문서가 10개뿐이라 fold 간 분산이 크다는 점도 함께 본다. 평균 차이가 fold별 분산보다
작으면 그것은 효과가 아니라 잡음이다. 실측 예가 둘 있다.

    학습 8문서 -> 9문서      평균 +0.002, fold별 -0.046 ~ +0.037   -> 잡음
    class_weight None -> balanced  평균 +0.078, 8/10 fold에서 우세  -> 효과

**둘의 크기 차이가 40배다.** 어느 파라미터를 먼저 만질지는 이런 크기 비교로 정한다.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any, Sequence

from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, recall_score
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from scripts.evaluation.duplication import DEFAULT_THRESHOLD
from scripts.evaluation.folds import (
    Fold,
    _repeat_flags,
    aggregate,
    diagnose_fold,
    make_lodo_folds,
)

LABELS: tuple[str, ...] = ("통상수용", "견적반영", "계약·질의검토")

# §10.2가 별도로 요구하는 라벨. 놓치면 검토 없이 넘어가는 조항이 생기므로
# 다른 두 라벨의 오류와 비용이 다르다.
REVIEW_LABEL = "계약·질의검토"

# 재현을 위해 고정한다. 선형 모델은 seed에 거의 흔들리지 않지만, 기록에 남는 값과
# 다시 돌린 값이 어긋나면 원인을 찾는 데 시간이 든다.
RANDOM_STATE = 42


@dataclass(frozen=True)
class ModelSpec:
    """파이프라인 설정 하나.

    파라미터를 **두 층으로 나눠** 적는다. TF-IDF + 선형 모델은 모델 하나가 아니라
    파이프라인 둘이고, 각 층의 파라미터는 성격이 다르기 때문이다.

    1층 표현(`analyzer` ~ `sublinear_tf`) — 텍스트를 어떻게 숫자로 바꾸는가.
      한국어 RFP는 띄어쓰기 변형("생성형AI" / "생성형 AI")과 기관별 용어, 영문 약어가
      많아 단어 단위로는 같은 문구가 다른 문구로 보인다. §9.2가 문자 n-gram을 필수로
      둔 이유다.

    2층 분류기(`classifier`, `C`) — 그 숫자로 어떻게 경계를 긋는가.
      보통 1층보다 효과가 작다. 표현이 나쁘면 `C`를 아무리 만져도 오르지 않는다.

    `class_weight`는 둘 중 어느 쪽도 아니다. 성능 파라미터가 아니라 **무엇을 잘하고
    싶은지의 선언**이다. 켜면 소수 클래스 쪽으로 결정 경계가 밀려 accuracy는 내려가고
    macro F1과 소수 클래스 recall은 올라간다. 그래서 다른 파라미터와 같은 그리드에
    섞어 "최고 점수"로 고르면 안 된다. 목적을 먼저 정하고 나서 고르는 값이다.
    """

    name: str
    analyzer: str = "char_wb"
    ngram_range: tuple[int, int] = (3, 4)
    min_df: int = 2
    sublinear_tf: bool = True
    classifier: str = "logistic"  # logistic | svm | dummy
    C: float = 1.0
    class_weight: str | None = "balanced"

    def build(self) -> Pipeline:
        if self.classifier == "dummy":
            # 최저 기준(§9.2). 본문을 보지 않고 학습 집합의 최빈 클래스만 찍는다.
            # 이 값을 넘지 못하면 모델이 텍스트에서 아무것도 배우지 못한 것이다.
            return Pipeline(
                [
                    ("tfidf", TfidfVectorizer(analyzer="char", ngram_range=(1, 1))),
                    ("clf", DummyClassifier(strategy="most_frequent")),
                ]
            )

        vectorizer = TfidfVectorizer(
            analyzer=self.analyzer,
            ngram_range=self.ngram_range,
            min_df=self.min_df,
            sublinear_tf=self.sublinear_tf,
        )
        if self.classifier == "logistic":
            clf = LogisticRegression(
                C=self.C,
                class_weight=self.class_weight,
                max_iter=2000,
                random_state=RANDOM_STATE,
            )
        elif self.classifier == "svm":
            clf = LinearSVC(
                C=self.C,
                class_weight=self.class_weight,
                random_state=RANDOM_STATE,
            )
        else:
            raise ValueError(f"알 수 없는 분류기: {self.classifier!r}")
        return Pipeline([("tfidf", vectorizer), ("clf", clf)])


@dataclass(frozen=True)
class FoldResult:
    """fold 하나의 결과.

    점수와 진단값을 **한 덩어리로** 묶어둔다. fold 점수는 그 자체로 비교할 수 없고
    난이도(`oracle_majority_accuracy`)와 점수 구성(`repeat_exposure_rate`)을 함께
    봐야 하는데, 따로 두면 점수만 떼어 비교하는 실수가 쉽게 일어난다.
    """

    fold_index: int
    test_document: str
    test_size: int
    train_size: int

    macro_f1: float
    accuracy: float
    per_class_f1: dict[str, float]
    review_recall: float

    # 이 fold를 어떻게 읽어야 하는가 (folds.py의 진단값)
    trained_majority_accuracy: float
    oracle_majority_accuracy: float
    repeat_exposure_rate: float

    # 결정 34의 세 갈래 보고. 해당 부분집합이 비어 있으면 None이다.
    macro_f1_repeat_excluded: float | None
    macro_f1_repeat_only: float | None

    # 세 갈래 점수를 읽을 때 **반드시 함께 봐야 하는 건수**. 반복 문구 부분집합은
    # fold에 따라 1건(genai)에서 44건(mfds)까지 흔들린다. 1건짜리에서 나온 0.000이나
    # 1.000은 성능이 아니라 표본 크기의 산물이다. 비율만 적어두면 이 사실이 보이지
    # 않으므로 건수를 필드로 들고 다닌다.
    repeat_count: int
    non_repeat_count: int

    @property
    def lift_over_dummy(self) -> float:
        """배포 가능한 Dummy 대비 정확도 개선폭.

        fold 간 절대 점수는 비교할 수 없지만 **기준선 대비 개선폭**은 비교할 수 있다
        (issues/003). koen에서 79%를 낸 모델과 defense에서 50%를 낸 모델 중 어느 쪽이
        나은지는 이 값으로 본다.
        """
        return self.accuracy - self.trained_majority_accuracy


def _score(gold: Sequence[str], pred: Sequence[str]) -> tuple[float, float, dict, float]:
    """지표를 한 번에 계산한다.

    `labels=LABELS`를 명시하는 것이 중요하다. 넘기지 않으면 sklearn이 정답·예측에
    등장한 라벨의 합집합을 쓰는데, 부분집합(예: 반복 문구만)에서는 세 클래스가 다
    나오지 않을 수 있다. 그러면 macro 평균의 분모가 조용히 2가 되어 다른 fold와
    비교할 수 없는 값이 나온다.
    """
    macro = f1_score(gold, pred, labels=LABELS, average="macro", zero_division=0)
    per_class = dict(
        zip(
            LABELS,
            f1_score(gold, pred, labels=LABELS, average=None, zero_division=0),
        )
    )
    review = recall_score(
        gold, pred, labels=[REVIEW_LABEL], average="macro", zero_division=0
    )
    return float(macro), float(accuracy_score(gold, pred)), per_class, float(review)


def evaluate_fold(
    fold: Fold,
    rows: Sequence[dict[str, Any]],
    spec: ModelSpec,
    *,
    repeat_flags: dict[str, bool],
    repeat_threshold: float = DEFAULT_THRESHOLD,
    use_nine_documents: bool = False,
) -> FoldResult:
    """fold 하나를 학습·평가한다.

    :param use_nine_documents: 검증 문서까지 학습에 넣을지. 기본은 False다.
        검증 문서를 비워두면 학습 데이터의 1/9을 쓰지 않는 셈이라 손해처럼 보이지만,
        실측하면 macro F1 평균 차이가 **+0.002**이고 fold별로는 -0.046 ~ +0.037로
        방향조차 일정하지 않다. 즉 이 손해는 잡음 수준이다. 대신 나중에 파인튜닝이
        그 자리를 early stopping에 쓸 때 **학습 데이터가 8문서로 같아야** 두 결과를
        나란히 놓을 수 있다(§9.3). 잃는 것이 거의 없으므로 비교 가능성을 택했다.
    """
    fit_rows, validation_rows, test_rows = fold.split(rows)
    train_rows = fit_rows + validation_rows if use_nine_documents else fit_rows

    pipeline = spec.build()
    pipeline.fit(
        [r["raw_requirement_text"] for r in train_rows],
        [r["primary_action"] for r in train_rows],
    )
    pred = list(pipeline.predict([r["raw_requirement_text"] for r in test_rows]))
    gold = [r["primary_action"] for r in test_rows]

    macro, acc, per_class, review = _score(gold, pred)

    # 결정 34의 세 갈래. 반복 문구만 모으면 fold에 따라 0건일 수 있어(ccrs 0.0%)
    # 그 경우 점수를 만들지 않고 None으로 둔다. 0.0으로 채우면 "성능이 0"으로
    # 잘못 읽힌다.
    flags = [repeat_flags[r["requirement_uid"]] for r in test_rows]
    excluded = [(g, p) for g, p, f in zip(gold, pred, flags) if not f]
    only = [(g, p) for g, p, f in zip(gold, pred, flags) if f]

    diagnostics = diagnose_fold(
        fold,
        rows,
        repeat_threshold=repeat_threshold,
        repeat_flags=repeat_flags,
    )

    return FoldResult(
        fold_index=fold.index,
        test_document=fold.test_document,
        test_size=len(test_rows),
        train_size=len(train_rows),
        macro_f1=macro,
        accuracy=acc,
        per_class_f1=per_class,
        review_recall=review,
        trained_majority_accuracy=diagnostics.trained_majority_accuracy,
        oracle_majority_accuracy=diagnostics.oracle_majority_accuracy,
        repeat_exposure_rate=diagnostics.repeat_exposure_rate,
        macro_f1_repeat_excluded=(
            _score(*zip(*excluded))[0] if excluded else None
        ),
        macro_f1_repeat_only=(_score(*zip(*only))[0] if only else None),
        repeat_count=len(only),
        non_repeat_count=len(excluded),
    )


def run_lodo(
    rows: Sequence[dict[str, Any]],
    spec: ModelSpec,
    *,
    repeat_threshold: float = DEFAULT_THRESHOLD,
    use_nine_documents: bool = False,
) -> list[FoldResult]:
    """모든 fold를 돌린다. 반복 문구 계산은 한 번만 한다."""
    repeat_flags = _repeat_flags(rows, repeat_threshold)
    return [
        evaluate_fold(
            fold,
            rows,
            spec,
            repeat_flags=repeat_flags,
            repeat_threshold=repeat_threshold,
            use_nine_documents=use_nine_documents,
        )
        for fold in make_lodo_folds(rows)
    ]


def summarize(results: Sequence[FoldResult]) -> dict[str, dict[str, float]]:
    """fold 결과를 집계한다.

    모든 지표를 **fold 평균과 건수 가중 둘 다** 낸다. 하나만 쓰면 어느 쪽이든 오해를
    만든다(issues/003). fold 평균은 49건짜리 korail과 192건짜리 mfds를 같은 무게로
    보고, 건수 가중은 큰 문서 셋(전체의 48.6%)이 결과를 지배한다.
    """
    sizes = [r.test_size for r in results]
    metrics = {
        "macro_f1": [r.macro_f1 for r in results],
        "accuracy": [r.accuracy for r in results],
        "review_recall": [r.review_recall for r in results],
        "lift_over_dummy": [r.lift_over_dummy for r in results],
    }
    return {name: aggregate(values, sizes) for name, values in metrics.items()}


@dataclass(frozen=True)
class Comparison:
    """설정 두 개의 통제 비교 결과."""

    baseline: ModelSpec
    variant: ModelSpec
    metric: str
    per_fold: list[tuple[str, float, float]]  # (문서, baseline, variant)

    @property
    def deltas(self) -> list[float]:
        return [v - b for _, b, v in self.per_fold]

    @property
    def mean_delta(self) -> float:
        return sum(self.deltas) / len(self.deltas)

    @property
    def variant_wins(self) -> int:
        return sum(1 for d in self.deltas if d > 0)

    @property
    def looks_like_noise(self) -> bool:
        """평균 효과가 fold별 편차보다 작으면 잡음으로 본다.

        문서가 10개뿐이라 fold 간 분산이 크다. 이 화면 없이 평균만 보면 잡음을
        효과로 착각한다. 실측 예: 8문서 -> 9문서는 평균 +0.002에 편차 폭 0.083으로
        잡음이었고, class_weight는 평균 +0.078로 효과였다.
        """
        spread = max(self.deltas) - min(self.deltas)
        return abs(self.mean_delta) < spread / 4


def compare(
    rows: Sequence[dict[str, Any]],
    baseline: ModelSpec,
    variant: ModelSpec,
    *,
    metric: str = "macro_f1",
    repeat_threshold: float = DEFAULT_THRESHOLD,
    **run_kwargs: Any,
) -> Comparison:
    """설정 하나만 바꿔 fold별로 비교한다.

    **한 번에 하나만 바꾼다.** 두 개를 동시에 바꾸면 어느 쪽이 효과를 냈는지 알 수
    없다. 이 프로젝트의 질문이 "최고 점수"가 아니라 "복잡하게 만든 값을 했는가"이므로
    (§9.3), 설명할 수 없는 점수는 쓸모가 적다.
    """
    a = run_lodo(rows, baseline, repeat_threshold=repeat_threshold, **run_kwargs)
    b = run_lodo(rows, variant, repeat_threshold=repeat_threshold, **run_kwargs)
    return Comparison(
        baseline=baseline,
        variant=variant,
        metric=metric,
        per_fold=[
            (x.test_document, getattr(x, metric), getattr(y, metric))
            for x, y in zip(a, b)
        ],
    )


# 지금까지 통제 비교로 확인한 설정들. 노트북 07이 이 목록을 쓴다.
DUMMY = ModelSpec(name="Dummy(최빈)", classifier="dummy")
CHAR_BALANCED = ModelSpec(name="char 3-4gram + balanced")
CHAR_UNWEIGHTED = ModelSpec(name="char 3-4gram + weight 없음", class_weight=None)
WORD_BALANCED = ModelSpec(
    name="word 1-2gram + balanced", analyzer="word", ngram_range=(1, 2)
)


def _main() -> None:
    from scripts.labeling.label_dataset import load_label_dataset

    rows, meta = load_label_dataset()
    print(f"데이터셋 {meta['dataset_version']} / {meta['row_count']}건\n")

    header = f"{'설정':<28}{'macroF1':>9}{'정확도':>9}{'계약recall':>11}{'Dummy대비':>11}"
    print(header)
    print("-" * 68)
    for spec in (DUMMY, WORD_BALANCED, CHAR_UNWEIGHTED, CHAR_BALANCED):
        s = summarize(run_lodo(rows, spec))
        print(
            f"{spec.name:<28}{s['macro_f1']['fold_mean']:>9.3f}"
            f"{s['accuracy']['fold_mean']:>9.3f}"
            f"{s['review_recall']['fold_mean']:>11.3f}"
            f"{s['lift_over_dummy']['fold_mean']:>+11.3f}"
        )
    print("\n(전부 fold 평균. 건수 가중은 summarize()가 함께 낸다)")


if __name__ == "__main__":
    _main()
