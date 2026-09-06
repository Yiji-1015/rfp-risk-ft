"""실행을 유일하게 가리키는 이름을 **기록할 때** 만든다.

읽는 쪽에서 설정을 보고 이름을 다시 만들면, 규칙이 조금만 어긋나도 서로 다른 실행이
같은 이름을 갖는다. 이름이 겹치면 딕셔너리에 나중 것이 덮어써지고 앞의 실행은
경고 없이 사라진다. 2026-09-06에 실제로 그렇게 됐다 — `ftL` 하나에 roberta-large
seed 42·7·13이 들어가 둘이 사라졌고, `ftB`가 크기 이름만 봐서 `kobigbird-bert-base`를
roberta-base로 읽었으며, 2분류 실행이 3분류 투표에 섞여 단독 macro F1 0.27로 찍혔다.

그래서 이름은 실행이 만들어질 때 한 번 정해 레코드에 박아 둔다. 읽는 쪽은 그 값을
그대로 쓰고, 없을 때만(옛 기록) 설정에서 되만든다.

이름에 들어가는 것은 **결과를 바꾸는 조건**뿐이다. 출력 경로나 장치처럼 결과와
무관한 값은 넣지 않는다 — 넣으면 같은 실험이 경로만 달라도 다른 실행으로 보인다.
"""

from __future__ import annotations

from typing import Any

# 결과를 바꾸는 조건. 여기 없는 설정은 이름에 영향을 주지 않는다.
IDENTITY_KEYS = ("model", "seed", "max_length", "mask", "binary", "fold")


def _short_model(model: str) -> str:
    """`klue/roberta-large` → `roberta-large`. 계열과 크기를 모두 남긴다."""
    return model.split("/")[-1]


def run_id(config: dict[str, Any], dataset_version: str) -> str:
    """설정과 데이터셋 버전에서 유일한 실행 이름을 만든다.

    `roberta-large.s42.len512.v4` 꼴이며, 마스킹과 2분류는 접미사로 붙는다.
    사람이 읽을 수 있어야 기록을 눈으로 훑을 때 쓸모가 있으므로 해시는 쓰지 않는다.
    """
    parts = [_short_model(str(config.get("model", "unknown")))]
    parts.append(f"s{config.get('seed', 'NA')}")
    if config.get("max_length"):
        parts.append(f"len{config['max_length']}")
    parts.append(dataset_version)
    if config.get("mask"):
        parts.append(f"mask-{config['mask']}")
    if config.get("binary"):
        parts.append("binary")
    fold = config.get("fold")
    if fold is not None and fold != -1:
        parts.append(f"fold{fold}")
    return ".".join(parts)


def identity(config: dict[str, Any]) -> tuple:
    """이름을 만들 때 실제로 본 값들. 충돌을 설명할 때 쓴다."""
    return tuple(config.get(key) for key in IDENTITY_KEYS)
