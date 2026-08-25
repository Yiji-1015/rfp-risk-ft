import json

import numpy as np
import pytest
import torch

from scripts.evaluation.embeddings import (
    E5_LOGISTIC,
    load_cached_embeddings,
    mean_pool,
    run_embedding_lodo,
)
from scripts.labeling.label_dataset import load_label_dataset


def test_mean_pool_ignores_padding_and_normalizes():
    hidden = torch.tensor([[[3.0, 0.0], [0.0, 4.0], [99.0, 99.0]]])
    mask = torch.tensor([[1, 1, 0]])

    pooled = mean_pool(hidden, mask)

    assert pooled.shape == (1, 2)
    assert torch.linalg.vector_norm(pooled, dim=1).item() == pytest.approx(1.0)
    assert pooled[0].tolist() == pytest.approx([0.6, 0.8])


def test_embedding_lodo_keeps_the_same_eight_document_training_split():
    rows, _ = load_label_dataset()
    embeddings = np.random.default_rng(42).normal(size=(len(rows), 8)).astype(np.float32)

    results = run_embedding_lodo(rows, embeddings, E5_LOGISTIC)

    assert len(results) == 10
    assert sum(result.test_size for result in results) == len(rows) - 100
    assert all(result.train_size < len(rows) - result.test_size for result in results)


def test_embedding_row_mismatch_fails_loudly():
    rows, _ = load_label_dataset()

    with pytest.raises(ValueError, match="임베딩 행 수"):
        run_embedding_lodo(rows, np.zeros((len(rows) - 1, 8)), E5_LOGISTIC)


def test_stale_embedding_cache_is_rejected(tmp_path):
    rows, _ = load_label_dataset()
    path = tmp_path / "stale.npz"
    np.savez_compressed(
        path,
        embeddings=np.zeros((len(rows), 8), dtype=np.float32),
        metadata=json.dumps({"pipeline_version": "old"}),
    )

    assert load_cached_embeddings(path, rows) is None
