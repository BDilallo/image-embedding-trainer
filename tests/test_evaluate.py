import torch

from image_embedding_trainer.evaluate import (
    compute_pair_score_stats,
    find_best_threshold,
    topk_retrieval_accuracy,
)


def _orthogonal_embeddings() -> tuple[torch.Tensor, torch.Tensor]:
    """3 classes, 2 samples each. Same-class samples are identical (sim=1),
    different-class samples are orthogonal (sim=0)."""

    embeddings = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0],
        ]
    )
    labels = torch.tensor([0, 0, 1, 1, 2, 2])

    return embeddings, labels


def test_topk_retrieval_accuracy_perfect_separation():
    embeddings, labels = _orthogonal_embeddings()

    results = topk_retrieval_accuracy(embeddings, labels, ks=(1, 5))

    assert results["top1"] == 1.0
    assert results["top5"] == 1.0


def test_topk_retrieval_accuracy_worst_case():
    # Each sample's nearest (non-self) neighbor is a different class.
    embeddings = torch.tensor(
        [
            [1.0, 0.0],
            [0.9, 0.1],
            [0.0, 1.0],
            [0.1, 0.9],
        ]
    )
    embeddings = torch.nn.functional.normalize(embeddings, dim=1)
    labels = torch.tensor([0, 1, 0, 1])

    results = topk_retrieval_accuracy(embeddings, labels, ks=(1,))

    assert results["top1"] == 0.0


def test_compute_pair_score_stats_counts_and_values():
    embeddings, labels = _orthogonal_embeddings()

    stats = compute_pair_score_stats(embeddings, labels)

    assert stats["num_pos_pairs"] == 3
    assert stats["num_neg_pairs"] == 12
    assert stats["pos_mean"] == 1.0
    assert stats["pos_std"] == 0.0
    assert stats["neg_mean"] == 0.0
    assert stats["neg_std"] == 0.0


def test_compute_pair_score_stats_no_negative_pairs():
    embeddings = torch.tensor([[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]])
    labels = torch.tensor([0, 0, 0])

    stats = compute_pair_score_stats(embeddings, labels)

    assert stats["num_pos_pairs"] == 3
    assert stats["num_neg_pairs"] == 0
    assert stats["neg_mean"] == 0.0
    assert stats["neg_std"] == 0.0


def test_find_best_threshold_perfect_separation():
    embeddings, labels = _orthogonal_embeddings()

    result = find_best_threshold(embeddings, labels)

    assert result["f1"] == 1.0
    assert result["precision"] == 1.0
    assert result["recall"] == 1.0
    assert 0.0 < result["threshold"] < 1.0
