from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm


@torch.no_grad()
def extract_embeddings(
    model: nn.Module,
    data_loader: DataLoader,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, List[str]]:

    model.eval()

    all_embeddings = []
    all_labels = []
    all_image_paths = []

    dataset = data_loader.dataset
    samples = dataset.samples

    images_seen = 0

    for images, labels in tqdm(data_loader, desc="Extract embeddings", leave=False):
        images = images.to(device, non_blocking=True)

        batch_embeddings = model(images).cpu()

        all_embeddings.append(batch_embeddings)
        all_labels.append(labels.cpu())

        batch_size = labels.size(0)
        batch_paths = [
            samples[index][0]
            for index in range(images_seen, images_seen + batch_size)
        ]

        all_image_paths.extend(batch_paths)
        images_seen += batch_size

    embeddings = torch.cat(all_embeddings, dim=0)
    labels = torch.cat(all_labels, dim=0)

    return embeddings, labels, all_image_paths


@torch.no_grad()
def topk_retrieval_accuracy(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    ks: Sequence[int] = (1, 5),
) -> Dict[str, float]:

    # The embeddings are normalized, so dot product is equivalent to cosine similarity.
    similarity_matrix = embeddings @ embeddings.T

    # Ignore self-comparisons. Otherwise, every image would be most similar to itself.
    similarity_matrix.fill_diagonal_(-1.0)

    max_k = max(ks)

    nearest_neighbor_indices = similarity_matrix.topk(
        k=max_k,
        dim=1,
    ).indices

    nearest_neighbor_labels = labels[nearest_neighbor_indices]

    results = {}

    for k in ks:
        matching_label_found = (
            nearest_neighbor_labels[:, :k] == labels.unsqueeze(1)
        ).any(dim=1)

        accuracy = matching_label_found.float().mean().item()
        results[f"top{k}"] = accuracy

    return results


@torch.no_grad()
def compute_pair_score_stats(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
) -> Dict[str, float]:

    similarity_matrix = embeddings @ embeddings.T
    num_images = similarity_matrix.size(0)

    positive_scores = []
    negative_scores = []

    for first_index in range(num_images):
        for second_index in range(first_index + 1, num_images):
            similarity_score = similarity_matrix[first_index, second_index].item()

            same_class = labels[first_index].item() == labels[second_index].item()

            if same_class:
                positive_scores.append(similarity_score)
            else:
                negative_scores.append(similarity_score)

    positive_mean, positive_std = _mean_and_std(positive_scores)
    negative_mean, negative_std = _mean_and_std(negative_scores)

    return {
        "pos_mean": positive_mean,
        "pos_std": positive_std,
        "neg_mean": negative_mean,
        "neg_std": negative_std,
        "num_pos_pairs": len(positive_scores),
        "num_neg_pairs": len(negative_scores),
    }


def _mean_and_std(values: List[float]) -> Tuple[float, float]:

    if not values:
        return 0.0, 0.0

    tensor_values = torch.tensor(values, dtype=torch.float32)

    mean = tensor_values.mean().item()
    std = tensor_values.std(unbiased=False).item()

    return mean, std


@torch.no_grad()
def find_best_threshold(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    thresholds: Optional[Sequence[float]] = None,
) -> Dict[str, float]:

    if thresholds is None:
        thresholds = [value / 100.0 for value in range(10, 100)]

    similarity_matrix = embeddings @ embeddings.T
    num_images = similarity_matrix.size(0)

    pair_scores = []
    pair_targets = []

    for first_index in range(num_images):
        for second_index in range(first_index + 1, num_images):
            similarity_score = similarity_matrix[first_index, second_index].item()
            same_class = labels[first_index].item() == labels[second_index].item()

            pair_scores.append(similarity_score)
            pair_targets.append(1 if same_class else 0)

    scores = torch.tensor(pair_scores, dtype=torch.float32)
    targets = torch.tensor(pair_targets, dtype=torch.int64)

    best_result = {
        "threshold": 0.5,
        "f1": -1.0,
        "precision": 0.0,
        "recall": 0.0,
    }

    for threshold in thresholds:
        predictions = (scores >= threshold).long()

        true_positives = (
            (predictions == 1) & (targets == 1)
        ).sum().item()

        false_positives = (
            (predictions == 1) & (targets == 0)
        ).sum().item()

        false_negatives = (
            (predictions == 0) & (targets == 1)
        ).sum().item()

        precision = (
            true_positives / (true_positives + false_positives)
            if (true_positives + false_positives) > 0
            else 0.0
        )

        recall = (
            true_positives / (true_positives + false_negatives)
            if (true_positives + false_negatives) > 0
            else 0.0
        )

        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )

        if f1 > best_result["f1"]:
            best_result = {
                "threshold": float(threshold),
                "f1": float(f1),
                "precision": float(precision),
                "recall": float(recall),
            }

    return best_result


@torch.no_grad()
def validate_retrieval(
    model: nn.Module,
    val_loader: DataLoader,
    device: torch.device,
) -> Dict[str, float]:

    embeddings, labels, _ = extract_embeddings(
        model=model,
        data_loader=val_loader,
        device=device,
    )

    topk_results = topk_retrieval_accuracy(
        embeddings=embeddings,
        labels=labels,
        ks=(1, 5),
    )

    score_stats = compute_pair_score_stats(
        embeddings=embeddings,
        labels=labels,
    )

    best_threshold = find_best_threshold(
        embeddings=embeddings,
        labels=labels,
    )

    validation_results = {}
    validation_results.update(topk_results)
    validation_results.update(score_stats)
    validation_results.update({
        "best_threshold": best_threshold["threshold"],
        "best_f1": best_threshold["f1"],
        "best_precision": best_threshold["precision"],
        "best_recall": best_threshold["recall"],
    })

    return validation_results