from __future__ import annotations

import math

import torch
from torch import Tensor
from torch.nn import functional as F


def _validate_scores(
    positive_scores: Tensor, positive_rows: Tensor, unjudged_scores: Tensor,
) -> Tensor:
    if unjudged_scores.ndim != 2 or 0 in unjudged_scores.shape:
        raise ValueError("objective: expected nonempty disease-by-negative scores")
    if positive_scores.ndim != 1 or positive_rows.shape != positive_scores.shape:
        raise ValueError("objective: positive scores and disease rows must be vectors")
    if positive_rows.dtype != torch.long:
        raise ValueError("objective: positive disease rows must be int64")
    if (
        positive_scores.device != unjudged_scores.device
        or positive_rows.device != unjudged_scores.device
    ):
        raise ValueError("objective: score tensors must share a device")
    if positive_scores.numel() == 0 or not torch.isfinite(positive_scores).all():
        raise ValueError("objective: positive scores must be nonempty and finite")
    if not torch.isfinite(unjudged_scores).all():
        raise ValueError("objective: unjudged scores must be finite")

    diseases = unjudged_scores.shape[0]
    if not torch.all((positive_rows >= 0) & (positive_rows < diseases)):
        raise ValueError("objective: positive disease row out of bounds")

    counts = torch.bincount(positive_rows, minlength=diseases)
    if not torch.all(counts > 0):
        raise ValueError("objective: every disease requires an observed indication")
    return counts


def sampled_bce_loss(
    positive_scores: Tensor,
    positive_rows: Tensor,
    unjudged_scores: Tensor,
    *,
    negative_weight: float,
) -> Tensor:
    """Disease-macro logistic loss with sampled unjudged candidates."""
    if not math.isfinite(negative_weight) or negative_weight < 0:
        raise ValueError("objective: negative weight must be finite and nonnegative")
    counts = _validate_scores(positive_scores, positive_rows, unjudged_scores)
    diseases = unjudged_scores.shape[0]

    positive_sums = positive_scores.new_zeros(diseases)
    positive_sums.index_add_(0, positive_rows, F.softplus(-positive_scores))
    positive_loss = positive_sums / counts
    unjudged_loss = F.softplus(unjudged_scores).mean(dim=1)

    return (positive_loss + negative_weight * unjudged_loss).mean()


def sampled_bpr_loss(
    positive_scores: Tensor, positive_rows: Tensor, unjudged_scores: Tensor,
) -> Tensor:
    """Disease-macro BPR: mean softplus(s_unjudged - s_positive) per disease."""
    counts = _validate_scores(positive_scores, positive_rows, unjudged_scores)
    differences = unjudged_scores[positive_rows] - positive_scores[:, None]
    per_positive = F.softplus(differences).mean(dim=1)
    per_disease = positive_scores.new_zeros(unjudged_scores.shape[0])
    per_disease.index_add_(0, positive_rows, per_positive)
    return (per_disease / counts).mean()


def sampled_lsep_loss(
    positive_scores: Tensor, positive_rows: Tensor, unjudged_scores: Tensor,
) -> Tensor:
    """Per-disease LSEP: log(1 + sum_{p,u} exp(s_u - s_p)), then disease mean.

    The pair sum is intentionally not divided by the number of pairs.
    """
    _validate_scores(positive_scores, positive_rows, unjudged_scores)
    diseases = unjudged_scores.shape[0]

    # log sum_p exp(-s_p) by disease; shift for stability even at extreme logits
    values = -positive_scores
    maxima = values.new_full((diseases,), -torch.inf)
    maxima.scatter_reduce_(0, positive_rows, values.detach(), reduce="amax")
    sums = values.new_zeros(diseases)
    sums.index_add_(0, positive_rows, (values - maxima[positive_rows]).exp())
    log_positive = maxima + sums.log()

    # log(1 + sum_{p,u} exp(s_u - s_p)) = softplus(logsumexp_u + logsumexp_p)
    return F.softplus(torch.logsumexp(unjudged_scores, dim=1) + log_positive).mean()


def mean_lsep_loss(
    positive_scores: Tensor,
    positive_rows: Tensor,
    candidate_scores: Tensor,
    *,
    eligible: Tensor | None = None,
) -> Tensor:
    # Disease-macro log(1 + mean_{p,u} exp(s_u - s_p))
    positive_counts = _validate_scores(positive_scores, positive_rows, candidate_scores)
    diseases, candidates = candidate_scores.shape

    if eligible is None:
        eligible_counts = torch.full(
            (diseases,), candidates, device=candidate_scores.device, dtype=torch.long,
        )
        negative = candidate_scores
    else:
        if eligible.shape != candidate_scores.shape or eligible.dtype != torch.bool:
            raise ValueError("objective: eligible mask must match scores and be boolean")
        eligible_counts = eligible.sum(dim=1)
        if not (eligible_counts > 0).all():
            raise ValueError("objective: every disease requires an eligible candidate")
        negative = candidate_scores.masked_fill(~eligible, -torch.inf)

    values = -positive_scores
    maxima = values.new_full((diseases,), -torch.inf)
    maxima.scatter_reduce_(0, positive_rows, values.detach(), reduce="amax")
    sums = values.new_zeros(diseases)
    sums.index_add_(0, positive_rows, (values - maxima[positive_rows]).exp())
    log_positive_mean = maxima + sums.log() - positive_counts.to(values.dtype).log()
    log_negative_mean = torch.logsumexp(negative, dim=1) - eligible_counts.to(negative.dtype).log()
    return F.softplus(log_positive_mean + log_negative_mean).mean()
