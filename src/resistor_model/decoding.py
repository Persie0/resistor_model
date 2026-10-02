"""Grammar-aware decoding of ResistorBandNet outputs."""

from __future__ import annotations

from dataclasses import dataclass, replace
import itertools
from typing import Sequence

import numpy as np
import torch

from resistor_model.constants import INDEX_TO_COLOR
from resistor_model.decoder import DIGIT, DecodeResult, decode_resistor

_E24 = (
    10, 11, 12, 13, 15, 16, 18, 20, 22, 24, 27, 30,
    33, 36, 39, 43, 47, 51, 56, 62, 68, 75, 82, 91,
)
_E96 = (
    100, 102, 105, 107, 110, 113, 115, 118, 121, 124,
    127, 130, 133, 137, 140, 143, 147, 150, 154, 158,
    162, 165, 169, 174, 178, 182, 187, 191, 196, 200,
    205, 210, 215, 221, 226, 232, 237, 243, 249, 255,
    261, 267, 274, 280, 287, 294, 301, 309, 316, 324,
    332, 340, 348, 357, 365, 374, 383, 392, 402, 412,
    422, 432, 442, 453, 464, 475, 487, 499, 511, 523,
    536, 549, 562, 576, 590, 604, 619, 634, 649, 665,
    681, 698, 715, 732, 750, 768, 787, 806, 825, 845,
    866, 887, 909, 931, 953, 976,
)
STANDARD_MANTISSAS = frozenset(_E96) | frozenset(10 * value for value in _E24)


@dataclass(frozen=True)
class DecodedBands:
    colors: list[str]
    score: float
    decode: DecodeResult
    count: int


def mantissa3(decode: DecodeResult) -> int | None:
    if not decode.valid or len(decode.colors) not in (3, 4, 5, 6):
        return None
    significant = 2 if len(decode.colors) <= 4 else 3
    value = int("".join(str(DIGIT[color]) for color in decode.colors[:significant]))
    return value * 10 if significant == 2 else value


def in_series(decode: DecodeResult) -> bool:
    if decode.valid:
        return mantissa3(decode) in STANDARD_MANTISSAS
    if decode.ambiguous:
        return any(in_series(candidate) for candidate in decode.candidates)
    return False


def best_valid_sequence(
    color_probs: np.ndarray,
    *,
    top_k: int = 3,
    min_prob: float = 0.02,
    series_bonus: float = 0.0,
    count_logp: float = 0.0,
) -> DecodedBands | None:
    """Return the highest-scoring electrically valid sequence among top color hypotheses."""
    n = int(color_probs.shape[0])
    logp = np.log(np.clip(color_probs, 1e-9, 1.0))
    per_slot: list[list[int]] = []
    for row in color_probs:
        order = np.argsort(-row)[:top_k]
        keep = [int(order[0])] + [int(color) for color in order[1:] if row[color] >= min_prob]
        per_slot.append(keep)

    best: DecodedBands | None = None
    for combo in itertools.product(*per_slot):
        names = [INDEX_TO_COLOR[color] for color in combo]
        decoded = decode_resistor(names)
        if not (decoded.valid or decoded.ambiguous):
            continue
        score = count_logp + float(sum(logp[i, color] for i, color in enumerate(combo)))
        if series_bonus and in_series(decoded):
            score += series_bonus
        candidate = DecodedBands(names, float(score), decoded, n)
        if best is None or candidate.score > best.score:
            best = candidate
    return best


def count_log_probs(
    count_logits: torch.Tensor,
    slot_exist_logits: torch.Tensor | None = None,
    mix: float = 0.5,
) -> torch.Tensor:
    """Log-probability over band count, optionally mixing count and prefix-existence heads."""
    count_logp = torch.log_softmax(count_logits.detach().float(), dim=-1)
    if slot_exist_logits is None or mix <= 0.0:
        return count_logp
    exist = torch.sigmoid(slot_exist_logits.detach().float()).clamp(1e-6, 1.0 - 1e-6)
    on = torch.log(exist)
    off = torch.log1p(-exist)
    columns = [on[:, :n].sum(1) + off[:, n:].sum(1) for n in range(exist.shape[1] + 1)]
    exist_logp = torch.stack(columns, dim=1)
    exist_logp = exist_logp - torch.logsumexp(exist_logp, dim=1, keepdim=True)
    mixed = (1.0 - float(mix)) * count_logp + float(mix) * exist_logp
    return mixed - torch.logsumexp(mixed, dim=1, keepdim=True)


def decode_slots(
    color_probs: np.ndarray,
    count_logp: np.ndarray | torch.Tensor,
    *,
    counts: Sequence[int] = (3, 4, 5, 6),
    count_min_prob: float = 0.02,
    top_k: int = 3,
    min_prob: float = 0.02,
    series_bonus: float = 0.0,
) -> DecodedBands:
    if isinstance(count_logp, torch.Tensor):
        count_logp = count_logp.detach().cpu().numpy()
    valid_counts = [n for n in counts if n <= color_probs.shape[0] and n < len(count_logp)]
    if not valid_counts:
        raise ValueError("no supported band counts fit the model outputs")
    best_n = max(valid_counts, key=lambda n: float(count_logp[n]))
    candidates = [n for n in valid_counts if n == best_n or float(np.exp(count_logp[n])) >= count_min_prob]
    best: DecodedBands | None = None
    for n in candidates:
        result = best_valid_sequence(
            color_probs[:n],
            top_k=top_k,
            min_prob=min_prob,
            series_bonus=series_bonus,
            count_logp=float(count_logp[n]),
        )
        if result is not None and (best is None or result.score > best.score):
            best = result
    if best is not None:
        return best
    names = [INDEX_TO_COLOR[int(i)] for i in color_probs[:best_n].argmax(axis=-1)]
    return DecodedBands(names, float(count_logp[best_n]), decode_resistor(names), best_n)


@dataclass(frozen=True)
class Segment:
    label: int
    start: int
    end: int
    color_probs: np.ndarray


def dense_segments(dense_logits: torch.Tensor, *, min_run: int = 2) -> list[Segment]:
    probs = torch.softmax(dense_logits.detach().float(), dim=0).cpu().numpy()
    num_colors = probs.shape[0] - 1
    labels = probs.argmax(axis=0)
    segments: list[Segment] = []
    cursor = 0
    while cursor < len(labels):
        end = cursor
        while end + 1 < len(labels) and labels[end + 1] == labels[cursor]:
            end += 1
        label = int(labels[cursor])
        if label < num_colors and end - cursor + 1 >= min_run:
            probabilities = probs[:num_colors, cursor:end + 1].mean(axis=1)
            probabilities = probabilities / max(float(probabilities.sum()), 1e-9)
            segments.append(Segment(label, cursor, end + 1, probabilities))
        cursor = end + 1
    return segments


def decode_dense_plain(segments: Sequence[Segment]) -> DecodedBands:
    names = [INDEX_TO_COLOR[segment.label] for segment in segments]
    return DecodedBands(names, 0.0, decode_resistor(names), len(names))


def decode_dense_constrained(
    segments: Sequence[Segment],
    *,
    top_k: int = 3,
    min_prob: float = 0.02,
    series_bonus: float = 0.0,
) -> DecodedBands:
    if not 3 <= len(segments) <= 6:
        return decode_dense_plain(segments)
    probabilities = np.stack([segment.color_probs for segment in segments])
    result = best_valid_sequence(probabilities, top_k=top_k, min_prob=min_prob, series_bonus=series_bonus)
    return result if result is not None else decode_dense_plain(segments)


def geometry_direction_score(centers: Sequence[float]) -> float:
    centers = sorted(float(center) for center in centers)
    if len(centers) not in (4, 5):
        return 0.0
    span = max(centers[-1] - centers[0], 1e-6)
    return ((centers[-1] - centers[-2]) - (centers[1] - centers[0])) / span


def resolve_ambiguous_by_geometry(
    decoded: DecodedBands,
    centers: Sequence[float],
    min_score: float = 0.05,
) -> DecodedBands:
    if not decoded.decode.ambiguous or len(decoded.decode.candidates) != 2:
        return decoded
    score = geometry_direction_score(list(centers)[:decoded.count])
    if abs(score) < min_score:
        return decoded
    return replace(decoded, decode=decoded.decode.candidates[0 if score > 0 else 1])


def decode_batch(
    outputs: dict[str, torch.Tensor],
    *,
    counts: Sequence[int] = (3, 4, 5, 6),
    top_k: int = 3,
    series_bonus: float = 0.0,
    count_mix: float = 0.5,
    use_geometry: bool = False,
    geometry_min_score: float = 0.05,
) -> list[DecodedBands]:
    probabilities = torch.softmax(outputs["slot_color_logits"].detach().float(), dim=-1).cpu().numpy()
    count_lp = count_log_probs(outputs["count_logits"], outputs.get("slot_exist_logits"), mix=count_mix)
    centers = outputs["slot_center"].detach().float().cpu().numpy() if use_geometry else None
    results: list[DecodedBands] = []
    for i in range(probabilities.shape[0]):
        result = decode_slots(
            probabilities[i],
            count_lp[i],
            counts=counts,
            top_k=top_k,
            series_bonus=series_bonus,
        )
        if centers is not None:
            result = resolve_ambiguous_by_geometry(result, centers[i], geometry_min_score)
        results.append(result)
    return results
