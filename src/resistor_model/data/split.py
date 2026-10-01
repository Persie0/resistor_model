from __future__ import annotations

from collections import defaultdict
import random
from typing import Sequence


def _connected_groups(records: Sequence[tuple[str, str, str | None]]) -> list[list[int]]:
    parent = list(range(len(records)))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    seen_r: dict[str, int] = {}
    seen_s: dict[str, int] = {}
    for i, (_, rid, session) in enumerate(records):
        if rid in seen_r:
            union(i, seen_r[rid])
        else:
            seen_r[rid] = i
        if session is not None:
            if session in seen_s:
                union(i, seen_s[session])
            else:
                seen_s[session] = i
    groups: dict[int, list[int]] = defaultdict(list)
    for i in range(len(records)):
        groups[find(i)].append(i)
    return list(groups.values())


def grouped_split(
    records: Sequence[tuple[str, str, str | None]],
    ratios: tuple[float, float, float] = (0.7, 0.15, 0.15),
    seed: int = 42,
    group_session: bool = False,
) -> dict[str, list[tuple[str, str, str | None]]]:
    if len(ratios) != 3 or any(r < 0 for r in ratios) or abs(sum(ratios) - 1.0) > 1e-6:
        raise ValueError("ratios must be three non-negative values summing to 1")
    if not records:
        return {"train": [], "val": [], "test": []}

    if group_session:
        index_groups = _connected_groups(records)
    else:
        by_id: dict[str, list[int]] = defaultdict(list)
        for i, (_, rid, _) in enumerate(records):
            by_id[rid].append(i)
        index_groups = list(by_id.values())

    rng = random.Random(seed)
    rng.shuffle(index_groups)
    index_groups.sort(key=len, reverse=True)
    names = ("train", "val", "test")
    target = {n: ratios[i] * len(records) for i, n in enumerate(names)}
    out_idx: dict[str, list[int]] = {n: [] for n in names}

    for group in index_groups:
        def score(name: str) -> tuple[float, int]:
            projected = len(out_idx[name]) + len(group)
            deficit = target[name] - len(out_idx[name])
            overshoot = max(0.0, projected - target[name])
            return (overshoot * 2.0 - deficit, len(out_idx[name]))
        choice = min(names, key=score)
        out_idx[choice].extend(group)

    return {n: [records[i] for i in idxs] for n, idxs in out_idx.items()}
