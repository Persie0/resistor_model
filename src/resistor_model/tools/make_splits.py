from __future__ import annotations

import argparse
import json
from pathlib import Path

from resistor_model.data.schema import load_manifest
from resistor_model.data.split import grouped_split


def main() -> None:
    ap = argparse.ArgumentParser(description="Create leakage-proof resistor-ID splits")
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--ratios", type=float, nargs=3, default=(0.7, 0.15, 0.15))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--group-session", action="store_true")
    args = ap.parse_args()
    rows = load_manifest(args.manifest)
    records = [(row.image, r.id, row.session_id) for row in rows for r in row.resistors]
    splits = grouped_split(records, tuple(args.ratios), args.seed, args.group_session)
    payload = {name: sorted({x[1] for x in values}) for name, values in splits.items()}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({k: len(v) for k, v in payload.items()}, indent=2))


if __name__ == "__main__":
    main()
