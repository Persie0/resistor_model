from __future__ import annotations

from dataclasses import dataclass

DIGIT = {"black": 0, "brown": 1, "red": 2, "orange": 3, "yellow": 4, "green": 5, "blue": 6, "violet": 7, "gray": 8, "white": 9}
MULTIPLIER = {**{k: 10.0 ** v for k, v in DIGIT.items()}, "gold": 0.1, "silver": 0.01}
TOLERANCE = {"brown": 1.0, "red": 2.0, "green": 0.5, "blue": 0.25, "violet": 0.1, "gray": 0.05, "gold": 5.0, "silver": 10.0}
TEMPCO = {"brown": 100.0, "red": 50.0, "orange": 15.0, "yellow": 25.0, "blue": 10.0, "violet": 5.0}


@dataclass(frozen=True)
class DecodeResult:
    valid: bool
    colors: list[str]
    ohms: float | None = None
    tolerance_percent: float | None = None
    tempco_ppm: float | None = None


def _decode_one(colors: list[str]) -> DecodeResult:
    colors = [c.lower() for c in colors]
    n = len(colors)
    if n not in (4, 5, 6):
        return DecodeResult(False, colors)
    digit_count = 2 if n == 4 else 3
    digits = colors[:digit_count]
    multiplier = colors[digit_count]
    tolerance = colors[digit_count + 1]
    if any(c not in DIGIT for c in digits) or multiplier not in MULTIPLIER or tolerance not in TOLERANCE:
        return DecodeResult(False, colors)
    tempco = None
    if n == 6:
        if colors[5] not in TEMPCO:
            return DecodeResult(False, colors)
        tempco = TEMPCO[colors[5]]
    significant = 0
    for c in digits:
        significant = significant * 10 + DIGIT[c]
    return DecodeResult(True, colors, significant * MULTIPLIER[multiplier], TOLERANCE[tolerance], tempco)


def decode_resistor(spatial_colors: list[str]) -> DecodeResult:
    forward = _decode_one(list(spatial_colors))
    reverse = _decode_one(list(reversed(spatial_colors)))
    if forward.valid:
        return forward
    if reverse.valid:
        return reverse
    return forward
