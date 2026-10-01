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
    ambiguous: bool = False
    candidates: tuple["DecodeResult", ...] = ()


def _decode_one(colors: list[str]) -> DecodeResult:
    colors = [c.lower() for c in colors]
    n = len(colors)
    if n not in (3, 4, 5, 6):
        return DecodeResult(False, colors)

    digit_count = 2 if n in (3, 4) else 3
    digits = colors[:digit_count]
    multiplier = colors[digit_count]
    if any(c not in DIGIT for c in digits) or multiplier not in MULTIPLIER:
        return DecodeResult(False, colors)

    if n == 3:
        tolerance_percent = 20.0
    else:
        tolerance = colors[digit_count + 1]
        if tolerance not in TOLERANCE:
            return DecodeResult(False, colors)
        tolerance_percent = TOLERANCE[tolerance]

    tempco = None
    if n == 6:
        if colors[5] not in TEMPCO:
            return DecodeResult(False, colors)
        tempco = TEMPCO[colors[5]]

    significant = 0
    for c in digits:
        significant = significant * 10 + DIGIT[c]
    return DecodeResult(True, colors, significant * MULTIPLIER[multiplier], tolerance_percent, tempco)


def _electrical_key(result: DecodeResult) -> tuple[float | None, float | None, float | None]:
    return result.ohms, result.tolerance_percent, result.tempco_ppm


def decode_resistor(spatial_colors: list[str]) -> DecodeResult:
    """Decode a spatial band sequence without inventing a reading direction.

    Resistor color grammar alone does not always determine which end is the
    first significant digit. If both directions are electrically valid and
    imply different values, return an explicit ambiguous result containing
    both candidates instead of silently preferring left-to-right.
    """
    spatial = [str(c).lower() for c in spatial_colors]
    forward = _decode_one(spatial)
    reverse = _decode_one(list(reversed(spatial)))

    if forward.valid and reverse.valid:
        if _electrical_key(forward) == _electrical_key(reverse):
            return forward
        return DecodeResult(
            valid=False,
            colors=spatial,
            ambiguous=True,
            candidates=(forward, reverse),
        )
    if forward.valid:
        return forward
    if reverse.valid:
        return reverse
    return forward
