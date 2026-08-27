"""How a channel's values may be combined by downstream processing.

Most image data is continuous: averaging four pixels to make one is exactly
what you want when downsampling or blending a stitch seam. Some data is not,
and for it averaging is not merely imprecise but *meaningless* -- and, worse,
silently so. The result is a well-formed image that no longer means anything:

* **Label / mask channels.** The mean of class 3 and class 7 is class 5, which
  may be a real class the pixel never belonged to.
* **Angular channels.** For an axial angle where 0 and 180 degrees are the same
  physical direction, the mean of 179 and 1 is 90 -- perpendicular to the truth
  and entirely plausible-looking.
* **Index or ID channels.** Averaging two object ids yields a third object.

None of these raise. Nothing downstream can tell that it happened, because the
output is a valid image of the right shape and dtype. So the channel has to say
so itself, travelling with the pixels rather than living in a reader's
assumptions.

The vocabulary is deliberately small and its default is deliberately unsafe-to-
extend-carelessly:

    RESAMPLE_LINEAR       values may be averaged, interpolated, blended
    RESAMPLE_NEAREST      values must be SELECTED, never combined
    RESAMPLE_ANGULAR_180  axial angle; combine only via sin(2t)/cos(2t)
    RESAMPLE_ANGULAR_360  directional angle; combine only via sin(t)/cos(t)

**Only the literal LINEAR authorises combining values.** Any other value --
including one a future version adds that this reader has never heard of -- must
be treated as non-combinable. That is what :func:`may_combine` implements, and
it is the whole point of the design: an unrecognised policy degrades to safe
behaviour rather than to silent corruption.

Absent metadata means LINEAR, because that is what every existing channel is
and what every existing reader already does.
"""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional

__all__ = [
    "RESAMPLE_ANGULAR_180",
    "RESAMPLE_ANGULAR_360",
    "RESAMPLE_KEY",
    "RESAMPLE_LINEAR",
    "RESAMPLE_NEAREST",
    "RESAMPLE_PERIOD_KEY",
    "RESAMPLE_REASON_KEY",
    "channel_handling",
    "may_combine",
    "resample_period",
    "resample_policy",
]

RESAMPLE_KEY = "qpsc.resample"
RESAMPLE_REASON_KEY = "qpsc.resample_reason"
RESAMPLE_PERIOD_KEY = "qpsc.resample_period"

RESAMPLE_LINEAR = "linear"
RESAMPLE_NEAREST = "nearest"
RESAMPLE_ANGULAR_180 = "angular180"
RESAMPLE_ANGULAR_360 = "angular360"

_KNOWN = frozenset({RESAMPLE_LINEAR, RESAMPLE_NEAREST, RESAMPLE_ANGULAR_180, RESAMPLE_ANGULAR_360})


_ANGULAR = frozenset({RESAMPLE_ANGULAR_180, RESAMPLE_ANGULAR_360})


def channel_handling(
    policy: str, reason: Optional[str] = None, period: Optional[float] = None
) -> Dict[str, Any]:
    """Build the annotation entries declaring how a channel may be resampled.

    Merge the result into the ``map_annotations`` passed to the OME writer.

    Args:
        policy: One of the ``RESAMPLE_*`` constants.
        period: For angular policies, the stored value spanning one full cycle
            (e.g. 18000 when 0..18000 counts span 0..180 degrees). Required for
            them, because a reader holding counts cannot otherwise recover
            angles to average.
        reason: Short human-readable justification. Worth supplying -- an
            operator reading the metadata later needs to know *why* a channel
            is restricted, not just that it is.

    Raises:
        ValueError: if ``policy`` is not a known constant. Declaring a policy
            no reader understands would be worse than declaring none, since it
            reads as deliberate.
    """
    if policy not in _KNOWN:
        raise ValueError(f"unknown resample policy {policy!r}; expected one of {sorted(_KNOWN)}")
    if policy in _ANGULAR and not period:
        raise ValueError(
            f"policy {policy!r} requires period=<stored value spanning one full cycle>. "
            "Without it a reader cannot convert counts to angles, so it can only fall "
            "back to nearest-neighbour and the channel loses the correct averaging it "
            "was declaring."
        )
    out: Dict[str, Any] = {RESAMPLE_KEY: policy}
    if reason:
        out[RESAMPLE_REASON_KEY] = reason
    if period:
        out[RESAMPLE_PERIOD_KEY] = period
    return out


def resample_period(annotations: Optional[Mapping[str, Any]]) -> Optional[float]:
    """Stored value spanning one full cycle, or None if not declared.

    Required to average an angular channel: the values are counts, and without
    the period there is no way to turn them into angles. A reader that finds an
    angular policy with no period must fall back to nearest-neighbour rather
    than guess a scale.
    """
    if not annotations:
        return None
    raw = annotations.get(RESAMPLE_PERIOD_KEY)
    if raw is None:
        return None
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return None
    return value if value > 0 else None


def resample_policy(annotations: Optional[Mapping[str, Any]]) -> str:
    """Return the declared policy, defaulting to LINEAR when absent.

    The value is returned verbatim even when unrecognised, so a caller can log
    what it actually saw. Use :func:`may_combine` to decide behaviour.
    """
    if not annotations:
        return RESAMPLE_LINEAR
    value = annotations.get(RESAMPLE_KEY)
    if value is None:
        return RESAMPLE_LINEAR
    return str(value).strip().lower()


def may_combine(annotations: Optional[Mapping[str, Any]]) -> bool:
    """Whether averaging, interpolating or blending this channel is permitted.

    True only for an explicit (or absent) LINEAR policy. Every other value,
    recognised or not, returns False -- see the module docstring: an unknown
    policy must fail towards preserving the data.
    """
    return resample_policy(annotations) == RESAMPLE_LINEAR
