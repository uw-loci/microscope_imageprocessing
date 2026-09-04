"""Which Z the SAMPLE is at, when the field also contains debris.

A whole-frame focus metric answers "how sharp is this frame", which is not the question an
autofocus actually needs. It needs "is the SAMPLE in focus", and those differ whenever the
field contains something sharper than the sample. Measured on PPM 20x, 2026-09-04, at stage
(6281, -38413): a single textile fibre lying on the slide produced a focus peak at Z=-219
while the tissue's own peak sat at Z=-327, 108 um away. Whole-frame ``brenner_gradient``
ranked the fibre higher, and the approach committed to it.

That is not a tuning problem. ``brenner_gradient`` is a mean of squared gradients, so it is
dominated by a few extreme pixels: a black fibre on white glass carries about 200 counts of
edge contrast, pale H&E texture about 20-40. A few hundred fibre pixels outweigh a whole
field of tissue, and no threshold changes that ordering.

Nor can colour settle it. Chroma is defocus-INVARIANT -- which is exactly why it is useful
for "is there tissue in this field" -- so both candidate planes in ONE field report nearly
the same chroma by construction. Measured on the two frames above, at every ``min_chroma``
from 10 to 28 the fibre and the tissue agreed to within 0.03 of stained area fraction. A
field-level test can never choose between two planes inside that field.

What DOES separate them is where the sharpness sits. Split the frame into a grid and let
every block find its own best focus:

* tissue is spread over the field, so many blocks agree on one plane, and those blocks are
  scattered across the frame;
* a fibre or a dust speck is small and local, so only the few blocks covering it agree, and
  those blocks are contiguous.

Measured on four scans at the four positions that failed a 4-slide run:

    position        agreeing blocks (of 64)   dispersion   what it was
    A  Z=-219                    8              0.247      fibre        -> reject
    A  Z=-327                   14              0.366      tissue       -> accept
    B  Z=-284                    4              0.199      dust specks  -> reject
    C  Z=-349                   63              0.424      tissue       -> accept

Both criteria separate those four on their own, and both are applied, because the cost of a
false accept (focusing on debris, or driving further into the slide than needed) is much
higher than the cost of a false reject (fall back to the caller's slower path).

This is calibrated on FOUR planes. The thresholds sit in the gaps -- 10 blocks between 8 and
14, dispersion 0.30 between 0.247 and 0.366 -- but those gaps are narrow and four points is
not a distribution. Re-measure before trusting these on a different stain or objective.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

logger = logging.getLogger(__name__)

#: Grid the frame is split into. 8x8 on a 772x1024 frame gives ~96x128 px blocks -- large
#: enough that a block of blank glass still has a meaningful gradient statistic, small
#: enough that a fibre spanning a corner lights up only a few of them.
DEFAULT_GRID = 8

#: A block is "live" -- it has a focus plane worth believing -- when its own peak stands
#: this far above its own median across the traverse. Per-block rather than per-frame, so a
#: dim corner is judged against itself.
BLOCK_LIVE_RATIO = 1.3

#: Blocks agreeing on a plane must number at least this many, of grid*grid.
MIN_AGREEING_BLOCKS = 10

#: ...and must be spread across the frame rather than clustered on one object. Mean pairwise
#: block distance over the grid diagonal; a uniform scatter scores about 0.42, a corner blob
#: about 0.2.
MIN_DISPERSION = 0.30

#: Two blocks are talking about the same plane if their best-focus Z agree within this.
CLUSTER_UM = 8.0


def block_focus_profile(image: np.ndarray, grid: int = DEFAULT_GRID) -> np.ndarray:
    """Mean squared gradient per block: a ``grid x grid`` array for one frame.

    The same quantity ``brenner_gradient`` computes, but kept per-block instead of summed,
    which is the entire point -- the sum is what lets one small bright object outvote the
    rest of the field.
    """
    a = np.asarray(image)
    a = a[:, :, :3].astype(np.float32).mean(axis=2) if a.ndim == 3 else a.astype(np.float32)
    gy, gx = np.gradient(a)
    e = gx * gx + gy * gy
    h, w = e.shape
    bh, bw = h // grid, w // grid
    if bh < 1 or bw < 1:
        raise ValueError(f"frame {h}x{w} is too small to split into a {grid}x{grid} grid")
    return e[: bh * grid, : bw * grid].reshape(grid, bh, grid, bw).mean(axis=(1, 3))


def _dispersion(block_indices: Sequence[int], grid: int) -> float:
    """Mean pairwise distance between blocks, over the grid diagonal.

    Scale-free, so it does not change meaning with grid size. Fewer than two blocks cannot
    be dispersed and score 0.
    """
    idx = np.asarray(list(block_indices), dtype=int)
    if idx.size < 2:
        return 0.0
    rows, cols = np.divmod(idx, grid)
    pts = np.stack([rows, cols], axis=1).astype(np.float64)
    d = np.hypot(pts[:, None, 0] - pts[None, :, 0], pts[:, None, 1] - pts[None, :, 1])
    iu = np.triu_indices(len(idx), k=1)
    return float(d[iu].mean() / np.hypot(grid - 1, grid - 1))


@dataclass
class PlaneResult:
    """Where the sample plane is, or why no plane could be claimed."""

    z: Optional[float]
    n_blocks: int
    dispersion: float
    n_live: int
    reason: str
    rejected: List[Dict[str, Any]] = field(default_factory=list)

    @property
    def found(self) -> bool:
        return self.z is not None


def detect_focus_plane(
    zs: Sequence[float],
    profiles: Sequence[np.ndarray],
    grid: int = DEFAULT_GRID,
    block_live_ratio: float = BLOCK_LIVE_RATIO,
    min_agreeing_blocks: int = MIN_AGREEING_BLOCKS,
    min_dispersion: float = MIN_DISPERSION,
    cluster_um: float = CLUSTER_UM,
) -> PlaneResult:
    """The Z where the SAMPLE is, from per-block focus profiles over a traverse.

    :param zs: Z of each sample, in traverse order
    :param profiles: per-frame ``grid x grid`` arrays from :func:`block_focus_profile`
    :returns: a :class:`PlaneResult`; ``found`` is False when no plane earned acceptance,
        which is a real answer -- on the measured data it correctly refuses a field whose
        only sharp objects are two dust specks, and one whose tissue lay beyond the scan.
    """
    if len(zs) != len(profiles) or len(zs) < 3:
        return PlaneResult(None, 0, 0.0, 0, "not enough samples")

    z = np.asarray(zs, dtype=np.float64)
    B = np.stack([np.asarray(p, dtype=np.float64).ravel() for p in profiles])

    peak = B.max(axis=0)
    base = np.median(B, axis=0)
    live = np.where(peak / np.maximum(base, 1e-9) >= block_live_ratio)[0]
    if live.size < min_agreeing_blocks:
        return PlaneResult(
            None,
            0,
            0.0,
            int(live.size),
            f"only {live.size} of {B.shape[1]} blocks have any focus peak "
            f"(need {min_agreeing_blocks}) -- nothing in this field is worth focusing on",
        )

    best_z = z[np.argmax(B, axis=0)]
    rejected: List[Dict[str, Any]] = []
    # Consider candidate planes strongest-support first, so the reported rejection list
    # reads in the order a human would check them.
    candidates = sorted(
        (
            {"z": float(c), "blocks": live[np.abs(best_z[live] - c) <= cluster_um]}
            for c in best_z[live]
        ),
        key=lambda d: -len(d["blocks"]),
    )
    seen: set = set()
    for cand in candidates:
        key = round(cand["z"] / cluster_um)
        if key in seen:
            continue
        seen.add(key)
        blocks = cand["blocks"]
        disp = _dispersion(blocks, grid)
        if len(blocks) < min_agreeing_blocks:
            rejected.append(
                {
                    "z": cand["z"],
                    "blocks": int(len(blocks)),
                    "dispersion": disp,
                    "why": "too few blocks agree",
                }
            )
            continue
        if disp < min_dispersion:
            rejected.append(
                {
                    "z": cand["z"],
                    "blocks": int(len(blocks)),
                    "dispersion": disp,
                    "why": "agreeing blocks are clustered on one object, not spread "
                    "over the field -- debris, not sample",
                }
            )
            continue
        return PlaneResult(
            float(cand["z"]), int(len(blocks)), disp, int(live.size), "accepted", rejected
        )

    return PlaneResult(
        None,
        0,
        0.0,
        int(live.size),
        "no candidate plane had enough well-spread agreement",
        rejected,
    )
