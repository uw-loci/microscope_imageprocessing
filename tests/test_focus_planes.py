"""Choosing the SAMPLE plane when the field also contains debris.

Fixtures are built to the numbers measured on PPM 20x, 2026-09-04, at the four positions
that failed a 4-slide run -- see the module docstring in focus/planes.py for the table.
The real frames are 7 GB and not in the repo, so these reproduce their block statistics:
how many of the 64 blocks agree on a plane, and how spread out those blocks are.
"""

from __future__ import annotations

import numpy as np
import pytest

from microscope_imageprocessing.focus.planes import (
    MIN_AGREEING_BLOCKS,
    MIN_DISPERSION,
    _dispersion,
    block_focus_profile,
    detect_focus_plane,
)

GRID = 8
N = 380
ZS = np.linspace(0.0, -360.0, N)


def _scan(peaks, noise=0.02, seed=0):
    """A traverse of per-block profiles.

    ``peaks`` is a list of (z, block_indices, height) -- each named block gets a gaussian
    focus peak at that z, ``height`` times its own flat baseline.
    """
    rng = np.random.default_rng(seed)
    B = np.ones((N, GRID * GRID)) + rng.normal(0, noise, (N, GRID * GRID))
    for z0, blocks, height in peaks:
        bump = np.exp(-((ZS - z0) ** 2) / (2 * 2.2**2))  # ~5 um FWHM, as measured
        for b in blocks:
            B[:, b] += (height - 1.0) * bump
    return ZS, [row.reshape(GRID, GRID) for row in B]


def _corner_blocks(n):
    """Contiguous blocks in one corner -- the footprint of a fibre or a speck."""
    out = []
    for r in range(GRID):
        for c in range(GRID):
            if len(out) < n and r < 3 and c >= GRID - 3:
                out.append(r * GRID + c)
    return out


def _scattered_blocks(n, seed=1):
    """Blocks spread over the frame -- the footprint of tissue."""
    rng = np.random.default_rng(seed)
    return sorted(rng.permutation(GRID * GRID)[:n].tolist())


def test_position_A_takes_the_tissue_not_the_fibre():
    """The failure this exists for.

    At (6281,-38413) a textile fibre peaked at Z=-219 and the tissue at Z=-327. Whole-frame
    brenner ranked the fibre HIGHER, and production committed to it. Here the fibre is given
    the taller peak, as measured, and the plane detector must still choose the tissue --
    because the decision is about WHERE the sharpness sits, not how much of it there is.
    """
    zs, profiles = _scan(
        [
            (-219.0, _corner_blocks(8), 2.6),  # fibre: fewer blocks, but a taller peak
            (-327.0, _scattered_blocks(14), 1.9),  # tissue: more blocks, spread out, lower peak
        ]
    )
    r = detect_focus_plane(zs, profiles)
    assert r.found
    assert r.z == pytest.approx(-327.0, abs=3.0), "must not land on the fibre at -219"
    assert r.n_blocks >= MIN_AGREEING_BLOCKS
    assert r.dispersion >= MIN_DISPERSION


def test_position_B_refuses_a_field_whose_only_sharp_things_are_dust():
    """Two dust specks on featureless stained background: 4 live blocks of 64.

    Refusing is the correct answer. Committing here focuses on debris, and the tile needs
    reselecting -- no threshold on a whole-frame metric can tell you that.
    """
    zs, profiles = _scan([(-284.0, _corner_blocks(4), 3.0)])
    r = detect_focus_plane(zs, profiles)
    assert not r.found
    assert "worth focusing on" in r.reason


def test_position_C_accepts_ordinary_full_field_tissue():
    """The case that already worked, which must keep working: 63 of 64 blocks agree."""
    zs, profiles = _scan([(-349.0, list(range(GRID * GRID - 1)), 2.6)])
    r = detect_focus_plane(zs, profiles)
    assert r.found
    assert r.z == pytest.approx(-349.0, abs=3.0)
    assert r.n_blocks > 50


def test_position_D_refuses_when_the_sample_lies_beyond_the_scan():
    """Tissue at -411 but the traverse stopped at -360.

    There is no signal to find: a ~5 um FWHM peak leaves nothing measurable 50 um away, so
    a scan that stopped short looks exactly like a scan over nothing. Refusing lets the
    caller extend the range instead of committing to noise.
    """
    zs, profiles = _scan([])
    r = detect_focus_plane(zs, profiles)
    assert not r.found


def test_a_clustered_plane_is_rejected_even_with_enough_blocks():
    """Dispersion is a SEPARATE gate from count, and this is the case that needs it.

    A larger piece of debris can light up plenty of blocks. What it cannot do is light up
    blocks scattered across the whole frame.
    """
    zs, profiles = _scan([(-250.0, _corner_blocks(9) + [GRID + 1, GRID + 2, 2 * GRID + 1], 3.0)])
    r = detect_focus_plane(zs, profiles)
    if r.found:
        pytest.fail(f"accepted a clustered plane at {r.z} with dispersion {r.dispersion:.3f}")
    assert (
        any(x["dispersion"] < MIN_DISPERSION for x in r.rejected) or "worth focusing on" in r.reason
    )


def test_dispersion_separates_a_corner_blob_from_a_scatter():
    corner = _dispersion(_corner_blocks(9), GRID)
    scatter = _dispersion(_scattered_blocks(14), GRID)
    assert corner < MIN_DISPERSION < scatter
    assert _dispersion([5], GRID) == 0.0


def test_thresholds_sit_in_the_measured_gaps():
    """Guards against a later tidy-up tightening these onto the observed values.

    Measured: fibre 8 blocks / 0.247 dispersion, dust 4 / 0.199, A-tissue 14 / 0.366,
    C-tissue 63 / 0.424. Four planes is not a distribution -- keep the bars in the gaps.
    """
    assert 8 < MIN_AGREEING_BLOCKS < 14
    assert 0.247 < MIN_DISPERSION < 0.366


def test_block_profile_shape_and_orientation():
    img = np.zeros((772, 1024, 3), dtype=np.uint8)
    img[:, :128] = 255  # a hard edge in the leftmost block column
    p = block_focus_profile(img)
    assert p.shape == (GRID, GRID)
    assert p[:, 0].mean() > p[:, 4].mean()  # energy where the edge is, not elsewhere


def test_too_few_samples_is_refused_not_guessed():
    assert not detect_focus_plane([0.0, -1.0], [np.ones((GRID, GRID))] * 2).found
