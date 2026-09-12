"""Which colour plane the focus metric is allowed to see.

The 2026-09-04 PPM traverses, scored with the metric the server actually runs: over a
field holding both a textile fibre and tissue, the equal-weighted mean of the colour
planes peaked at Z=-219.2 -- the fibre -- while the green plane alone peaked at -327.6,
against a hand focus of -332.

The mechanism is that debris is achromatic. A fibre or a dust speck absorbs across the
whole spectrum, so it carries its full contrast into every channel; stained tissue
carries most of its contrast where the stain absorbs. Averaging the three planes keeps
all of the debris signal and dilutes the tissue's, so on a field containing both, the
debris wins by construction rather than by being sharper.
"""

import numpy as np
import pytest

from microscope_imageprocessing.focus.metrics import resolve_metric, to_gray

ACHROMATIC_COL, STAINED_COL = 12, 42


def _field(shape=(64, 64)):
    """A frame with an ACHROMATIC bar and a GREEN-ABSORBING bar of equal green depth.

    Bars run VERTICALLY on purpose: brenner_gradient is a lag-2 HORIZONTAL difference,
    and a horizontal bar is invisible to it.
    """
    a = np.full(shape + (4,), 200, dtype=np.uint8)
    a[:, 10:14, :3] = 60  # achromatic -- dark in R, G and B alike
    a[:, 40:44, 1] = 60  # stain-like -- dark in G only
    a[:, :, 3] = 255
    return a


def test_green_reduction_takes_the_green_plane_verbatim():
    g = to_gray(_field(), "green")
    assert g[0, ACHROMATIC_COL] == 60
    assert g[0, STAINED_COL] == 60  # the stained bar survives at full depth
    assert g[0, 0] == 200


def test_equal_mean_dilutes_the_stained_bar_but_not_the_achromatic_one():
    g = to_gray(_field())
    assert g[0, ACHROMATIC_COL] == pytest.approx(60.0)
    assert g[0, STAINED_COL] == pytest.approx((200 + 60 + 200) / 3)
    # Equal contrast in green, unequal after averaging. That asymmetry is the defect.
    assert abs(g[0, ACHROMATIC_COL] - 200) > 2 * abs(g[0, STAINED_COL] - 200)


def test_equal_mean_is_still_the_default_everywhere():
    """Changing this globally would desync streaming AF from standard AF."""
    a = _field()
    assert np.array_equal(to_gray(a), to_gray(a, "equal_mean"))
    assert not np.array_equal(to_gray(a), to_gray(a, "green"))


def test_resolve_metric_threads_the_reduction_through():
    a = _field()
    eq = resolve_metric("brenner_gradient")(a)
    gr = resolve_metric("brenner_gradient", "green")(a)
    assert eq > 0 and gr > 0
    assert resolve_metric("brenner_gradient", "equal_mean")(a) == eq
    # Green sees both bars at full depth; equal-mean sees the stained one washed out.
    assert gr > eq


def test_the_stained_bar_alone_is_what_equal_mean_loses():
    """Isolate the claim: on a field with ONLY stained contrast, green scores higher."""
    a = np.full((64, 64, 4), 200, dtype=np.uint8)
    a[:, 40:44, 1] = 60
    a[:, :, 3] = 255
    assert resolve_metric("brenner_gradient", "green")(a) > resolve_metric("brenner_gradient")(a)


def test_mono_and_degenerate_input_ignore_the_reduction():
    mono = np.zeros((16, 16), dtype=np.uint8)
    assert np.array_equal(to_gray(mono, "green"), to_gray(mono))
    two = np.zeros((8, 8, 2), dtype=np.uint8)
    assert np.array_equal(to_gray(two, "green"), to_gray(two))  # no green plane to take
    assert to_gray(None, "green").size == 0
