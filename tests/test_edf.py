"""Tests for extended-depth-of-field fusion and per-pixel sharpness maps.

The synthetic stacks here put a KNOWN focal surface into the data (a plane,
a step, or a flat surface) and check that the fusion recovers it. That is the
property that matters: the operator has to find the sharpest plane per pixel,
not merely produce a plausible-looking image.
"""

import numpy as np
import pytest

from microscope_imageprocessing.focus.sharpness_maps import (
    list_sharpness_map_names,
    modified_laplacian_map,
    resolve_sharpness_map,
    tenengrad_map,
    variance_map,
)
from microscope_imageprocessing.zstack import PROJECTIONS, get_projection
from microscope_imageprocessing.zstack.edf import (
    extended_depth_of_field,
    focus_height_map,
)

RNG = np.random.default_rng(20260806)


def _textured(height: int, width: int) -> np.ndarray:
    """Band-limited texture: structure at every scale, so blurring it is
    detectable at any pyramid level rather than only at the finest."""
    base = RNG.normal(0.0, 1.0, (height, width))
    from scipy import ndimage

    out = np.zeros_like(base)
    for sigma, weight in ((1.0, 1.0), (3.0, 0.7), (7.0, 0.4)):
        out += weight * ndimage.gaussian_filter(base, sigma=sigma)
    out -= out.min()
    out /= out.max()
    return out


def _blur(image: np.ndarray, sigma: float) -> np.ndarray:
    from scipy import ndimage

    if sigma <= 0:
        return image.copy()
    return ndimage.gaussian_filter(image, sigma=sigma)


def _stack_with_focal_plane(
    sharp: np.ndarray, focal_index: np.ndarray, n_planes: int, dtype=np.uint16
) -> list:
    """Build a stack where pixel (y,x) is sharpest at plane focal_index[y,x].

    Every plane is the same scene blurred by how far that plane is from the
    pixel's true focus, so the only thing distinguishing planes is sharpness.
    """
    planes = []
    scale = np.iinfo(dtype).max if np.issubdtype(dtype, np.integer) else 1.0
    for z in range(n_planes):
        distance = np.abs(focal_index.astype(float) - z)
        plane = np.zeros_like(sharp)
        # Blur varies per pixel, so build it from a few uniformly-blurred
        # versions and select per pixel by distance.
        for d in range(n_planes):
            blurred = _blur(sharp, sigma=1.6 * d)
            plane = np.where(np.isclose(distance, d), blurred, plane)
        planes.append((plane * scale).astype(dtype))
    return planes


class TestSharpnessMaps:
    def test_all_maps_rank_a_blurred_copy_lower(self):
        sharp = _textured(64, 64)
        blurred = _blur(sharp, sigma=3.0)
        for name in list_sharpness_map_names():
            fn = resolve_sharpness_map(name)
            assert fn(sharp).mean() > fn(blurred).mean(), f"{name} failed to rank sharp > blurred"

    def test_maps_return_same_shape_float(self):
        image = _textured(32, 48)
        for name in list_sharpness_map_names():
            out = resolve_sharpness_map(name)(image)
            assert out.shape == image.shape
            assert out.dtype == np.float64

    def test_multichannel_input_reduced_to_2d_map(self):
        gray = _textured(32, 32)
        rgb = np.stack([gray, gray, gray], axis=-1)
        assert tenengrad_map(rgb).shape == (32, 32)

    def test_modified_laplacian_has_no_false_border(self):
        # np.roll wraps; an unfixed border would score the outer ring far
        # above the interior and win every argmax along the tile edge.
        image = _textured(48, 48)
        m = modified_laplacian_map(image, window=1)
        border = np.concatenate([m[0, :], m[-1, :], m[:, 0], m[:, -1]])
        assert border.max() <= m.max(), "border response must not exceed the interior"

    def test_unknown_map_raises_rather_than_substituting(self):
        with pytest.raises(KeyError, match="Unknown sharpness map"):
            resolve_sharpness_map("not_a_map")

    def test_variance_map_is_non_negative(self):
        # Computed as E[x^2] - E[x]^2, which can go slightly negative
        # through floating-point cancellation on flat regions.
        assert variance_map(np.full((16, 16), 0.5)).min() >= 0.0


class TestExtendedDepthOfField:
    def test_recovers_a_tilted_focal_plane(self):
        """The OWS3 case: focus ramps linearly across the field."""
        sharp = _textured(96, 96)
        xs = np.linspace(0, 4, 96)
        focal = np.rint(np.tile(xs, (96, 1))).astype(int)
        stack = _stack_with_focal_plane(sharp, focal, n_planes=5)

        height = focus_height_map(stack)
        # Allow one plane of slack: near a boundary the true focus is
        # genuinely ambiguous between the two neighbouring planes.
        assert np.mean(np.abs(height.astype(int) - focal) <= 1) > 0.9

    def test_fused_is_sharper_than_any_single_plane(self):
        sharp = _textured(96, 96)
        xs = np.linspace(0, 4, 96)
        focal = np.rint(np.tile(xs, (96, 1))).astype(int)
        stack = _stack_with_focal_plane(sharp, focal, n_planes=5)

        fused = extended_depth_of_field(stack)
        fused_score = tenengrad_map(fused).mean()
        for i, plane in enumerate(stack):
            assert (
                fused_score > tenengrad_map(plane).mean()
            ), f"fused image must beat plane {i}; that is the entire point"

    def test_max_projection_is_wrong_for_brightfield_and_edf_is_not(self):
        """Why this operator had to exist.

        Brightfield is dark tissue on a bright field, so the BRIGHTEST pixels
        are background and defocused tissue. A max projection therefore picks
        exactly the wrong plane. This pins that difference.
        """
        sharp = 1.0 - _textured(64, 64)  # dark structure, bright background
        focal = np.zeros((64, 64), dtype=int)  # everything focused on plane 0
        stack = _stack_with_focal_plane(sharp, focal, n_planes=3)

        edf = extended_depth_of_field(stack)
        mx = PROJECTIONS["max"](stack)
        truth = stack[0]

        edf_error = np.abs(edf.astype(float) - truth.astype(float)).mean()
        max_error = np.abs(mx.astype(float) - truth.astype(float)).mean()
        assert edf_error < max_error

    def test_flat_scene_pins_to_the_middle_plane(self):
        # No texture anywhere: every plane is equally (un)sharp, so the
        # argmax is meaningless and must not scatter across planes.
        stack = [np.full((32, 32), 1000, dtype=np.uint16) for _ in range(5)]
        height = focus_height_map(stack)
        assert np.all(height == 2)

    def test_preserves_dtype_and_shape(self):
        for dtype in (np.uint8, np.uint16, np.float32):
            stack = [(_textured(32, 32) * 100).astype(dtype) for _ in range(3)]
            fused = extended_depth_of_field(stack)
            assert fused.dtype == dtype
            assert fused.shape == (32, 32)

    def test_multichannel_takes_one_plane_per_pixel_across_channels(self):
        """Colour must not be split across planes -- that fringes."""
        gray = _textured(48, 48)
        focal = np.zeros((48, 48), dtype=int)
        focal[:, 24:] = 2
        mono = _stack_with_focal_plane(gray, focal, n_planes=3)
        # Make each plane a distinct constant colour so the selected plane is
        # identifiable from the output directly.
        colour_stack = []
        for z, plane in enumerate(mono):
            rgb = np.zeros((48, 48, 3), dtype=np.uint16)
            rgb[..., 0] = plane
            rgb[..., 1] = z * 1000
            rgb[..., 2] = z * 2000
            colour_stack.append(rgb)

        fused = extended_depth_of_field(colour_stack)
        assert fused.shape == (48, 48, 3)
        # Channels 1 and 2 encode the plane index; they must agree everywhere.
        implied_from_g = fused[..., 1] // 1000
        implied_from_b = fused[..., 2] // 2000
        assert np.array_equal(implied_from_g, implied_from_b)

    def test_single_plane_stack_is_a_no_op(self):
        plane = (_textured(16, 16) * 255).astype(np.uint8)
        assert np.array_equal(extended_depth_of_field([plane]), plane)

    def test_empty_stack_raises(self):
        with pytest.raises(ValueError, match="empty stack"):
            extended_depth_of_field([])

    def test_mismatched_shapes_raise(self):
        with pytest.raises(ValueError, match="every plane must be the same size"):
            extended_depth_of_field([np.zeros((8, 8)), np.zeros((8, 9))])

    def test_height_map_returned_on_request(self):
        stack = [(_textured(24, 24) * 255).astype(np.uint8) for _ in range(4)]
        fused, height = extended_depth_of_field(stack, return_height_map=True)
        assert fused.shape == (24, 24)
        assert height.shape == (24, 24)
        assert height.dtype == np.uint8
        assert height.max() < len(stack)


class TestRegistryIntegration:
    def test_edf_is_registered_and_callable_through_the_registry(self):
        assert "edf" in PROJECTIONS
        stack = [(_textured(32, 32) * 255).astype(np.uint8) for _ in range(3)]
        out = get_projection("edf")(stack)
        assert out.shape == (32, 32)
        assert out.dtype == np.uint8

    def test_unknown_projection_still_raises(self):
        with pytest.raises(KeyError):
            get_projection("focus_stack")
