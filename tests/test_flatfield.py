"""Tests for the BaSiC flat-field fit and profiles."""

import numpy as np
import pytest
import tifffile as tf

from microscope_imageprocessing.correction import flatfield as ff
from microscope_imageprocessing.correction.flatfield import (
    FlatfieldProfile,
    estimate_offset,
    fit_profile,
    load_profile_set,
    read_tile_planes,
)

OFFSET = 6500.0
SIZE = 128


def known_shading(size=SIZE):
    y, x = np.mgrid[0:size, 0:size] / (size - 1.0)
    shading = 1 + 0.10 * (x - 0.5) - 0.12 * (y - 0.5) - 0.15 * ((x - 0.5) ** 2 + (y - 0.5) ** 2)
    return (shading / shading.mean()).astype(np.float32)


def sparse_field(rng, size=SIZE):
    """Random fibers on a faint haze, like SHG tissue."""
    field = np.full((size, size), rng.uniform(300, 900), dtype=np.float32)
    for _ in range(40):
        y, x = rng.integers(0, size, 2)
        field[y, max(0, x - 6) : x + 6] += rng.uniform(500, 3000)
    return field


def tiles(count, shading, seed=0, empty_every=0):
    rng = np.random.default_rng(seed)
    out = []
    for index in range(count):
        empty = empty_every and index % empty_every == 0
        signal = np.zeros((SIZE, SIZE), dtype=np.float32) if empty else sparse_field(rng)
        noise = rng.normal(0, 30, (SIZE, SIZE))
        out.append(np.clip(OFFSET + signal * shading + noise, 0, 65535).astype(np.uint16))
    return out


def test_fit_recovers_a_known_shading():
    shading = known_shading()
    profile = fit_profile(tiles(120, shading), offset=OFFSET)
    rms = np.sqrt(np.mean((profile.flatfield - shading) ** 2))
    assert rms < 0.03
    assert np.corrcoef(profile.flatfield.ravel(), shading.ravel())[0, 1] > 0.9


def test_flatfield_has_mean_one_at_full_size():
    profile = fit_profile(tiles(12, known_shading()), offset=OFFSET)
    assert profile.flatfield.shape == (SIZE, SIZE)
    assert profile.flatfield.mean() == pytest.approx(1.0, abs=1e-5)


def test_dark_half_is_left_out_of_a_large_fit():
    profile = fit_profile(tiles(100, known_shading(), empty_every=2), offset=OFFSET)
    assert profile.metadata["images_read"] == 100
    assert profile.metadata["images_used"] == 50


def test_selection_never_takes_a_fit_below_the_minimum():
    profile = fit_profile(tiles(60, known_shading(), empty_every=2), offset=OFFSET)
    assert profile.metadata["images_used"] == ff.MIN_IMAGES_FOR_FIT


def test_a_small_fit_uses_every_image_and_warns(caplog):
    profile = fit_profile(tiles(12, known_shading()), offset=OFFSET)
    assert profile.metadata["images_used"] == 12
    assert "nearly flat" in caplog.text


def test_a_profile_set_covers_channels_with_their_own_folder(tmp_path):
    common = FlatfieldProfile(flatfield=np.ones((4, 4), dtype=np.float32), offset=1.0)
    dapi = FlatfieldProfile(flatfield=np.ones((4, 4), dtype=np.float32), offset=2.0)
    common.save(tmp_path)
    dapi.save(tmp_path / "DAPI")
    profiles = load_profile_set(tmp_path, ["DAPI", "FITC"])
    assert sorted(str(k) for k in profiles) == ["DAPI", "None"]
    assert profiles["DAPI"].offset == 2.0
    assert profiles[None].offset == 1.0


def test_a_folder_with_no_profile_gives_an_empty_set(tmp_path):
    assert load_profile_set(tmp_path, ["DAPI"]) == {}


def test_offset_comes_from_the_empty_fields_not_the_noise_floor():
    images = tiles(50, known_shading(), empty_every=2)
    assert estimate_offset(images) == pytest.approx(OFFSET, abs=3)


def test_apply_keeps_the_background_where_it_was():
    # Dividing raw values would lift a 6500 offset by 700 at flatfield 0.9
    flatfield = np.full((4, 4), 0.9, dtype=np.float32)
    profile = FlatfieldProfile(flatfield=flatfield, offset=OFFSET)
    image = np.full((4, 4), 6500, dtype=np.uint16)
    corrected, clipped = profile.apply(image)
    assert np.all(corrected == 6500)
    assert clipped == 0


def test_apply_scales_only_the_signal_and_rounds_half_up():
    flatfield = np.array([[0.5, 2.0]], dtype=np.float32)
    profile = FlatfieldProfile(flatfield=flatfield, offset=100.0)
    corrected, _ = profile.apply(np.array([[150, 101]], dtype=np.uint16))
    assert corrected.tolist() == [[200, 101]]
    assert corrected.dtype == np.uint16


def test_apply_counts_clipped_pixels():
    flatfield = np.array([[0.5, 1.0]], dtype=np.float32)
    profile = FlatfieldProfile(flatfield=flatfield, offset=0.0)
    corrected, clipped = profile.apply(np.array([[200, 200]], dtype=np.uint8))
    assert corrected.tolist() == [[255, 200]]
    assert clipped == 1


def test_apply_commutes_with_averaging_and_max_projection():
    rng = np.random.default_rng(3)
    planes = rng.integers(6500, 9000, (5, 16, 16)).astype(np.float64)
    flatfield = rng.uniform(0.8, 1.2, (16, 16)).astype(np.float32)
    profile = FlatfieldProfile(flatfield=flatfield, offset=OFFSET)
    each = np.stack([profile.apply(plane)[0] for plane in planes])
    np.testing.assert_allclose(each.mean(axis=0), profile.apply(planes.mean(axis=0))[0], rtol=1e-6)
    np.testing.assert_allclose(each.max(axis=0), profile.apply(planes.max(axis=0))[0], rtol=1e-6)


def test_apply_refuses_a_different_size():
    profile = FlatfieldProfile(flatfield=np.ones((4, 4), dtype=np.float32), offset=0.0)
    with pytest.raises(ValueError, match="does not match"):
        profile.apply(np.zeros((8, 8), dtype=np.uint16))


def test_save_and_load_round_trip(tmp_path):
    profile = fit_profile(
        tiles(8, known_shading()), offset=OFFSET, metadata={"detector": "PMT_top", "zoom": 1.5}
    )
    profile.save(tmp_path / "profile")
    loaded = FlatfieldProfile.load(tmp_path / "profile")
    np.testing.assert_array_equal(loaded.flatfield, profile.flatfield)
    assert loaded.offset == OFFSET
    assert loaded.metadata["detector"] == "PMT_top"
    assert loaded.metadata["algorithm"] == "basic-approximate"
    assert loaded.metadata["width"] == SIZE
    assert (tmp_path / "profile" / ff.PREVIEW_FILE).exists()


def confluent_tiles(count=80, size=64, seed=0):
    """Tissue everywhere, so no tile is an empty field, with a dim corner."""
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0:size, 0:size] / (size - 1.0)
    shading = 1.15 - 0.9 * (x**2 + y**2) / 2
    return [
        np.clip(
            OFFSET + rng.uniform(900, 1800) * shading + rng.normal(0, 20, (size, size)), 0, None
        ).astype(np.uint16)
        for _ in range(count)
    ]


def test_an_offset_read_from_tissue_is_refused_not_saved():
    with pytest.raises(ValueError, match="measured offset"):
        fit_profile(confluent_tiles())


def test_the_same_tiles_fit_with_a_measured_offset():
    profile = fit_profile(confluent_tiles(), offset=OFFSET)
    # The true corner is 0.25 / 0.85 of the mean
    assert profile.flatfield.min() == pytest.approx(0.29, abs=0.03)


def test_a_flatfield_at_or_below_zero_cannot_be_built_or_loaded(tmp_path):
    flat = np.ones((4, 4), dtype=np.float32)
    flat[0, 0] = 0.0
    with pytest.raises(ValueError, match="above zero"):
        FlatfieldProfile(flatfield=flat, offset=0.0)
    FlatfieldProfile(flatfield=np.ones((4, 4), dtype=np.float32), offset=0.0).save(tmp_path)
    tf.imwrite(tmp_path / ff.FLATFIELD_FILE, flat)
    with pytest.raises(ValueError, match="above zero"):
        FlatfieldProfile.load(tmp_path)


def test_fit_refuses_one_image_and_no_signal():
    with pytest.raises(ValueError, match="at least two"):
        fit_profile(tiles(1, known_shading()), offset=OFFSET)
    with pytest.raises(ValueError, match="No signal"):
        fit_profile([np.zeros((8, 8), dtype=np.uint16)] * 3, offset=100)


def test_images_with_nothing_in_common_are_refused_not_returned_as_nan():
    # Bright lines on nothing at all: every pixel is an outlier, so the common part collapses
    rng = np.random.default_rng(0)
    images = []
    for _ in range(12):
        image = np.full((SIZE, SIZE), int(OFFSET), dtype=np.uint16)
        for _ in range(40):
            y, x = rng.integers(0, SIZE, 2)
            image[y, max(0, x - 6) : x + 6] += int(rng.uniform(500, 3000))
        images.append(image)
    with pytest.raises(ValueError, match="nothing common"):
        fit_profile(images, offset=OFFSET)


def test_reads_a_micromanager_single_plane_series_by_channel(tmp_path):
    for position in range(3):
        folder = tmp_path / f"Pos{position}"
        folder.mkdir()
        for channel, value in ((0, 10), (1, 20)):
            name = f"img_channel00{channel}_position00{position}_time000000000_z000.tif"
            tf.imwrite(folder / name, np.full((8, 8), value, dtype=np.uint16))
    planes = read_tile_planes(tmp_path, channel=1)
    assert len(planes) == 3
    assert all(plane[0, 0] == 20 for plane in planes)


def test_reads_plain_tiles_and_skips_rgb(tmp_path):
    tf.imwrite(tmp_path / "0.tif", np.full((8, 8), 5, dtype=np.uint16))
    tf.imwrite(tmp_path / "1.tif", np.zeros((8, 8, 3), dtype=np.uint8))
    assert len(read_tile_planes(tmp_path)) == 1


def test_reading_samples_evenly(tmp_path):
    for index in range(10):
        tf.imwrite(tmp_path / f"{index}.tif", np.full((4, 4), index, dtype=np.uint16))
    assert len(read_tile_planes(tmp_path, max_images=4)) == 4


def test_empty_folder_is_an_error(tmp_path):
    with pytest.raises(FileNotFoundError):
        read_tile_planes(tmp_path)


def test_command_line_writes_a_profile(tmp_path, capsys):
    source = tmp_path / "tiles"
    source.mkdir()
    for index, tile in enumerate(tiles(10, known_shading())):
        tf.imwrite(source / f"{index}.tif", tile)
    ff.main([str(source), str(tmp_path / "profile"), "--offset", str(OFFSET)])
    assert "Fitted 10 of 10 planes" in capsys.readouterr().out
    assert (tmp_path / "profile" / ff.FLATFIELD_FILE).exists()
