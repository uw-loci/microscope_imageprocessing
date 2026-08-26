"""The resample-policy contract, especially its fail-safe direction.

Every rule here exists because violating it produces a well-formed image that
is silently meaningless, rather than an error.
"""

import pytest

from microscope_imageprocessing.io import (
    RESAMPLE_ANGULAR_180,
    RESAMPLE_KEY,
    RESAMPLE_LINEAR,
    RESAMPLE_NEAREST,
    RESAMPLE_REASON_KEY,
    channel_handling,
    may_combine,
    resample_policy,
)


class TestDefaults:
    @pytest.mark.parametrize("annotations", [None, {}, {"unrelated": "x"}])
    def test_absent_metadata_means_linear(self, annotations):
        """Every channel written before this convention existed is continuous,
        and every existing reader already averages it."""
        assert resample_policy(annotations) == RESAMPLE_LINEAR
        assert may_combine(annotations) is True

    def test_explicit_linear_permits_combining(self):
        assert may_combine({RESAMPLE_KEY: RESAMPLE_LINEAR}) is True


class TestFailSafe:
    """The whole point of the design: unknown means don't touch."""

    @pytest.mark.parametrize(
        "value",
        [
            RESAMPLE_NEAREST,
            RESAMPLE_ANGULAR_180,
            "angular360",
            "label",  # plausible future addition
            "something-nobody-has-written-yet",
            "",
        ],
    )
    def test_anything_but_linear_forbids_combining(self, value):
        assert may_combine({RESAMPLE_KEY: value}) is False

    def test_a_future_policy_degrades_to_safe_not_to_corruption(self):
        """A reader built today, handed a file written by a newer writer, must
        preserve the data rather than average it because it did not recognise
        the label."""
        future = {RESAMPLE_KEY: "quaternion", RESAMPLE_REASON_KEY: "not invented yet"}
        assert may_combine(future) is False
        assert resample_policy(future) == "quaternion"  # reported verbatim, for logging

    @pytest.mark.parametrize("value", ["LINEAR", "Linear", "  linear  "])
    def test_case_and_whitespace_do_not_accidentally_forbid(self, value):
        """Being strict here would flip a continuous channel to nearest and
        quietly degrade every pyramid built from it."""
        assert may_combine({RESAMPLE_KEY: value}) is True


class TestBuilder:
    def test_builds_the_entries(self):
        got = channel_handling(RESAMPLE_ANGULAR_180, reason="axial: 0 and 180 are the same axis")
        assert got[RESAMPLE_KEY] == RESAMPLE_ANGULAR_180
        assert "axial" in got[RESAMPLE_REASON_KEY]

    def test_reason_is_optional(self):
        assert channel_handling(RESAMPLE_NEAREST) == {RESAMPLE_KEY: RESAMPLE_NEAREST}

    def test_unknown_policy_is_rejected_at_write_time(self):
        """Declaring a policy no reader understands is worse than declaring
        none, because it reads as deliberate."""
        with pytest.raises(ValueError, match="unknown resample policy"):
            channel_handling("mean-ish")

    def test_round_trips_through_may_combine(self):
        assert may_combine(channel_handling(RESAMPLE_LINEAR)) is True
        assert may_combine(channel_handling(RESAMPLE_NEAREST)) is False
