"""Tests for Value at Risk backtesting."""

import numpy as np
import pytest

from src.backtesting import (
    KupiecTestResult,
    VaRBacktestResult,
    backtest_value_at_risk,
    basel_traffic_light_zone,
    identify_var_exceptions,
    kupiec_proportion_of_failures_test,
)


def _exceptions_with_count(
    count: int,
    observations: int = 250,
) -> np.ndarray:
    """Create an exception array with a known count."""
    exceptions = np.zeros(
        observations,
        dtype=bool,
    )

    exceptions[:count] = True

    return exceptions


def test_identify_exceptions_with_scalar_var() -> None:
    """A scalar VaR forecast should apply to every loss."""
    actual_losses = np.array(
        [50.0, 120.0, 100.0, 140.0]
    )

    result = identify_var_exceptions(
        actual_losses=actual_losses,
        var_forecasts=100.0,
    )

    expected = np.array(
        [False, True, False, True]
    )

    np.testing.assert_array_equal(result, expected)


def test_identify_exceptions_with_varying_var() -> None:
    """Every day may have a different VaR forecast."""
    actual_losses = np.array(
        [50.0, 120.0, 90.0, 140.0]
    )

    forecasts = np.array(
        [60.0, 100.0, 100.0, 150.0]
    )

    result = identify_var_exceptions(
        actual_losses,
        forecasts,
    )

    expected = np.array(
        [False, True, False, False]
    )

    np.testing.assert_array_equal(result, expected)


def test_loss_equal_to_var_is_not_exception() -> None:
    """Only losses strictly above VaR are exceptions."""
    result = identify_var_exceptions(
        actual_losses=np.array([100.0]),
        var_forecasts=100.0,
    )

    assert not result[0]


def test_kupiec_known_result() -> None:
    """Kupiec test should match a known numerical result."""
    exceptions = _exceptions_with_count(3)

    result = kupiec_proportion_of_failures_test(
        exceptions=exceptions,
        confidence_level=0.99,
        significance_level=0.05,
    )

    assert result.likelihood_ratio == pytest.approx(
        0.09494012266443264
    )
    assert result.p_value == pytest.approx(
        0.75798832137329
    )


def test_kupiec_accepts_reasonable_exception_count() -> None:
    """Three exceptions should not reject 99% VaR."""
    exceptions = _exceptions_with_count(3)

    result = kupiec_proportion_of_failures_test(
        exceptions,
        confidence_level=0.99,
    )

    assert not result.rejected


def test_kupiec_rejects_zero_exceptions() -> None:
    """Zero exceptions may indicate an overly conservative model."""
    exceptions = _exceptions_with_count(0)

    result = kupiec_proportion_of_failures_test(
        exceptions,
        confidence_level=0.99,
    )

    assert result.rejected
    assert result.p_value == pytest.approx(
        0.02498150305344973
    )


def test_kupiec_rejects_too_many_exceptions() -> None:
    """Many exceptions should reject an underestimated VaR."""
    exceptions = _exceptions_with_count(15)

    result = kupiec_proportion_of_failures_test(
        exceptions,
        confidence_level=0.99,
    )

    assert result.rejected
    assert result.p_value < 0.05


def test_kupiec_handles_all_exceptions() -> None:
    """The likelihood calculation must handle a rate of one."""
    exceptions = np.ones(250, dtype=bool)

    result = kupiec_proportion_of_failures_test(
        exceptions,
        confidence_level=0.99,
    )

    assert np.isfinite(result.likelihood_ratio)
    assert result.rejected


def test_basel_green_zone_with_zero_exceptions() -> None:
    """Zero exceptions belong to the green zone."""
    exceptions = _exceptions_with_count(0)

    assert (
        basel_traffic_light_zone(exceptions)
        == "green"
    )


def test_basel_green_zone_with_four_exceptions() -> None:
    """Four exceptions are the upper green boundary."""
    exceptions = _exceptions_with_count(4)

    assert (
        basel_traffic_light_zone(exceptions)
        == "green"
    )


def test_basel_yellow_zone_with_five_exceptions() -> None:
    """Five exceptions begin the yellow zone."""
    exceptions = _exceptions_with_count(5)

    assert (
        basel_traffic_light_zone(exceptions)
        == "yellow"
    )


def test_basel_yellow_zone_with_nine_exceptions() -> None:
    """Nine exceptions are the upper yellow boundary."""
    exceptions = _exceptions_with_count(9)

    assert (
        basel_traffic_light_zone(exceptions)
        == "yellow"
    )


def test_basel_red_zone_with_ten_exceptions() -> None:
    """Ten exceptions begin the red zone."""
    exceptions = _exceptions_with_count(10)

    assert (
        basel_traffic_light_zone(exceptions)
        == "red"
    )


def test_basel_not_applicable_for_other_sample_size() -> None:
    """The regulatory classification requires 250 days."""
    exceptions = np.zeros(100, dtype=bool)

    assert (
        basel_traffic_light_zone(exceptions)
        == "not_applicable"
    )


def test_basel_not_applicable_for_other_confidence() -> None:
    """The regulatory classification requires 99% VaR."""
    exceptions = np.zeros(250, dtype=bool)

    assert (
        basel_traffic_light_zone(
            exceptions,
            confidence_level=0.95,
        )
        == "not_applicable"
    )


def test_complete_backtest_result() -> None:
    """Complete backtest should return known statistics."""
    actual_losses = np.array(
        [0.0, 5.0, 12.0, 3.0]
    )

    result = backtest_value_at_risk(
        actual_losses=actual_losses,
        var_forecasts=10.0,
        confidence_level=0.75,
        significance_level=0.05,
    )

    assert result.observations == 4
    assert result.expected_exceptions == pytest.approx(1.0)
    assert result.actual_exceptions == 1
    assert result.exception_rate == pytest.approx(0.25)
    assert result.basel_zone == "not_applicable"


def test_backtest_returns_dataclasses() -> None:
    """Backtest and Kupiec outputs should be structured."""
    result = backtest_value_at_risk(
        actual_losses=np.array(
            [0.0, 5.0, 12.0, 3.0]
        ),
        var_forecasts=10.0,
        confidence_level=0.75,
    )

    assert isinstance(result, VaRBacktestResult)
    assert isinstance(result.kupiec_test, KupiecTestResult)


def test_two_dimensional_losses_raise_value_error() -> None:
    """Actual losses must represent one portfolio."""
    losses = np.array(
        [
            [10.0, 20.0],
            [30.0, 40.0],
        ]
    )

    with pytest.raises(
        ValueError,
        match="Actual losses must be a one-dimensional array",
    ):
        identify_var_exceptions(
            losses,
            var_forecasts=100.0,
        )


def test_empty_losses_raise_value_error() -> None:
    """Loss history cannot be empty."""
    with pytest.raises(
        ValueError,
        match="Actual losses cannot be empty",
    ):
        identify_var_exceptions(
            np.array([]),
            var_forecasts=100.0,
        )


def test_non_finite_losses_raise_value_error() -> None:
    """Actual losses must contain finite values."""
    losses = np.array(
        [10.0, np.nan, 20.0]
    )

    with pytest.raises(
        ValueError,
        match=(
            "Actual losses must contain "
            "only finite values"
        ),
    ):
        identify_var_exceptions(
            losses,
            var_forecasts=100.0,
        )


def test_forecast_length_mismatch_raises_value_error() -> None:
    """Every loss must have a corresponding VaR forecast."""
    losses = np.array([10.0, 20.0, 30.0])
    forecasts = np.array([100.0, 100.0])

    with pytest.raises(
        ValueError,
        match=(
            "VaR forecasts and actual losses "
            "must have the same length"
        ),
    ):
        identify_var_exceptions(
            losses,
            forecasts,
        )


def test_negative_var_forecast_raises_value_error() -> None:
    """Reported VaR forecasts cannot be negative."""
    with pytest.raises(
        ValueError,
        match="VaR forecasts cannot be negative",
    ):
        identify_var_exceptions(
            actual_losses=np.array([10.0, 20.0]),
            var_forecasts=-100.0,
        )


def test_invalid_exception_values_raise_value_error() -> None:
    """Exception indicators must contain zero or one."""
    exceptions = np.array([0, 1, 2])

    with pytest.raises(
        ValueError,
        match=(
            "Exceptions must contain "
            "only boolean values"
        ),
    ):
        kupiec_proportion_of_failures_test(
            exceptions,  # type: ignore[arg-type]
            confidence_level=0.99,
        )


@pytest.mark.parametrize(
    "confidence_level",
    [0.0, 1.0],
)
def test_invalid_confidence_level_raises_value_error(
    confidence_level: float,
) -> None:
    """Confidence must be strictly between zero and one."""
    exceptions = np.zeros(250, dtype=bool)

    with pytest.raises(
        ValueError,
        match="Confidence level must be between 0 and 1",
    ):
        kupiec_proportion_of_failures_test(
            exceptions,
            confidence_level=confidence_level,
        )