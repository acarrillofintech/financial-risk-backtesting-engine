"""Tests for rolling Value at Risk forecasts."""

import numpy as np
import pytest
from scipy.stats import norm

from src.rolling_var import (
    RollingVaRForecasts,
    calculate_rolling_var_forecasts,
    rolling_historical_var,
    rolling_parametric_var,
)


def sample_returns() -> np.ndarray:
    """Return a small deterministic return series."""
    return np.array(
        [0.01, -0.02, 0.005, -0.01, 0.03],
        dtype=float,
    )


def test_historical_var_known_values() -> None:
    """Historical rolling VaR should match manual values."""
    result = rolling_historical_var(
        returns=sample_returns(),
        portfolio_value=1_000.0,
        window_size=3,
        confidence_level=0.95,
    )

    expected = np.array([17.5, 19.0])

    assert result == pytest.approx(expected)


def test_parametric_var_known_values() -> None:
    """Parametric rolling VaR should match its formula."""
    returns = sample_returns()
    portfolio_value = 1_000.0
    confidence_level = 0.95
    quantile = norm.ppf(confidence_level)

    result = rolling_parametric_var(
        returns=returns,
        portfolio_value=portfolio_value,
        window_size=3,
        confidence_level=confidence_level,
    )

    expected = []

    for index in range(2):
        window = returns[index:index + 3]
        forecast = portfolio_value * (
            quantile * np.std(window, ddof=1)
            - np.mean(window)
        )
        expected.append(max(float(forecast), 0.0))

    assert result == pytest.approx(expected)


def test_forecast_count() -> None:
    """Forecast count should equal observations minus window."""
    returns = sample_returns()

    result = rolling_historical_var(
        returns=returns,
        portfolio_value=1_000.0,
        window_size=3,
        confidence_level=0.95,
    )

    assert len(result) == len(returns) - 3


def test_actual_losses_are_aligned() -> None:
    """Actual losses should begin after the first window."""
    result = calculate_rolling_var_forecasts(
        returns=sample_returns(),
        portfolio_value=1_000.0,
        window_size=3,
        confidence_level=0.95,
    )

    assert result.actual_losses == pytest.approx(
        np.array([10.0, -30.0])
    )


def test_combined_result_is_dataclass() -> None:
    """Combined calculation should return its dataclass."""
    result = calculate_rolling_var_forecasts(
        returns=sample_returns(),
        portfolio_value=1_000.0,
        window_size=3,
        confidence_level=0.95,
    )

    assert isinstance(result, RollingVaRForecasts)
    assert result.window_size == 3
    assert result.confidence_level == 0.95
    assert result.portfolio_value == 1_000.0


def test_combined_forecasts_have_equal_lengths() -> None:
    """Losses and both forecasts should align."""
    result = calculate_rolling_var_forecasts(
        returns=sample_returns(),
        portfolio_value=1_000.0,
        window_size=3,
        confidence_level=0.95,
    )

    assert len(result.actual_losses) == 2
    assert len(result.historical_var) == 2
    assert len(result.parametric_var) == 2


def test_forecasts_do_not_use_future_return() -> None:
    """Changing the final return must not alter its VaR forecast."""
    original_returns = sample_returns()
    changed_returns = original_returns.copy()
    changed_returns[-1] = -0.50

    original = calculate_rolling_var_forecasts(
        returns=original_returns,
        portfolio_value=1_000.0,
        window_size=3,
        confidence_level=0.95,
    )

    changed = calculate_rolling_var_forecasts(
        returns=changed_returns,
        portfolio_value=1_000.0,
        window_size=3,
        confidence_level=0.95,
    )

    assert changed.historical_var == pytest.approx(
        original.historical_var
    )
    assert changed.parametric_var == pytest.approx(
        original.parametric_var
    )
    assert (
        changed.actual_losses[-1]
        != original.actual_losses[-1]
    )


def test_forecasts_are_nonnegative() -> None:
    """Reported VaR forecasts should never be negative."""
    positive_returns = np.array(
        [0.01, 0.02, 0.03, 0.04],
        dtype=float,
    )

    historical = rolling_historical_var(
        returns=positive_returns,
        portfolio_value=1_000.0,
        window_size=2,
        confidence_level=0.50,
    )

    parametric = rolling_parametric_var(
        returns=positive_returns,
        portfolio_value=1_000.0,
        window_size=2,
        confidence_level=0.50,
    )

    assert np.all(historical >= 0.0)
    assert np.all(parametric >= 0.0)


def test_two_dimensional_returns_raise_value_error() -> None:
    """Returns must be one-dimensional."""
    with pytest.raises(
        ValueError,
        match="one-dimensional",
    ):
        rolling_historical_var(
            returns=np.ones((3, 2)),
            portfolio_value=1_000.0,
            window_size=2,
        )


def test_too_few_returns_raise_value_error() -> None:
    """At least three returns should be required."""
    with pytest.raises(
        ValueError,
        match="At least three",
    ):
        rolling_historical_var(
            returns=np.array([0.01, 0.02]),
            portfolio_value=1_000.0,
            window_size=1,
        )


def test_non_finite_returns_raise_value_error() -> None:
    """Returns cannot contain NaN or infinity."""
    with pytest.raises(
        ValueError,
        match="finite values",
    ):
        rolling_historical_var(
            returns=np.array([0.01, np.nan, 0.02]),
            portfolio_value=1_000.0,
            window_size=2,
        )


@pytest.mark.parametrize("window_size", [0, 1, -1])
def test_invalid_window_size_raises_value_error(
    window_size: int,
) -> None:
    """Window size must contain at least two observations."""
    with pytest.raises(
        ValueError,
        match="at least 2",
    ):
        rolling_historical_var(
            returns=sample_returns(),
            portfolio_value=1_000.0,
            window_size=window_size,
        )


def test_window_equal_to_observations_raises_value_error() -> None:
    """A window must leave at least one forecast day."""
    returns = sample_returns()

    with pytest.raises(
        ValueError,
        match="smaller",
    ):
        rolling_historical_var(
            returns=returns,
            portfolio_value=1_000.0,
            window_size=len(returns),
        )


@pytest.mark.parametrize("window_size", [2.5, True, "3"])
def test_non_integer_window_raises_type_error(
    window_size: object,
) -> None:
    """Window size must be an integer."""
    with pytest.raises(
        TypeError,
        match="integer",
    ):
        rolling_historical_var(
            returns=sample_returns(),
            portfolio_value=1_000.0,
            window_size=window_size,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize(
    "confidence_level",
    [0.0, 1.0, -0.1, 1.1],
)
def test_invalid_confidence_level_raises_value_error(
    confidence_level: float,
) -> None:
    """Confidence must be strictly between zero and one."""
    with pytest.raises(
        ValueError,
        match="between 0 and 1",
    ):
        rolling_historical_var(
            returns=sample_returns(),
            portfolio_value=1_000.0,
            window_size=3,
            confidence_level=confidence_level,
        )


@pytest.mark.parametrize(
    "portfolio_value",
    [0.0, -1_000.0, np.inf, np.nan],
)
def test_invalid_portfolio_value_raises_value_error(
    portfolio_value: float,
) -> None:
    """Portfolio value must be finite and positive."""
    with pytest.raises(
        ValueError,
        match="finite and positive",
    ):
        rolling_historical_var(
            returns=sample_returns(),
            portfolio_value=portfolio_value,
            window_size=3,
        )


def test_non_numeric_portfolio_value_raises_type_error() -> None:
    """Portfolio value must be numeric."""
    with pytest.raises(
        TypeError,
        match="real number",
    ):
        rolling_historical_var(
            returns=sample_returns(),
            portfolio_value="1000",  # type: ignore[arg-type]
            window_size=3,
        )