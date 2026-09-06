"""Tests for normal parametric risk models."""

import numpy as np
import pytest

from src.var_models import (
    ParametricRiskMetrics,
    calculate_parametric_risk_metrics,
    parametric_expected_shortfall,
    parametric_value_at_risk,
)


@pytest.fixture
def symmetric_returns() -> np.ndarray:
    """Provide returns with a known mean and volatility."""
    return np.array(
        [-0.02, -0.01, 0.00, 0.01, 0.02]
    )


def test_parametric_value_at_risk(
    symmetric_returns: np.ndarray,
) -> None:
    """Parametric VaR should match the known normal result."""
    result = parametric_value_at_risk(
        returns=symmetric_returns,
        portfolio_value=1_000_000.0,
        confidence_level=0.95,
        time_horizon_days=1,
    )

    assert result == pytest.approx(
        26_007.41939377787
    )


def test_parametric_expected_shortfall(
    symmetric_returns: np.ndarray,
) -> None:
    """Parametric ES should match the known normal result."""
    result = parametric_expected_shortfall(
        returns=symmetric_returns,
        portfolio_value=1_000_000.0,
        confidence_level=0.95,
        time_horizon_days=1,
    )

    assert result == pytest.approx(
        32_614.35315261968
    )


def test_combined_result_is_dataclass(
    symmetric_returns: np.ndarray,
) -> None:
    """Combined calculation should return structured metrics."""
    result = calculate_parametric_risk_metrics(
        returns=symmetric_returns,
        portfolio_value=1_000_000.0,
        confidence_level=0.95,
        time_horizon_days=1,
    )

    assert isinstance(result, ParametricRiskMetrics)


def test_combined_result_contains_distribution_estimates(
    symmetric_returns: np.ndarray,
) -> None:
    """The result should contain sample mean and volatility."""
    result = calculate_parametric_risk_metrics(
        returns=symmetric_returns,
        portfolio_value=1_000_000.0,
        confidence_level=0.95,
        time_horizon_days=1,
    )

    assert result.mean_return == pytest.approx(0.0)
    assert result.volatility == pytest.approx(
        0.015811388300841896
    )
    assert result.confidence_level == pytest.approx(0.95)
    assert result.time_horizon_days == 1


def test_expected_shortfall_exceeds_value_at_risk(
    symmetric_returns: np.ndarray,
) -> None:
    """Expected tail loss should be greater than VaR."""
    value_at_risk = parametric_value_at_risk(
        returns=symmetric_returns,
        portfolio_value=1_000_000.0,
        confidence_level=0.95,
    )

    expected_shortfall = parametric_expected_shortfall(
        returns=symmetric_returns,
        portfolio_value=1_000_000.0,
        confidence_level=0.95,
    )

    assert expected_shortfall > value_at_risk


def test_square_root_of_time_scaling(
    symmetric_returns: np.ndarray,
) -> None:
    """Zero-mean VaR should scale with the square root of time."""
    one_day_var = parametric_value_at_risk(
        returns=symmetric_returns,
        portfolio_value=1_000_000.0,
        confidence_level=0.95,
        time_horizon_days=1,
    )

    ten_day_var = parametric_value_at_risk(
        returns=symmetric_returns,
        portfolio_value=1_000_000.0,
        confidence_level=0.95,
        time_horizon_days=10,
    )

    assert ten_day_var == pytest.approx(
        one_day_var * np.sqrt(10)
    )


def test_two_dimensional_returns_raise_value_error() -> None:
    """Returns must represent one portfolio series."""
    returns = np.array(
        [
            [0.01, 0.02],
            [-0.01, 0.00],
        ]
    )

    with pytest.raises(
        ValueError,
        match="Returns must be a one-dimensional array",
    ):
        parametric_value_at_risk(returns)


def test_single_return_raises_value_error() -> None:
    """At least two observations are required."""
    with pytest.raises(
        ValueError,
        match="Returns must contain at least two observations",
    ):
        parametric_value_at_risk(np.array([0.01]))


def test_non_finite_returns_raise_value_error() -> None:
    """Returns must not contain missing values."""
    returns = np.array([0.01, np.nan, -0.02])

    with pytest.raises(
        ValueError,
        match="Returns must contain only finite values",
    ):
        parametric_value_at_risk(returns)


@pytest.mark.parametrize(
    "confidence_level",
    [0.0, 1.0, -0.10, 1.10],
)
def test_invalid_confidence_level_raises_value_error(
    confidence_level: float,
    symmetric_returns: np.ndarray,
) -> None:
    """Confidence must be strictly between zero and one."""
    with pytest.raises(
        ValueError,
        match="Confidence level must be between 0 and 1",
    ):
        parametric_value_at_risk(
            returns=symmetric_returns,
            confidence_level=confidence_level,
        )


def test_boolean_confidence_level_raises_type_error(
    symmetric_returns: np.ndarray,
) -> None:
    """Boolean confidence values must be rejected."""
    with pytest.raises(
        TypeError,
        match="Confidence level must be a real number",
    ):
        parametric_value_at_risk(
            returns=symmetric_returns,
            confidence_level=True,
        )


@pytest.mark.parametrize(
    "time_horizon_days",
    [0, -1],
)
def test_invalid_time_horizon_raises_value_error(
    time_horizon_days: int,
    symmetric_returns: np.ndarray,
) -> None:
    """Risk horizon must be positive."""
    with pytest.raises(
        ValueError,
        match="Time horizon must be greater than zero",
    ):
        parametric_value_at_risk(
            returns=symmetric_returns,
            time_horizon_days=time_horizon_days,
        )


def test_non_integer_time_horizon_raises_type_error(
    symmetric_returns: np.ndarray,
) -> None:
    """Risk horizon must contain a whole number of days."""
    with pytest.raises(
        TypeError,
        match="Time horizon must be an integer",
    ):
        parametric_value_at_risk(
            returns=symmetric_returns,
            time_horizon_days=2.5,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize(
    "portfolio_value",
    [0.0, -1_000.0, np.inf],
)
def test_invalid_portfolio_value_raises_value_error(
    portfolio_value: float,
    symmetric_returns: np.ndarray,
) -> None:
    """Portfolio value must be positive and finite."""
    with pytest.raises(
        ValueError,
        match=(
            "Portfolio value must be "
            "a positive finite number"
        ),
    ):
        parametric_value_at_risk(
            returns=symmetric_returns,
            portfolio_value=portfolio_value,
        )


def test_constant_returns_raise_value_error() -> None:
    """Parametric risk requires positive volatility."""
    constant_returns = np.array(
        [0.01, 0.01, 0.01, 0.01]
    )

    with pytest.raises(
        ValueError,
        match="Return volatility must be greater than zero",
    ):
        parametric_value_at_risk(constant_returns)