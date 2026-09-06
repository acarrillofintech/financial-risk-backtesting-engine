"""Tests for historical financial risk metrics."""

import numpy as np
import pytest

from src.risk_metrics import (
    HistoricalRiskMetrics,
    calculate_historical_risk_metrics,
    historical_expected_shortfall,
    historical_value_at_risk,
    returns_to_losses,
)


@pytest.fixture
def sample_returns() -> np.ndarray:
    """Provide a known sample of portfolio returns."""
    return np.array(
        [
            0.020,
            -0.010,
            0.005,
            -0.030,
            0.010,
            -0.020,
            0.015,
            -0.040,
            0.000,
            -0.005,
        ]
    )


@pytest.fixture
def sample_losses() -> np.ndarray:
    """Provide the monetary losses for the sample returns."""
    return np.array(
        [
            -2_000.0,
            1_000.0,
            -500.0,
            3_000.0,
            -1_000.0,
            2_000.0,
            -1_500.0,
            4_000.0,
            0.0,
            500.0,
        ]
    )


def test_returns_to_losses(
    sample_returns: np.ndarray,
    sample_losses: np.ndarray,
) -> None:
    """Returns should be converted to monetary losses."""
    result = returns_to_losses(
        returns=sample_returns,
        portfolio_value=100_000.0,
    )

    np.testing.assert_allclose(result, sample_losses)


def test_historical_value_at_risk(
    sample_losses: np.ndarray,
) -> None:
    """Historical VaR should match the known quantile."""
    result = historical_value_at_risk(
        losses=sample_losses,
        confidence_level=0.95,
    )

    assert result == pytest.approx(3_550.0)


def test_historical_expected_shortfall(
    sample_losses: np.ndarray,
) -> None:
    """Expected Shortfall should average losses beyond VaR."""
    result = historical_expected_shortfall(
        losses=sample_losses,
        confidence_level=0.95,
    )

    assert result == pytest.approx(4_000.0)


def test_calculate_historical_risk_metrics(
    sample_returns: np.ndarray,
) -> None:
    """Combined calculation should return all known metrics."""
    result = calculate_historical_risk_metrics(
        returns=sample_returns,
        portfolio_value=100_000.0,
        confidence_level=0.95,
    )

    assert result.confidence_level == pytest.approx(0.95)
    assert result.value_at_risk == pytest.approx(3_550.0)
    assert result.expected_shortfall == pytest.approx(
        4_000.0
    )
    assert result.maximum_loss == pytest.approx(4_000.0)
    assert result.average_loss == pytest.approx(550.0)


def test_result_is_dataclass(
    sample_returns: np.ndarray,
) -> None:
    """Risk calculation should return a structured result."""
    result = calculate_historical_risk_metrics(
        returns=sample_returns,
        portfolio_value=100_000.0,
        confidence_level=0.95,
    )

    assert isinstance(result, HistoricalRiskMetrics)


def test_expected_shortfall_is_not_less_than_var(
    sample_losses: np.ndarray,
) -> None:
    """Tail-average loss should not be below the VaR threshold."""
    value_at_risk = historical_value_at_risk(
        sample_losses,
        confidence_level=0.90,
    )

    expected_shortfall = historical_expected_shortfall(
        sample_losses,
        confidence_level=0.90,
    )

    assert expected_shortfall >= value_at_risk


def test_two_dimensional_returns_raise_value_error() -> None:
    """Returns must be represented by one portfolio series."""
    returns = np.array(
        [
            [0.01, 0.02],
            [-0.01, 0.03],
        ]
    )

    with pytest.raises(
        ValueError,
        match="Returns must be a one-dimensional array",
    ):
        returns_to_losses(returns)


def test_empty_returns_raise_value_error() -> None:
    """Return history cannot be empty."""
    with pytest.raises(
        ValueError,
        match="Returns cannot be empty",
    ):
        returns_to_losses(np.array([]))


def test_non_finite_returns_raise_value_error() -> None:
    """Returns cannot contain missing or infinite values."""
    returns = np.array([0.01, np.nan, -0.02])

    with pytest.raises(
        ValueError,
        match="Returns must contain only finite values",
    ):
        returns_to_losses(returns)


@pytest.mark.parametrize(
    "portfolio_value",
    [0.0, -1_000.0, np.inf],
)
def test_invalid_portfolio_value_raises_value_error(
    portfolio_value: float,
    sample_returns: np.ndarray,
) -> None:
    """Portfolio value must be positive and finite."""
    with pytest.raises(
        ValueError,
        match=(
            "Portfolio value must be "
            "a positive finite number"
        ),
    ):
        returns_to_losses(
            sample_returns,
            portfolio_value=portfolio_value,
        )


def test_non_numeric_portfolio_value_raises_type_error(
    sample_returns: np.ndarray,
) -> None:
    """Portfolio value must be numeric."""
    with pytest.raises(
        TypeError,
        match="Portfolio value must be a real number",
    ):
        returns_to_losses(
            sample_returns,
            portfolio_value="invalid",  # type: ignore[arg-type]
        )


@pytest.mark.parametrize(
    "confidence_level",
    [0.0, 1.0, -0.10, 1.10],
)
def test_invalid_confidence_level_raises_value_error(
    confidence_level: float,
    sample_losses: np.ndarray,
) -> None:
    """Confidence level must be strictly between zero and one."""
    with pytest.raises(
        ValueError,
        match="Confidence level must be between 0 and 1",
    ):
        historical_value_at_risk(
            losses=sample_losses,
            confidence_level=confidence_level,
        )


def test_boolean_confidence_level_raises_type_error(
    sample_losses: np.ndarray,
) -> None:
    """Boolean values are not valid confidence levels."""
    with pytest.raises(
        TypeError,
        match="Confidence level must be a real number",
    ):
        historical_value_at_risk(
            losses=sample_losses,
            confidence_level=True,
        )