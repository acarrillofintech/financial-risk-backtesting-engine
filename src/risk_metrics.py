"""Core historical financial risk metrics."""

from dataclasses import dataclass
from numbers import Real

import numpy as np
from numpy.typing import NDArray


FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class HistoricalRiskMetrics:
    """Historical risk measures for a portfolio."""

    confidence_level: float
    value_at_risk: float
    expected_shortfall: float
    maximum_loss: float
    average_loss: float


def _validate_numeric_array(
    values: FloatArray,
    name: str,
) -> FloatArray:
    """Validate and normalize a one-dimensional numeric array."""
    normalized_values = np.asarray(values, dtype=float)

    if normalized_values.ndim != 1:
        raise ValueError(
            f"{name} must be a one-dimensional array."
        )

    if normalized_values.size == 0:
        raise ValueError(f"{name} cannot be empty.")

    if not np.all(np.isfinite(normalized_values)):
        raise ValueError(
            f"{name} must contain only finite values."
        )

    return normalized_values


def _validate_confidence_level(
    confidence_level: float,
) -> float:
    """Validate a probability used as a confidence level."""
    if (
        isinstance(confidence_level, bool)
        or not isinstance(confidence_level, Real)
    ):
        raise TypeError(
            "Confidence level must be a real number."
        )

    normalized_confidence = float(confidence_level)

    if not 0.0 < normalized_confidence < 1.0:
        raise ValueError(
            "Confidence level must be between 0 and 1."
        )

    return normalized_confidence


def _validate_portfolio_value(
    portfolio_value: float,
) -> float:
    """Validate the current monetary value of a portfolio."""
    if (
        isinstance(portfolio_value, bool)
        or not isinstance(portfolio_value, Real)
    ):
        raise TypeError(
            "Portfolio value must be a real number."
        )

    normalized_value = float(portfolio_value)

    if (
        not np.isfinite(normalized_value)
        or normalized_value <= 0.0
    ):
        raise ValueError(
            "Portfolio value must be a positive finite number."
        )

    return normalized_value


def returns_to_losses(
    returns: FloatArray,
    portfolio_value: float = 1.0,
) -> FloatArray:
    """Convert portfolio returns into monetary losses.

    Positive values represent losses.
    Negative values represent gains.
    """
    normalized_returns = _validate_numeric_array(
        returns,
        "Returns",
    )

    normalized_portfolio_value = (
        _validate_portfolio_value(portfolio_value)
    )

    return -normalized_returns * normalized_portfolio_value


def historical_value_at_risk(
    losses: FloatArray,
    confidence_level: float = 0.95,
) -> float:
    """Calculate Historical Value at Risk from a loss distribution."""
    normalized_losses = _validate_numeric_array(
        losses,
        "Losses",
    )

    normalized_confidence = _validate_confidence_level(
        confidence_level
    )

    value_at_risk = np.quantile(
        normalized_losses,
        normalized_confidence,
        method="linear",
    )

    return float(value_at_risk)


def historical_expected_shortfall(
    losses: FloatArray,
    confidence_level: float = 0.95,
) -> float:
    """Calculate the average loss beyond Historical VaR."""
    normalized_losses = _validate_numeric_array(
        losses,
        "Losses",
    )

    normalized_confidence = _validate_confidence_level(
        confidence_level
    )

    value_at_risk = historical_value_at_risk(
        normalized_losses,
        normalized_confidence,
    )

    tail_losses = normalized_losses[
        normalized_losses >= value_at_risk
    ]

    if tail_losses.size == 0:
        raise RuntimeError(
            "Expected Shortfall tail cannot be empty."
        )

    return float(np.mean(tail_losses))


def calculate_historical_risk_metrics(
    returns: FloatArray,
    portfolio_value: float = 1.0,
    confidence_level: float = 0.95,
) -> HistoricalRiskMetrics:
    """Calculate the principal historical portfolio risk metrics."""
    normalized_confidence = _validate_confidence_level(
        confidence_level
    )

    losses = returns_to_losses(
        returns=returns,
        portfolio_value=portfolio_value,
    )

    value_at_risk = historical_value_at_risk(
        losses=losses,
        confidence_level=normalized_confidence,
    )

    expected_shortfall = historical_expected_shortfall(
        losses=losses,
        confidence_level=normalized_confidence,
    )

    return HistoricalRiskMetrics(
        confidence_level=normalized_confidence,
        value_at_risk=value_at_risk,
        expected_shortfall=expected_shortfall,
        maximum_loss=float(np.max(losses)),
        average_loss=float(np.mean(losses)),
    )


def main() -> None:
    """Run a reproducible historical-risk demonstration."""
    random_generator = np.random.default_rng(seed=42)

    daily_returns = random_generator.normal(
        loc=0.0004,
        scale=0.012,
        size=1_000,
    )

    portfolio_value = 1_000_000.0
    confidence_level = 0.95

    metrics = calculate_historical_risk_metrics(
        returns=daily_returns,
        portfolio_value=portfolio_value,
        confidence_level=confidence_level,
    )

    print("Historical portfolio risk")
    print(f"Observations: {len(daily_returns):,}")
    print(f"Portfolio value: ${portfolio_value:,.2f}")
    print(
        f"Confidence level: "
        f"{metrics.confidence_level:.2%}"
    )

    print("\nRisk metrics")
    print(
        f"Historical VaR: "
        f"${metrics.value_at_risk:,.2f}"
    )
    print(
        f"Expected Shortfall: "
        f"${metrics.expected_shortfall:,.2f}"
    )
    print(
        f"Maximum historical loss: "
        f"${metrics.maximum_loss:,.2f}"
    )
    print(
        f"Average loss: "
        f"${metrics.average_loss:,.2f}"
    )


if __name__ == "__main__":
    main()