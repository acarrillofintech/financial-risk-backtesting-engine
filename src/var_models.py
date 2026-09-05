"""Parametric Value at Risk models."""

from dataclasses import dataclass
from numbers import Real

import numpy as np
from numpy.typing import NDArray
from scipy.stats import norm


FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class ParametricRiskMetrics:
    """Risk metrics produced by the normal parametric model."""

    confidence_level: float
    time_horizon_days: int
    mean_return: float
    volatility: float
    value_at_risk: float
    expected_shortfall: float


def _validate_returns(
    returns: FloatArray,
) -> FloatArray:
    """Validate a portfolio return series."""
    normalized_returns = np.asarray(returns, dtype=float)

    if normalized_returns.ndim != 1:
        raise ValueError(
            "Returns must be a one-dimensional array."
        )

    if normalized_returns.size < 2:
        raise ValueError(
            "Returns must contain at least two observations."
        )

    if not np.all(np.isfinite(normalized_returns)):
        raise ValueError(
            "Returns must contain only finite values."
        )

    return normalized_returns


def _validate_confidence_level(
    confidence_level: float,
) -> float:
    """Validate the confidence level."""
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
    """Validate the portfolio monetary value."""
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


def _validate_time_horizon(
    time_horizon_days: int,
) -> int:
    """Validate the risk-measurement time horizon."""
    if (
        isinstance(time_horizon_days, bool)
        or not isinstance(
            time_horizon_days,
            (int, np.integer),
        )
    ):
        raise TypeError(
            "Time horizon must be an integer."
        )

    normalized_horizon = int(time_horizon_days)

    if normalized_horizon <= 0:
        raise ValueError(
            "Time horizon must be greater than zero."
        )

    return normalized_horizon


def _estimate_normal_parameters(
    returns: FloatArray,
) -> tuple[float, float]:
    """Estimate daily mean and sample volatility."""
    normalized_returns = _validate_returns(returns)

    mean_return = float(np.mean(normalized_returns))
    volatility = float(
        np.std(normalized_returns, ddof=1)
    )

    if volatility <= 0.0:
        raise ValueError(
            "Return volatility must be greater than zero."
        )

    return mean_return, volatility


def parametric_value_at_risk(
    returns: FloatArray,
    portfolio_value: float = 1.0,
    confidence_level: float = 0.95,
    time_horizon_days: int = 1,
) -> float:
    """Calculate normal parametric Value at Risk."""
    mean_return, volatility = (
        _estimate_normal_parameters(returns)
    )

    normalized_value = _validate_portfolio_value(
        portfolio_value
    )

    normalized_confidence = _validate_confidence_level(
        confidence_level
    )

    normalized_horizon = _validate_time_horizon(
        time_horizon_days
    )

    critical_value = float(
        norm.ppf(normalized_confidence)
    )

    horizon_mean = (
        mean_return * normalized_horizon
    )

    horizon_volatility = (
        volatility * np.sqrt(normalized_horizon)
    )

    value_at_risk = normalized_value * (
        critical_value * horizon_volatility
        - horizon_mean
    )

    return float(value_at_risk)


def parametric_expected_shortfall(
    returns: FloatArray,
    portfolio_value: float = 1.0,
    confidence_level: float = 0.95,
    time_horizon_days: int = 1,
) -> float:
    """Calculate normal parametric Expected Shortfall."""
    mean_return, volatility = (
        _estimate_normal_parameters(returns)
    )

    normalized_value = _validate_portfolio_value(
        portfolio_value
    )

    normalized_confidence = _validate_confidence_level(
        confidence_level
    )

    normalized_horizon = _validate_time_horizon(
        time_horizon_days
    )

    critical_value = float(
        norm.ppf(normalized_confidence)
    )

    tail_density = float(
        norm.pdf(critical_value)
    )

    horizon_mean = (
        mean_return * normalized_horizon
    )

    horizon_volatility = (
        volatility * np.sqrt(normalized_horizon)
    )

    expected_shortfall = normalized_value * (
        (
            horizon_volatility
            * tail_density
            / (1.0 - normalized_confidence)
        )
        - horizon_mean
    )

    return float(expected_shortfall)


def calculate_parametric_risk_metrics(
    returns: FloatArray,
    portfolio_value: float = 1.0,
    confidence_level: float = 0.95,
    time_horizon_days: int = 1,
) -> ParametricRiskMetrics:
    """Calculate all normal parametric risk metrics."""
    normalized_returns = _validate_returns(returns)

    normalized_value = _validate_portfolio_value(
        portfolio_value
    )

    normalized_confidence = _validate_confidence_level(
        confidence_level
    )

    normalized_horizon = _validate_time_horizon(
        time_horizon_days
    )

    mean_return, volatility = (
        _estimate_normal_parameters(
            normalized_returns
        )
    )

    value_at_risk = parametric_value_at_risk(
        returns=normalized_returns,
        portfolio_value=normalized_value,
        confidence_level=normalized_confidence,
        time_horizon_days=normalized_horizon,
    )

    expected_shortfall = parametric_expected_shortfall(
        returns=normalized_returns,
        portfolio_value=normalized_value,
        confidence_level=normalized_confidence,
        time_horizon_days=normalized_horizon,
    )

    return ParametricRiskMetrics(
        confidence_level=normalized_confidence,
        time_horizon_days=normalized_horizon,
        mean_return=mean_return,
        volatility=volatility,
        value_at_risk=value_at_risk,
        expected_shortfall=expected_shortfall,
    )


def main() -> None:
    """Run a reproducible parametric-risk demonstration."""
    random_generator = np.random.default_rng(seed=42)

    daily_returns = random_generator.normal(
        loc=0.0004,
        scale=0.012,
        size=1_000,
    )

    portfolio_value = 1_000_000.0
    confidence_level = 0.95

    one_day_metrics = calculate_parametric_risk_metrics(
        returns=daily_returns,
        portfolio_value=portfolio_value,
        confidence_level=confidence_level,
        time_horizon_days=1,
    )

    ten_day_metrics = calculate_parametric_risk_metrics(
        returns=daily_returns,
        portfolio_value=portfolio_value,
        confidence_level=confidence_level,
        time_horizon_days=10,
    )

    print("Normal parametric risk model")
    print(f"Observations: {len(daily_returns):,}")
    print(
        f"Estimated daily mean: "
        f"{one_day_metrics.mean_return:.4%}"
    )
    print(
        f"Estimated daily volatility: "
        f"{one_day_metrics.volatility:.4%}"
    )

    print("\nOne-day risk")
    print(
        f"Parametric VaR: "
        f"${one_day_metrics.value_at_risk:,.2f}"
    )
    print(
        f"Parametric Expected Shortfall: "
        f"${one_day_metrics.expected_shortfall:,.2f}"
    )

    print("\nTen-day risk")
    print(
        f"Parametric VaR: "
        f"${ten_day_metrics.value_at_risk:,.2f}"
    )
    print(
        f"Parametric Expected Shortfall: "
        f"${ten_day_metrics.expected_shortfall:,.2f}"
    )


if __name__ == "__main__":
    main()