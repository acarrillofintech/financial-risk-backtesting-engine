"""Backtesting tools for Value at Risk models."""

from dataclasses import dataclass
from numbers import Real

import numpy as np
from numpy.typing import NDArray
from scipy.special import xlogy
from scipy.stats import chi2, norm


FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]


@dataclass(frozen=True)
class KupiecTestResult:
    """Result of the Kupiec proportion-of-failures test."""

    likelihood_ratio: float
    p_value: float
    significance_level: float
    rejected: bool


@dataclass(frozen=True)
class VaRBacktestResult:
    """Complete result of a Value at Risk backtest."""

    confidence_level: float
    observations: int
    expected_exceptions: float
    actual_exceptions: int
    exception_rate: float
    exceptions: BoolArray
    kupiec_test: KupiecTestResult
    basel_zone: str


def _validate_confidence_level(
    confidence_level: float,
) -> float:
    """Validate a confidence level."""
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


def _validate_significance_level(
    significance_level: float,
) -> float:
    """Validate the statistical significance level."""
    if (
        isinstance(significance_level, bool)
        or not isinstance(significance_level, Real)
    ):
        raise TypeError(
            "Significance level must be a real number."
        )

    normalized_significance = float(
        significance_level
    )

    if not 0.0 < normalized_significance < 1.0:
        raise ValueError(
            "Significance level must be between 0 and 1."
        )

    return normalized_significance


def _validate_losses(
    actual_losses: FloatArray,
) -> FloatArray:
    """Validate realized portfolio losses."""
    normalized_losses = np.asarray(
        actual_losses,
        dtype=float,
    )

    if normalized_losses.ndim != 1:
        raise ValueError(
            "Actual losses must be a one-dimensional array."
        )

    if normalized_losses.size == 0:
        raise ValueError(
            "Actual losses cannot be empty."
        )

    if not np.all(np.isfinite(normalized_losses)):
        raise ValueError(
            "Actual losses must contain only finite values."
        )

    return normalized_losses


def _normalize_var_forecasts(
    var_forecasts: FloatArray | float,
    observations: int,
) -> FloatArray:
    """Convert scalar or array VaR forecasts into one array."""
    normalized_forecasts = np.asarray(
        var_forecasts,
        dtype=float,
    )

    if normalized_forecasts.ndim == 0:
        normalized_forecasts = np.full(
            observations,
            float(normalized_forecasts),
        )

    if normalized_forecasts.ndim != 1:
        raise ValueError(
            "VaR forecasts must be a scalar "
            "or one-dimensional array."
        )

    if len(normalized_forecasts) != observations:
        raise ValueError(
            "VaR forecasts and actual losses "
            "must have the same length."
        )

    if not np.all(np.isfinite(normalized_forecasts)):
        raise ValueError(
            "VaR forecasts must contain only finite values."
        )

    if np.any(normalized_forecasts < 0.0):
        raise ValueError(
            "VaR forecasts cannot be negative."
        )

    return normalized_forecasts.astype(float)


def identify_var_exceptions(
    actual_losses: FloatArray,
    var_forecasts: FloatArray | float,
) -> BoolArray:
    """Identify days when realized loss exceeded forecast VaR."""
    normalized_losses = _validate_losses(actual_losses)

    normalized_forecasts = _normalize_var_forecasts(
        var_forecasts=var_forecasts,
        observations=len(normalized_losses),
    )

    return (
        normalized_losses > normalized_forecasts
    ).astype(bool)


def _validate_exceptions(
    exceptions: BoolArray,
) -> BoolArray:
    """Validate a sequence of VaR exception indicators."""
    normalized_exceptions = np.asarray(exceptions)

    if normalized_exceptions.ndim != 1:
        raise ValueError(
            "Exceptions must be a one-dimensional array."
        )

    if normalized_exceptions.size == 0:
        raise ValueError(
            "Exceptions cannot be empty."
        )

    if not np.all(
        np.isin(normalized_exceptions, [False, True, 0, 1])
    ):
        raise ValueError(
            "Exceptions must contain only boolean values."
        )

    return normalized_exceptions.astype(bool)


def kupiec_proportion_of_failures_test(
    exceptions: BoolArray,
    confidence_level: float = 0.99,
    significance_level: float = 0.05,
) -> KupiecTestResult:
    """Perform the Kupiec unconditional-coverage test."""
    normalized_exceptions = _validate_exceptions(
        exceptions
    )

    normalized_confidence = _validate_confidence_level(
        confidence_level
    )

    normalized_significance = (
        _validate_significance_level(
            significance_level
        )
    )

    observations = len(normalized_exceptions)
    actual_exceptions = int(
        np.sum(normalized_exceptions)
    )

    expected_probability = (
        1.0 - normalized_confidence
    )

    observed_probability = (
        actual_exceptions / observations
    )

    restricted_log_likelihood = (
        xlogy(
            observations - actual_exceptions,
            1.0 - expected_probability,
        )
        + xlogy(
            actual_exceptions,
            expected_probability,
        )
    )

    unrestricted_log_likelihood = (
        xlogy(
            observations - actual_exceptions,
            1.0 - observed_probability,
        )
        + xlogy(
            actual_exceptions,
            observed_probability,
        )
    )

    likelihood_ratio = float(
        -2.0
        * (
            restricted_log_likelihood
            - unrestricted_log_likelihood
        )
    )

    likelihood_ratio = max(
        likelihood_ratio,
        0.0,
    )

    p_value = float(
        chi2.sf(likelihood_ratio, df=1)
    )

    return KupiecTestResult(
        likelihood_ratio=likelihood_ratio,
        p_value=p_value,
        significance_level=normalized_significance,
        rejected=bool(
            p_value < normalized_significance
        ),
    )


def basel_traffic_light_zone(
    exceptions: BoolArray,
    confidence_level: float = 0.99,
) -> str:
    """Classify a regulatory 250-day, 99% VaR backtest."""
    normalized_exceptions = _validate_exceptions(
        exceptions
    )

    normalized_confidence = _validate_confidence_level(
        confidence_level
    )

    if (
        len(normalized_exceptions) != 250
        or not np.isclose(
            normalized_confidence,
            0.99,
        )
    ):
        return "not_applicable"

    actual_exceptions = int(
        np.sum(normalized_exceptions)
    )

    if actual_exceptions <= 4:
        return "green"

    if actual_exceptions <= 9:
        return "yellow"

    return "red"


def backtest_value_at_risk(
    actual_losses: FloatArray,
    var_forecasts: FloatArray | float,
    confidence_level: float = 0.99,
    significance_level: float = 0.05,
) -> VaRBacktestResult:
    """Run exception analysis, Kupiec test, and Basel zone."""
    normalized_losses = _validate_losses(
        actual_losses
    )

    normalized_confidence = _validate_confidence_level(
        confidence_level
    )

    normalized_significance = (
        _validate_significance_level(
            significance_level
        )
    )

    exceptions = identify_var_exceptions(
        actual_losses=normalized_losses,
        var_forecasts=var_forecasts,
    )

    observations = len(exceptions)
    actual_exceptions = int(np.sum(exceptions))

    expected_exceptions = (
        observations
        * (1.0 - normalized_confidence)
    )

    exception_rate = (
        actual_exceptions / observations
    )

    kupiec_result = (
        kupiec_proportion_of_failures_test(
            exceptions=exceptions,
            confidence_level=normalized_confidence,
            significance_level=normalized_significance,
        )
    )

    basel_zone = basel_traffic_light_zone(
        exceptions=exceptions,
        confidence_level=normalized_confidence,
    )

    return VaRBacktestResult(
        confidence_level=normalized_confidence,
        observations=observations,
        expected_exceptions=expected_exceptions,
        actual_exceptions=actual_exceptions,
        exception_rate=exception_rate,
        exceptions=exceptions,
        kupiec_test=kupiec_result,
        basel_zone=basel_zone,
    )


def main() -> None:
    """Run a reproducible 250-day VaR backtest."""
    random_generator = np.random.default_rng(seed=42)

    observations = 250
    portfolio_value = 1_000_000.0
    daily_volatility = 0.012
    confidence_level = 0.99

    simulated_returns = random_generator.normal(
        loc=0.0004,
        scale=daily_volatility,
        size=observations,
    )

    actual_losses = (
        -simulated_returns * portfolio_value
    )

    constant_var_forecast = (
        portfolio_value
        * daily_volatility
        * norm.ppf(confidence_level)
    )

    result = backtest_value_at_risk(
        actual_losses=actual_losses,
        var_forecasts=constant_var_forecast,
        confidence_level=confidence_level,
        significance_level=0.05,
    )

    print("Value at Risk backtesting")
    print(f"Observations: {result.observations}")
    print(
        f"Expected exceptions: "
        f"{result.expected_exceptions:.2f}"
    )
    print(
        f"Actual exceptions: "
        f"{result.actual_exceptions}"
    )
    print(
        f"Exception rate: "
        f"{result.exception_rate:.2%}"
    )

    print("\nKupiec test")
    print(
        f"Likelihood-ratio statistic: "
        f"{result.kupiec_test.likelihood_ratio:.6f}"
    )
    print(
        f"P-value: "
        f"{result.kupiec_test.p_value:.6f}"
    )
    print(
        f"Model rejected: "
        f"{result.kupiec_test.rejected}"
    )

    print("\nBasel traffic-light classification")
    print(f"Zone: {result.basel_zone}")


if __name__ == "__main__":
    main()