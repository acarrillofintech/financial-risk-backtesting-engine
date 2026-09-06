"""Rolling Value at Risk forecasts."""

from dataclasses import dataclass
from numbers import Integral, Real

import numpy as np
from numpy.typing import NDArray
from scipy.stats import norm

from src.backtesting import (
    VaRBacktestResult,
    backtest_value_at_risk,
)


FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class RollingVaRForecasts:
    """Rolling historical and parametric VaR forecasts."""

    actual_losses: FloatArray
    historical_var: FloatArray
    parametric_var: FloatArray
    window_size: int
    confidence_level: float
    portfolio_value: float


def _validate_returns(
    returns: FloatArray,
) -> FloatArray:
    """Validate a one-dimensional return series."""
    normalized_returns = np.asarray(
        returns,
        dtype=float,
    )

    if normalized_returns.ndim != 1:
        raise ValueError(
            "Returns must be a one-dimensional array."
        )

    if normalized_returns.size < 3:
        raise ValueError(
            "At least three return observations are required."
        )

    if not np.all(np.isfinite(normalized_returns)):
        raise ValueError(
            "Returns must contain only finite values."
        )

    return normalized_returns


def _validate_window_size(
    window_size: int,
    observations: int,
) -> int:
    """Validate the rolling estimation window."""
    if (
        isinstance(window_size, bool)
        or not isinstance(window_size, Integral)
    ):
        raise TypeError(
            "Window size must be an integer."
        )

    normalized_window = int(window_size)

    if normalized_window < 2:
        raise ValueError(
            "Window size must be at least 2."
        )

    if normalized_window >= observations:
        raise ValueError(
            "Window size must be smaller than "
            "the number of return observations."
        )

    return normalized_window


def _validate_confidence_level(
    confidence_level: float,
) -> float:
    """Validate a VaR confidence level."""
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
            "Portfolio value must be finite and positive."
        )

    return normalized_value


def rolling_historical_var(
    returns: FloatArray,
    portfolio_value: float,
    window_size: int = 250,
    confidence_level: float = 0.99,
) -> FloatArray:
    """Calculate historical VaR using prior rolling windows."""
    normalized_returns = _validate_returns(returns)

    normalized_window = _validate_window_size(
        window_size=window_size,
        observations=len(normalized_returns),
    )

    normalized_confidence = (
        _validate_confidence_level(confidence_level)
    )

    normalized_value = _validate_portfolio_value(
        portfolio_value
    )

    forecast_count = (
        len(normalized_returns) - normalized_window
    )

    forecasts = np.empty(
        forecast_count,
        dtype=float,
    )

    for forecast_index in range(forecast_count):
        window_start = forecast_index
        window_end = (
            forecast_index + normalized_window
        )

        estimation_returns = normalized_returns[
            window_start:window_end
        ]

        estimation_losses = (
            -estimation_returns * normalized_value
        )

        forecast = np.quantile(
            estimation_losses,
            normalized_confidence,
        )

        forecasts[forecast_index] = max(
            float(forecast),
            0.0,
        )

    return forecasts


def rolling_parametric_var(
    returns: FloatArray,
    portfolio_value: float,
    window_size: int = 250,
    confidence_level: float = 0.99,
) -> FloatArray:
    """Calculate normal parametric VaR on rolling windows."""
    normalized_returns = _validate_returns(returns)

    normalized_window = _validate_window_size(
        window_size=window_size,
        observations=len(normalized_returns),
    )

    normalized_confidence = (
        _validate_confidence_level(confidence_level)
    )

    normalized_value = _validate_portfolio_value(
        portfolio_value
    )

    forecast_count = (
        len(normalized_returns) - normalized_window
    )

    forecasts = np.empty(
        forecast_count,
        dtype=float,
    )

    normal_quantile = float(
        norm.ppf(normalized_confidence)
    )

    for forecast_index in range(forecast_count):
        window_start = forecast_index
        window_end = (
            forecast_index + normalized_window
        )

        estimation_returns = normalized_returns[
            window_start:window_end
        ]

        estimated_mean = float(
            np.mean(estimation_returns)
        )

        estimated_volatility = float(
            np.std(
                estimation_returns,
                ddof=1,
            )
        )

        forecast = normalized_value * (
            normal_quantile * estimated_volatility
            - estimated_mean
        )

        forecasts[forecast_index] = max(
            float(forecast),
            0.0,
        )

    return forecasts


def calculate_rolling_var_forecasts(
    returns: FloatArray,
    portfolio_value: float,
    window_size: int = 250,
    confidence_level: float = 0.99,
) -> RollingVaRForecasts:
    """Calculate aligned losses and rolling VaR forecasts."""
    normalized_returns = _validate_returns(returns)

    normalized_window = _validate_window_size(
        window_size=window_size,
        observations=len(normalized_returns),
    )

    normalized_confidence = (
        _validate_confidence_level(confidence_level)
    )

    normalized_value = _validate_portfolio_value(
        portfolio_value
    )

    historical_forecasts = rolling_historical_var(
        returns=normalized_returns,
        portfolio_value=normalized_value,
        window_size=normalized_window,
        confidence_level=normalized_confidence,
    )

    parametric_forecasts = rolling_parametric_var(
        returns=normalized_returns,
        portfolio_value=normalized_value,
        window_size=normalized_window,
        confidence_level=normalized_confidence,
    )

    actual_losses = (
        -normalized_returns[normalized_window:]
        * normalized_value
    )

    return RollingVaRForecasts(
        actual_losses=actual_losses,
        historical_var=historical_forecasts,
        parametric_var=parametric_forecasts,
        window_size=normalized_window,
        confidence_level=normalized_confidence,
        portfolio_value=normalized_value,
    )


def _print_backtest(
    model_name: str,
    result: VaRBacktestResult,
) -> None:
    """Print a concise rolling VaR backtest result."""
    print(f"\n{model_name}")
    print(
        f"Average VaR: "
        f"${np.mean(result.exceptions * 0):,.2f}"
        if False
        else f"Observations: {result.observations}"
    )
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
    print(
        f"Kupiec p-value: "
        f"{result.kupiec_test.p_value:.6f}"
    )
    print(
        f"Kupiec rejected: "
        f"{result.kupiec_test.rejected}"
    )


def main() -> None:
    """Run a reproducible rolling VaR demonstration."""
    random_generator = np.random.default_rng(seed=42)

    calm_returns = random_generator.normal(
        loc=0.0004,
        scale=0.009,
        size=500,
    )

    volatile_returns = random_generator.normal(
        loc=-0.0002,
        scale=0.018,
        size=250,
    )

    returns = np.concatenate(
        [calm_returns, volatile_returns]
    )

    portfolio_value = 1_000_000.0
    window_size = 250
    confidence_level = 0.99

    forecasts = calculate_rolling_var_forecasts(
        returns=returns,
        portfolio_value=portfolio_value,
        window_size=window_size,
        confidence_level=confidence_level,
    )

    historical_backtest = backtest_value_at_risk(
        actual_losses=forecasts.actual_losses,
        var_forecasts=forecasts.historical_var,
        confidence_level=confidence_level,
    )

    parametric_backtest = backtest_value_at_risk(
        actual_losses=forecasts.actual_losses,
        var_forecasts=forecasts.parametric_var,
        confidence_level=confidence_level,
    )

    print("Rolling Value at Risk")
    print(f"Total returns: {len(returns)}")
    print(f"Estimation window: {window_size}")
    print(
        f"Out-of-sample forecasts: "
        f"{len(forecasts.actual_losses)}"
    )
    print(
        f"Portfolio value: "
        f"${portfolio_value:,.2f}"
    )

    print("\nAverage forecasts")
    print(
        f"Historical VaR: "
        f"${np.mean(forecasts.historical_var):,.2f}"
    )
    print(
        f"Parametric VaR: "
        f"${np.mean(forecasts.parametric_var):,.2f}"
    )

    _print_backtest(
        "Historical rolling VaR backtest",
        historical_backtest,
    )

    _print_backtest(
        "Parametric rolling VaR backtest",
        parametric_backtest,
    )


if __name__ == "__main__":
    main()