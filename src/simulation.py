"""Monte Carlo simulation for financial risk measurement."""

from dataclasses import dataclass
from numbers import Real

import numpy as np
from numpy.typing import NDArray

from src.risk_metrics import (
    historical_expected_shortfall,
    historical_value_at_risk,
)


FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class MonteCarloRiskResult:
    """Risk measures and simulated portfolio outcomes."""

    confidence_level: float
    time_horizon_days: int
    simulations: int
    mean_return: float
    volatility: float
    value_at_risk: float
    expected_shortfall: float
    simulated_returns: FloatArray
    simulated_losses: FloatArray


def _validate_returns(
    returns: FloatArray,
) -> FloatArray:
    """Validate the historical return series."""
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

    volatility = float(
        np.std(normalized_returns, ddof=1)
    )

    if volatility <= 0.0:
        raise ValueError(
            "Return volatility must be greater than zero."
        )

    return normalized_returns


def _validate_portfolio_value(
    portfolio_value: float,
) -> float:
    """Validate the current portfolio value."""
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


def _validate_confidence_level(
    confidence_level: float,
) -> float:
    """Validate the selected confidence level."""
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


def _validate_time_horizon(
    time_horizon_days: int,
) -> int:
    """Validate the simulation horizon."""
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


def _validate_simulations(
    simulations: int,
) -> int:
    """Validate the number of Monte Carlo simulations."""
    if (
        isinstance(simulations, bool)
        or not isinstance(
            simulations,
            (int, np.integer),
        )
    ):
        raise TypeError(
            "Simulations must be an integer."
        )

    normalized_simulations = int(simulations)

    if normalized_simulations < 2:
        raise ValueError(
            "Simulations must be at least two."
        )

    return normalized_simulations


def _validate_seed(
    seed: int | None,
) -> int | None:
    """Validate the random-number-generator seed."""
    if seed is not None and (
        isinstance(seed, bool)
        or not isinstance(seed, (int, np.integer))
    ):
        raise TypeError(
            "Seed must be an integer or None."
        )

    if seed is None:
        return None

    return int(seed)


def generate_antithetic_shocks(
    simulations: int,
    seed: int | None = None,
) -> FloatArray:
    """Generate standard-normal antithetic random shocks."""
    normalized_simulations = _validate_simulations(
        simulations
    )

    normalized_seed = _validate_seed(seed)

    random_generator = np.random.default_rng(
        normalized_seed
    )

    required_shocks = (
        normalized_simulations + 1
    ) // 2

    positive_shocks = random_generator.standard_normal(
        required_shocks
    )

    antithetic_shocks = np.concatenate(
        [
            positive_shocks,
            -positive_shocks,
        ]
    )

    return antithetic_shocks[
        :normalized_simulations
    ].astype(float)


def simulate_portfolio_returns(
    returns: FloatArray,
    simulations: int = 100_000,
    time_horizon_days: int = 1,
    seed: int | None = None,
) -> FloatArray:
    """Simulate normally distributed portfolio returns."""
    normalized_returns = _validate_returns(returns)

    normalized_simulations = _validate_simulations(
        simulations
    )

    normalized_horizon = _validate_time_horizon(
        time_horizon_days
    )

    normalized_seed = _validate_seed(seed)

    mean_return = float(
        np.mean(normalized_returns)
    )

    volatility = float(
        np.std(normalized_returns, ddof=1)
    )

    horizon_mean = (
        mean_return * normalized_horizon
    )

    horizon_volatility = (
        volatility * np.sqrt(normalized_horizon)
    )

    shocks = generate_antithetic_shocks(
        simulations=normalized_simulations,
        seed=normalized_seed,
    )

    simulated_returns = (
        horizon_mean
        + horizon_volatility * shocks
    )

    return simulated_returns.astype(float)


def calculate_monte_carlo_risk(
    returns: FloatArray,
    portfolio_value: float = 1.0,
    confidence_level: float = 0.95,
    time_horizon_days: int = 1,
    simulations: int = 100_000,
    seed: int | None = None,
) -> MonteCarloRiskResult:
    """Calculate VaR and Expected Shortfall using Monte Carlo."""
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

    normalized_simulations = _validate_simulations(
        simulations
    )

    normalized_seed = _validate_seed(seed)

    mean_return = float(
        np.mean(normalized_returns)
    )

    volatility = float(
        np.std(normalized_returns, ddof=1)
    )

    simulated_returns = simulate_portfolio_returns(
        returns=normalized_returns,
        simulations=normalized_simulations,
        time_horizon_days=normalized_horizon,
        seed=normalized_seed,
    )

    simulated_losses = (
        -simulated_returns * normalized_value
    )

    value_at_risk = historical_value_at_risk(
        losses=simulated_losses,
        confidence_level=normalized_confidence,
    )

    expected_shortfall = historical_expected_shortfall(
        losses=simulated_losses,
        confidence_level=normalized_confidence,
    )

    return MonteCarloRiskResult(
        confidence_level=normalized_confidence,
        time_horizon_days=normalized_horizon,
        simulations=normalized_simulations,
        mean_return=mean_return,
        volatility=volatility,
        value_at_risk=value_at_risk,
        expected_shortfall=expected_shortfall,
        simulated_returns=simulated_returns,
        simulated_losses=simulated_losses,
    )


def main() -> None:
    """Run a reproducible Monte Carlo risk demonstration."""
    random_generator = np.random.default_rng(seed=42)

    historical_returns = random_generator.normal(
        loc=0.0004,
        scale=0.012,
        size=1_000,
    )

    result = calculate_monte_carlo_risk(
        returns=historical_returns,
        portfolio_value=1_000_000.0,
        confidence_level=0.95,
        time_horizon_days=1,
        simulations=500_000,
        seed=42,
    )

    print("Monte Carlo portfolio risk")
    print(
        f"Historical observations: "
        f"{len(historical_returns):,}"
    )
    print(
        f"Number of simulations: "
        f"{result.simulations:,}"
    )
    print(
        f"Portfolio value: "
        f"${1_000_000.0:,.2f}"
    )
    print(
        f"Confidence level: "
        f"{result.confidence_level:.2%}"
    )
    print(
        f"Time horizon: "
        f"{result.time_horizon_days} day"
    )

    print("\nEstimated distribution")
    print(
        f"Daily mean return: "
        f"{result.mean_return:.4%}"
    )
    print(
        f"Daily volatility: "
        f"{result.volatility:.4%}"
    )

    print("\nMonte Carlo risk metrics")
    print(
        f"Value at Risk: "
        f"${result.value_at_risk:,.2f}"
    )
    print(
        f"Expected Shortfall: "
        f"${result.expected_shortfall:,.2f}"
    )


if __name__ == "__main__":
    main()