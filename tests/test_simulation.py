"""Tests for Monte Carlo financial risk simulation."""

import numpy as np
import pytest

from src.simulation import (
    MonteCarloRiskResult,
    calculate_monte_carlo_risk,
    generate_antithetic_shocks,
    simulate_portfolio_returns,
)
from src.var_models import (
    parametric_expected_shortfall,
    parametric_value_at_risk,
)


@pytest.fixture
def historical_returns() -> np.ndarray:
    """Provide returns with known distribution parameters."""
    return np.array(
        [-0.02, -0.01, 0.00, 0.01, 0.02]
    )


def test_antithetic_shocks_have_expected_shape() -> None:
    """Shock array must match the requested simulation count."""
    shocks = generate_antithetic_shocks(
        simulations=1_000,
        seed=42,
    )

    assert shocks.shape == (1_000,)


def test_antithetic_shocks_are_opposites() -> None:
    """The second half should oppose the first half."""
    shocks = generate_antithetic_shocks(
        simulations=1_000,
        seed=42,
    )

    np.testing.assert_allclose(
        shocks[:500],
        -shocks[500:],
    )


def test_antithetic_shocks_are_reproducible() -> None:
    """The same seed must generate identical shocks."""
    first = generate_antithetic_shocks(
        simulations=1_000,
        seed=42,
    )

    second = generate_antithetic_shocks(
        simulations=1_000,
        seed=42,
    )

    np.testing.assert_array_equal(first, second)


def test_different_seeds_generate_different_shocks() -> None:
    """Different seeds should produce different scenarios."""
    first = generate_antithetic_shocks(
        simulations=100,
        seed=1,
    )

    second = generate_antithetic_shocks(
        simulations=100,
        seed=2,
    )

    assert not np.array_equal(first, second)


def test_simulated_returns_have_expected_shape(
    historical_returns: np.ndarray,
) -> None:
    """Simulated-return array must have the requested size."""
    result = simulate_portfolio_returns(
        returns=historical_returns,
        simulations=2_000,
        seed=42,
    )

    assert result.shape == (2_000,)


def test_simulated_returns_are_reproducible(
    historical_returns: np.ndarray,
) -> None:
    """The same seed must reproduce simulated returns."""
    first = simulate_portfolio_returns(
        returns=historical_returns,
        simulations=2_000,
        seed=42,
    )

    second = simulate_portfolio_returns(
        returns=historical_returns,
        simulations=2_000,
        seed=42,
    )

    np.testing.assert_array_equal(first, second)


def test_simulated_distribution_matches_horizon(
    historical_returns: np.ndarray,
) -> None:
    """Simulation should follow scaled mean and volatility."""
    result = simulate_portfolio_returns(
        returns=historical_returns,
        simulations=50_000,
        time_horizon_days=10,
        seed=42,
    )

    expected_mean = (
        np.mean(historical_returns) * 10
    )

    expected_volatility = (
        np.std(historical_returns, ddof=1)
        * np.sqrt(10)
    )

    assert np.mean(result) == pytest.approx(
        expected_mean,
        abs=1e-10,
    )

    assert np.std(result, ddof=1) == pytest.approx(
        expected_volatility,
        rel=0.02,
    )


def test_result_is_monte_carlo_dataclass(
    historical_returns: np.ndarray,
) -> None:
    """Risk calculation should return structured output."""
    result = calculate_monte_carlo_risk(
        returns=historical_returns,
        portfolio_value=1_000_000.0,
        simulations=10_000,
        seed=42,
    )

    assert isinstance(result, MonteCarloRiskResult)


def test_result_arrays_have_expected_dimensions(
    historical_returns: np.ndarray,
) -> None:
    """Result arrays must contain every simulation."""
    result = calculate_monte_carlo_risk(
        returns=historical_returns,
        portfolio_value=1_000_000.0,
        simulations=10_000,
        seed=42,
    )

    assert result.simulated_returns.shape == (10_000,)
    assert result.simulated_losses.shape == (10_000,)
    assert result.simulations == 10_000


def test_losses_are_negative_returns_times_value(
    historical_returns: np.ndarray,
) -> None:
    """Monetary losses must follow the loss convention."""
    result = calculate_monte_carlo_risk(
        returns=historical_returns,
        portfolio_value=1_000_000.0,
        simulations=10_000,
        seed=42,
    )

    expected_losses = (
        -result.simulated_returns * 1_000_000.0
    )

    np.testing.assert_allclose(
        result.simulated_losses,
        expected_losses,
    )


def test_expected_shortfall_exceeds_var(
    historical_returns: np.ndarray,
) -> None:
    """Average tail loss should exceed the VaR threshold."""
    result = calculate_monte_carlo_risk(
        returns=historical_returns,
        portfolio_value=1_000_000.0,
        confidence_level=0.95,
        simulations=100_000,
        seed=42,
    )

    assert result.expected_shortfall > result.value_at_risk


def test_monte_carlo_var_is_close_to_parametric_var(
    historical_returns: np.ndarray,
) -> None:
    """Large normal simulation should approach parametric VaR."""
    monte_carlo = calculate_monte_carlo_risk(
        returns=historical_returns,
        portfolio_value=1_000_000.0,
        confidence_level=0.95,
        simulations=100_000,
        seed=42,
    )

    parametric = parametric_value_at_risk(
        returns=historical_returns,
        portfolio_value=1_000_000.0,
        confidence_level=0.95,
    )

    assert monte_carlo.value_at_risk == pytest.approx(
        parametric,
        rel=0.01,
    )


def test_monte_carlo_es_is_close_to_parametric_es(
    historical_returns: np.ndarray,
) -> None:
    """Large normal simulation should approach parametric ES."""
    monte_carlo = calculate_monte_carlo_risk(
        returns=historical_returns,
        portfolio_value=1_000_000.0,
        confidence_level=0.95,
        simulations=100_000,
        seed=42,
    )

    parametric = parametric_expected_shortfall(
        returns=historical_returns,
        portfolio_value=1_000_000.0,
        confidence_level=0.95,
    )

    assert monte_carlo.expected_shortfall == pytest.approx(
        parametric,
        rel=0.01,
    )


@pytest.mark.parametrize(
    "simulations",
    [0, 1, -100],
)
def test_invalid_simulation_count_raises_value_error(
    simulations: int,
) -> None:
    """At least two simulations are required."""
    with pytest.raises(
        ValueError,
        match="Simulations must be at least two",
    ):
        generate_antithetic_shocks(
            simulations=simulations,
        )


def test_non_integer_simulations_raise_type_error() -> None:
    """Simulation count must be an integer."""
    with pytest.raises(
        TypeError,
        match="Simulations must be an integer",
    ):
        generate_antithetic_shocks(
            simulations=100.5,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize(
    "time_horizon_days",
    [0, -1],
)
def test_invalid_time_horizon_raises_value_error(
    historical_returns: np.ndarray,
    time_horizon_days: int,
) -> None:
    """Time horizon must be positive."""
    with pytest.raises(
        ValueError,
        match="Time horizon must be greater than zero",
    ):
        simulate_portfolio_returns(
            returns=historical_returns,
            simulations=100,
            time_horizon_days=time_horizon_days,
        )


def test_invalid_seed_raises_type_error() -> None:
    """Random seed must be an integer or None."""
    with pytest.raises(
        TypeError,
        match="Seed must be an integer or None",
    ):
        generate_antithetic_shocks(
            simulations=100,
            seed=42.5,  # type: ignore[arg-type]
        )


def test_invalid_confidence_level_raises_value_error(
    historical_returns: np.ndarray,
) -> None:
    """Confidence level must be below one."""
    with pytest.raises(
        ValueError,
        match="Confidence level must be between 0 and 1",
    ):
        calculate_monte_carlo_risk(
            returns=historical_returns,
            confidence_level=1.0,
            simulations=100,
        )


def test_constant_returns_raise_value_error() -> None:
    """Monte Carlo risk requires positive volatility."""
    constant_returns = np.array(
        [0.01, 0.01, 0.01, 0.01]
    )

    with pytest.raises(
        ValueError,
        match="Return volatility must be greater than zero",
    ):
        simulate_portfolio_returns(
            returns=constant_returns,
            simulations=100,
        )