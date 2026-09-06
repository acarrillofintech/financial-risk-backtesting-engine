"""Tests for the end-to-end market VaR backtest."""

import numpy as np
import pandas as pd
import pytest

import src.market_backtest as market_backtest
from src.data_loader import MarketData
from src.market_backtest import (
    MarketBacktestResult,
    ModelBacktestResult,
    _evaluate_model,
    run_market_backtest,
)


def sample_market_data() -> MarketData:
    """Create deterministic market data without Internet."""
    dates = pd.date_range(
        "2025-01-01",
        periods=11,
        freq="D",
    )

    prices = pd.Series(
        [
            100.0,
            99.0,
            101.0,
            100.5,
            102.0,
            98.0,
            99.5,
            97.0,
            100.0,
            96.0,
            98.0,
        ],
        index=dates,
        name="SPY",
    )

    returns = prices.pct_change().dropna()
    returns.name = "SPY"

    return MarketData(
        ticker="SPY",
        prices=prices,
        returns=returns,
        start_date="2025-01-01",
        end_date="2025-01-12",
    )


def install_fake_market_loader(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Replace the Internet loader with deterministic data."""
    def fake_load_market_data(
        ticker: str,
        start_date: str,
        end_date: str,
    ) -> MarketData:
        assert ticker.upper() == "SPY"
        assert start_date == "2025-01-01"
        assert end_date == "2025-01-12"

        return sample_market_data()

    monkeypatch.setattr(
        market_backtest,
        "load_market_data",
        fake_load_market_data,
    )


def test_evaluate_model_returns_dataclass() -> None:
    """One model evaluation should return its dataclass."""
    result = _evaluate_model(
        model_name="Test VaR",
        actual_losses=np.array(
            [5.0, 20.0, 3.0, 30.0]
        ),
        var_forecasts=np.array(
            [10.0, 10.0, 10.0, 10.0]
        ),
        confidence_level=0.75,
        significance_level=0.05,
    )

    assert isinstance(result, ModelBacktestResult)
    assert result.model_name == "Test VaR"


def test_evaluate_model_var_statistics() -> None:
    """VaR summary statistics should be correct."""
    forecasts = np.array(
        [10.0, 20.0, 30.0, 40.0]
    )

    result = _evaluate_model(
        model_name="Test VaR",
        actual_losses=np.array(
            [5.0, 25.0, 10.0, 50.0]
        ),
        var_forecasts=forecasts,
        confidence_level=0.75,
        significance_level=0.05,
    )

    assert result.average_var == pytest.approx(25.0)
    assert result.minimum_var == pytest.approx(10.0)
    assert result.maximum_var == pytest.approx(40.0)


def test_evaluate_model_counts_exceptions() -> None:
    """Actual losses above VaR should be exceptions."""
    result = _evaluate_model(
        model_name="Test VaR",
        actual_losses=np.array(
            [5.0, 20.0, 3.0, 30.0]
        ),
        var_forecasts=np.full(4, 10.0),
        confidence_level=0.75,
        significance_level=0.05,
    )

    assert result.backtest.actual_exceptions == 2
    assert result.backtest.exception_rate == pytest.approx(
        0.50
    )


def test_evaluate_model_runs_conditional_coverage() -> None:
    """Evaluation should include both statistical tests."""
    result = _evaluate_model(
        model_name="Test VaR",
        actual_losses=np.array(
            [5.0, 20.0, 3.0, 30.0]
        ),
        var_forecasts=np.full(4, 10.0),
        confidence_level=0.75,
        significance_level=0.05,
    )

    combined = result.conditional_coverage

    assert combined.likelihood_ratio == pytest.approx(
        combined.kupiec_test.likelihood_ratio
        + combined.independence_test.likelihood_ratio
    )


def test_run_market_backtest_returns_dataclass(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Complete execution should return its dataclass."""
    install_fake_market_loader(monkeypatch)

    result = run_market_backtest(
        ticker="SPY",
        start_date="2025-01-01",
        end_date="2025-01-12",
        portfolio_value=100_000.0,
        window_size=4,
        confidence_level=0.90,
    )

    assert isinstance(result, MarketBacktestResult)


def test_run_market_backtest_preserves_market_data(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The result should include downloaded market data."""
    install_fake_market_loader(monkeypatch)

    result = run_market_backtest(
        ticker="SPY",
        start_date="2025-01-01",
        end_date="2025-01-12",
        portfolio_value=100_000.0,
        window_size=4,
        confidence_level=0.90,
    )

    assert result.market_data.ticker == "SPY"
    assert len(result.market_data.prices) == 11
    assert len(result.market_data.returns) == 10


def test_forecasts_are_out_of_sample(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Forecast count should exclude the estimation window."""
    install_fake_market_loader(monkeypatch)

    result = run_market_backtest(
        ticker="SPY",
        start_date="2025-01-01",
        end_date="2025-01-12",
        portfolio_value=100_000.0,
        window_size=4,
        confidence_level=0.90,
    )

    expected_forecasts = (
        len(result.market_data.returns) - 4
    )

    assert len(
        result.forecasts.actual_losses
    ) == expected_forecasts

    assert len(
        result.forecasts.historical_var
    ) == expected_forecasts

    assert len(
        result.forecasts.parametric_var
    ) == expected_forecasts


def test_both_var_models_are_evaluated(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Historical and parametric models should be present."""
    install_fake_market_loader(monkeypatch)

    result = run_market_backtest(
        ticker="SPY",
        start_date="2025-01-01",
        end_date="2025-01-12",
        portfolio_value=100_000.0,
        window_size=4,
        confidence_level=0.90,
    )

    assert (
        result.historical_model.model_name
        == "Historical rolling VaR"
    )
    assert (
        result.parametric_model.model_name
        == "Parametric rolling VaR"
    )


def test_model_observations_match_forecasts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Backtest observations should align with forecasts."""
    install_fake_market_loader(monkeypatch)

    result = run_market_backtest(
        ticker="SPY",
        start_date="2025-01-01",
        end_date="2025-01-12",
        portfolio_value=100_000.0,
        window_size=4,
        confidence_level=0.90,
    )

    forecast_count = len(
        result.forecasts.actual_losses
    )

    assert (
        result.historical_model.backtest.observations
        == forecast_count
    )
    assert (
        result.parametric_model.backtest.observations
        == forecast_count
    )


def test_portfolio_value_is_preserved(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Forecast result should retain portfolio value."""
    install_fake_market_loader(monkeypatch)

    result = run_market_backtest(
        ticker="SPY",
        start_date="2025-01-01",
        end_date="2025-01-12",
        portfolio_value=250_000.0,
        window_size=4,
        confidence_level=0.90,
    )

    assert (
        result.forecasts.portfolio_value
        == 250_000.0
    )


def test_invalid_window_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Invalid rolling windows should propagate errors."""
    install_fake_market_loader(monkeypatch)

    with pytest.raises(
        ValueError,
        match="smaller",
    ):
        run_market_backtest(
            ticker="SPY",
            start_date="2025-01-01",
            end_date="2025-01-12",
            portfolio_value=100_000.0,
            window_size=10,
            confidence_level=0.90,
        )


def test_invalid_portfolio_value_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A negative portfolio value should be rejected."""
    install_fake_market_loader(monkeypatch)

    with pytest.raises(
        ValueError,
        match="finite and positive",
    ):
        run_market_backtest(
            ticker="SPY",
            start_date="2025-01-01",
            end_date="2025-01-12",
            portfolio_value=-100_000.0,
            window_size=4,
            confidence_level=0.90,
        )