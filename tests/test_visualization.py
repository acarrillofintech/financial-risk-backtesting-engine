"""Tests for risk backtesting visualizations."""

from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import pytest

import src.market_backtest as market_backtest
from src.data_loader import MarketData
from src.market_backtest import run_market_backtest
from src.visualization import (
    VisualizationPaths,
    create_all_visualizations,
    plot_var_backtest,
    plot_var_model_comparison,
)


matplotlib.use("Agg")


def sample_plot_data() -> tuple[
    pd.DatetimeIndex,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """Create deterministic plotting data."""
    dates = pd.date_range(
        "2025-01-01",
        periods=5,
        freq="D",
    )

    losses = np.array(
        [5.0, 12.0, -3.0, 20.0, 7.0]
    )

    forecasts = np.array(
        [10.0, 10.0, 11.0, 15.0, 12.0]
    )

    exceptions = losses > forecasts

    return dates, losses, forecasts, exceptions


def sample_market_data() -> MarketData:
    """Create market data without downloading it."""
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


def create_market_result(
    monkeypatch: pytest.MonkeyPatch,
):
    """Create a complete backtest without Internet."""
    def fake_loader(
        ticker: str,
        start_date: str,
        end_date: str,
    ) -> MarketData:
        return sample_market_data()

    monkeypatch.setattr(
        market_backtest,
        "load_market_data",
        fake_loader,
    )

    return run_market_backtest(
        ticker="SPY",
        start_date="2025-01-01",
        end_date="2025-01-12",
        portfolio_value=100_000.0,
        window_size=4,
        confidence_level=0.90,
    )


def test_plot_var_backtest_creates_file(
    tmp_path: Path,
) -> None:
    """A backtesting chart should be saved."""
    dates, losses, forecasts, exceptions = (
        sample_plot_data()
    )

    output_path = tmp_path / "backtest.png"

    result = plot_var_backtest(
        dates=dates,
        actual_losses=losses,
        var_forecasts=forecasts,
        exceptions=exceptions,
        model_name="Test VaR",
        output_path=output_path,
    )

    assert result == output_path
    assert output_path.exists()
    assert output_path.stat().st_size > 0


def test_plot_comparison_creates_file(
    tmp_path: Path,
) -> None:
    """The comparison chart should be saved."""
    dates, losses, historical, _ = (
        sample_plot_data()
    )

    parametric = historical + 2.0
    output_path = tmp_path / "comparison.png"

    result = plot_var_model_comparison(
        dates=dates,
        actual_losses=losses,
        historical_var=historical,
        parametric_var=parametric,
        output_path=output_path,
    )

    assert result == output_path
    assert output_path.exists()
    assert output_path.stat().st_size > 0


def test_plot_creates_parent_directories(
    tmp_path: Path,
) -> None:
    """Missing output directories should be created."""
    dates, losses, forecasts, exceptions = (
        sample_plot_data()
    )

    output_path = (
        tmp_path
        / "nested"
        / "figures"
        / "chart.png"
    )

    plot_var_backtest(
        dates=dates,
        actual_losses=losses,
        var_forecasts=forecasts,
        exceptions=exceptions,
        model_name="Test VaR",
        output_path=output_path,
    )

    assert output_path.exists()


def test_create_all_visualizations(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The complete function should create three figures."""
    result = create_market_result(monkeypatch)

    paths = create_all_visualizations(
        result=result,
        output_directory=tmp_path,
    )

    assert isinstance(paths, VisualizationPaths)
    assert paths.historical_backtest.exists()
    assert paths.parametric_backtest.exists()
    assert paths.model_comparison.exists()


def test_visualization_filenames(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Generated figures should use stable filenames."""
    result = create_market_result(monkeypatch)

    paths = create_all_visualizations(
        result=result,
        output_directory=tmp_path,
    )

    assert (
        paths.historical_backtest.name
        == "historical_var_backtest.png"
    )
    assert (
        paths.parametric_backtest.name
        == "parametric_var_backtest.png"
    )
    assert (
        paths.model_comparison.name
        == "var_model_comparison.png"
    )


def test_empty_plot_data_raises_value_error(
    tmp_path: Path,
) -> None:
    """Empty plotting arrays should be rejected."""
    with pytest.raises(
        ValueError,
        match="cannot be empty",
    ):
        plot_var_backtest(
            dates=pd.DatetimeIndex([]),
            actual_losses=np.array([]),
            var_forecasts=np.array([]),
            exceptions=np.array([], dtype=bool),
            model_name="Test VaR",
            output_path=tmp_path / "empty.png",
        )


def test_mismatched_dates_raise_value_error(
    tmp_path: Path,
) -> None:
    """Dates and losses must have equal lengths."""
    _, losses, forecasts, exceptions = (
        sample_plot_data()
    )

    short_dates = pd.date_range(
        "2025-01-01",
        periods=4,
    )

    with pytest.raises(
        ValueError,
        match="same length",
    ):
        plot_var_backtest(
            dates=short_dates,
            actual_losses=losses,
            var_forecasts=forecasts,
            exceptions=exceptions,
            model_name="Test VaR",
            output_path=tmp_path / "invalid.png",
        )


def test_mismatched_exceptions_raise_value_error(
    tmp_path: Path,
) -> None:
    """Exception indicators must align with losses."""
    dates, losses, forecasts, _ = (
        sample_plot_data()
    )

    with pytest.raises(
        ValueError,
        match="same length",
    ):
        plot_var_backtest(
            dates=dates,
            actual_losses=losses,
            var_forecasts=forecasts,
            exceptions=np.array(
                [False, True],
                dtype=bool,
            ),
            model_name="Test VaR",
            output_path=tmp_path / "invalid.png",
        )


def test_mismatched_parametric_var_raises_value_error(
    tmp_path: Path,
) -> None:
    """Both VaR series must align with losses."""
    dates, losses, historical, _ = (
        sample_plot_data()
    )

    with pytest.raises(
        ValueError,
        match="same length",
    ):
        plot_var_model_comparison(
            dates=dates,
            actual_losses=losses,
            historical_var=historical,
            parametric_var=np.array([10.0, 11.0]),
            output_path=tmp_path / "invalid.png",
        )


def test_non_finite_losses_raise_value_error(
    tmp_path: Path,
) -> None:
    """Losses cannot contain NaN values."""
    dates, losses, forecasts, exceptions = (
        sample_plot_data()
    )

    losses[2] = np.nan

    with pytest.raises(
        ValueError,
        match="must be finite",
    ):
        plot_var_backtest(
            dates=dates,
            actual_losses=losses,
            var_forecasts=forecasts,
            exceptions=exceptions,
            model_name="Test VaR",
            output_path=tmp_path / "invalid.png",
        )


def test_non_finite_forecasts_raise_value_error(
    tmp_path: Path,
) -> None:
    """VaR forecasts cannot contain infinity."""
    dates, losses, forecasts, exceptions = (
        sample_plot_data()
    )

    forecasts[1] = np.inf

    with pytest.raises(
        ValueError,
        match="must be finite",
    ):
        plot_var_backtest(
            dates=dates,
            actual_losses=losses,
            var_forecasts=forecasts,
            exceptions=exceptions,
            model_name="Test VaR",
            output_path=tmp_path / "invalid.png",
        )