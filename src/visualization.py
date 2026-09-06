"""Visualizations for Value at Risk backtesting."""

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from numpy.typing import NDArray

from src.market_backtest import (
    MarketBacktestResult,
    run_market_backtest,
)


FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]

DEFAULT_FIGURES_DIRECTORY = Path("results/figures")


@dataclass(frozen=True)
class VisualizationPaths:
    """Paths of all generated backtesting figures."""

    historical_backtest: Path
    parametric_backtest: Path
    model_comparison: Path


def _validate_plot_inputs(
    dates: pd.Index,
    actual_losses: FloatArray,
    var_forecasts: FloatArray,
    exceptions: BoolArray | None = None,
) -> tuple[
    pd.DatetimeIndex,
    FloatArray,
    FloatArray,
    BoolArray | None,
]:
    """Validate and normalize plotting inputs."""
    normalized_dates = pd.DatetimeIndex(dates)

    normalized_losses = np.asarray(
        actual_losses,
        dtype=float,
    )

    normalized_forecasts = np.asarray(
        var_forecasts,
        dtype=float,
    )

    if normalized_losses.ndim != 1:
        raise ValueError(
            "Actual losses must be one-dimensional."
        )

    if normalized_forecasts.ndim != 1:
        raise ValueError(
            "VaR forecasts must be one-dimensional."
        )

    observations = len(normalized_losses)

    if observations == 0:
        raise ValueError(
            "Plotting data cannot be empty."
        )

    if (
        len(normalized_dates) != observations
        or len(normalized_forecasts) != observations
    ):
        raise ValueError(
            "Dates, losses, and forecasts "
            "must have the same length."
        )

    if not np.all(np.isfinite(normalized_losses)):
        raise ValueError(
            "Actual losses must be finite."
        )

    if not np.all(np.isfinite(normalized_forecasts)):
        raise ValueError(
            "VaR forecasts must be finite."
        )

    normalized_exceptions = None

    if exceptions is not None:
        normalized_exceptions = np.asarray(
            exceptions,
            dtype=bool,
        )

        if normalized_exceptions.ndim != 1:
            raise ValueError(
                "Exceptions must be one-dimensional."
            )

        if len(normalized_exceptions) != observations:
            raise ValueError(
                "Exceptions and losses "
                "must have the same length."
            )

    return (
        normalized_dates,
        normalized_losses,
        normalized_forecasts,
        normalized_exceptions,
    )


def _prepare_output_path(
    output_path: str | Path,
) -> Path:
    """Create the parent directory for a figure."""
    normalized_path = Path(output_path)

    normalized_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    return normalized_path


def _format_loss_axis(axis: Axes) -> None:
    """Format monetary values on a chart axis."""
    axis.yaxis.set_major_formatter(
        lambda value, position: f"${value / 1_000:,.0f}K"
    )


def plot_var_backtest(
    dates: pd.Index,
    actual_losses: FloatArray,
    var_forecasts: FloatArray,
    exceptions: BoolArray,
    model_name: str,
    output_path: str | Path,
) -> Path:
    """Plot realized losses against rolling VaR forecasts."""
    (
        normalized_dates,
        normalized_losses,
        normalized_forecasts,
        normalized_exceptions,
    ) = _validate_plot_inputs(
        dates=dates,
        actual_losses=actual_losses,
        var_forecasts=var_forecasts,
        exceptions=exceptions,
    )

    if normalized_exceptions is None:
        raise ValueError(
            "Exception indicators are required."
        )

    normalized_output = _prepare_output_path(
        output_path
    )

    figure, axis = plt.subplots(
        figsize=(15, 7)
    )

    axis.plot(
        normalized_dates,
        normalized_losses,
        color="#4C72B0",
        linewidth=0.8,
        alpha=0.75,
        label="Realized daily loss",
    )

    axis.plot(
        normalized_dates,
        normalized_forecasts,
        color="#DD8452",
        linewidth=2.0,
        label=model_name,
    )

    exception_dates = normalized_dates[
        normalized_exceptions
    ]

    exception_losses = normalized_losses[
        normalized_exceptions
    ]

    axis.scatter(
        exception_dates,
        exception_losses,
        color="#C44E52",
        edgecolor="black",
        linewidth=0.5,
        s=48,
        zorder=5,
        label="VaR exceptions",
    )

    axis.axhline(
        y=0.0,
        color="black",
        linewidth=0.8,
        alpha=0.6,
    )

    axis.set_title(
        f"{model_name}: Realized Losses and Exceptions",
        fontsize=15,
        fontweight="bold",
    )

    axis.set_xlabel("Date")
    axis.set_ylabel("Daily loss")
    axis.legend(loc="upper left")
    axis.grid(alpha=0.25)

    _format_loss_axis(axis)

    figure.tight_layout()
    figure.savefig(
        normalized_output,
        dpi=160,
        bbox_inches="tight",
    )
    plt.close(figure)

    return normalized_output


def plot_var_model_comparison(
    dates: pd.Index,
    actual_losses: FloatArray,
    historical_var: FloatArray,
    parametric_var: FloatArray,
    output_path: str | Path,
) -> Path:
    """Compare historical and parametric rolling VaR."""
    (
        normalized_dates,
        normalized_losses,
        normalized_historical,
        _,
    ) = _validate_plot_inputs(
        dates=dates,
        actual_losses=actual_losses,
        var_forecasts=historical_var,
    )

    normalized_parametric = np.asarray(
        parametric_var,
        dtype=float,
    )

    if normalized_parametric.ndim != 1:
        raise ValueError(
            "Parametric VaR must be one-dimensional."
        )

    if len(normalized_parametric) != len(
        normalized_losses
    ):
        raise ValueError(
            "Parametric VaR and losses "
            "must have the same length."
        )

    if not np.all(
        np.isfinite(normalized_parametric)
    ):
        raise ValueError(
            "Parametric VaR must be finite."
        )

    normalized_output = _prepare_output_path(
        output_path
    )

    figure, axis = plt.subplots(
        figsize=(15, 7)
    )

    axis.plot(
        normalized_dates,
        normalized_losses,
        color="#9A9A9A",
        linewidth=0.6,
        alpha=0.45,
        label="Realized daily loss",
    )

    axis.plot(
        normalized_dates,
        normalized_historical,
        color="#4C72B0",
        linewidth=2.0,
        label="Historical rolling VaR",
    )

    axis.plot(
        normalized_dates,
        normalized_parametric,
        color="#DD8452",
        linewidth=2.0,
        label="Parametric rolling VaR",
    )

    axis.fill_between(
        normalized_dates,
        normalized_historical,
        normalized_parametric,
        color="#55A868",
        alpha=0.15,
        label="Difference between models",
    )

    axis.set_title(
        "Rolling Value at Risk Model Comparison",
        fontsize=15,
        fontweight="bold",
    )

    axis.set_xlabel("Date")
    axis.set_ylabel("Value at Risk")
    axis.legend(loc="upper left")
    axis.grid(alpha=0.25)

    _format_loss_axis(axis)

    figure.tight_layout()
    figure.savefig(
        normalized_output,
        dpi=160,
        bbox_inches="tight",
    )
    plt.close(figure)

    return normalized_output


def create_all_visualizations(
    result: MarketBacktestResult,
    output_directory: str | Path = (
        DEFAULT_FIGURES_DIRECTORY
    ),
) -> VisualizationPaths:
    """Generate every chart for a market backtest."""
    normalized_directory = Path(
        output_directory
    )

    forecast_dates = (
        result.market_data.returns.index[
            result.forecasts.window_size:
        ]
    )

    historical_path = plot_var_backtest(
        dates=forecast_dates,
        actual_losses=result.forecasts.actual_losses,
        var_forecasts=result.forecasts.historical_var,
        exceptions=(
            result
            .historical_model
            .backtest
            .exceptions
        ),
        model_name="Historical Rolling VaR",
        output_path=(
            normalized_directory
            / "historical_var_backtest.png"
        ),
    )

    parametric_path = plot_var_backtest(
        dates=forecast_dates,
        actual_losses=result.forecasts.actual_losses,
        var_forecasts=result.forecasts.parametric_var,
        exceptions=(
            result
            .parametric_model
            .backtest
            .exceptions
        ),
        model_name="Parametric Rolling VaR",
        output_path=(
            normalized_directory
            / "parametric_var_backtest.png"
        ),
    )

    comparison_path = plot_var_model_comparison(
        dates=forecast_dates,
        actual_losses=result.forecasts.actual_losses,
        historical_var=result.forecasts.historical_var,
        parametric_var=result.forecasts.parametric_var,
        output_path=(
            normalized_directory
            / "var_model_comparison.png"
        ),
    )

    return VisualizationPaths(
        historical_backtest=historical_path,
        parametric_backtest=parametric_path,
        model_comparison=comparison_path,
    )


def main() -> None:
    """Run the market backtest and generate its figures."""
    result = run_market_backtest(
        ticker="SPY",
        start_date="2021-01-01",
        end_date="2026-01-01",
        portfolio_value=1_000_000.0,
        window_size=250,
        confidence_level=0.99,
        significance_level=0.05,
    )

    paths = create_all_visualizations(result)

    print("Generated risk visualizations")
    print(f"- {paths.historical_backtest}")
    print(f"- {paths.parametric_backtest}")
    print(f"- {paths.model_comparison}")


if __name__ == "__main__":
    main()