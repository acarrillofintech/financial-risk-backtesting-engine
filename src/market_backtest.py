"""End-to-end Value at Risk backtesting with market data."""

from dataclasses import dataclass

import numpy as np

from src.backtesting import (
    VaRBacktestResult,
    backtest_value_at_risk,
)
from src.christoffersen import (
    ConditionalCoverageTestResult,
    conditional_coverage_test,
)
from src.data_loader import (
    MarketData,
    load_market_data,
)
from src.rolling_var import (
    RollingVaRForecasts,
    calculate_rolling_var_forecasts,
)


@dataclass(frozen=True)
class ModelBacktestResult:
    """Backtesting results for one rolling VaR model."""

    model_name: str
    average_var: float
    minimum_var: float
    maximum_var: float
    backtest: VaRBacktestResult
    conditional_coverage: ConditionalCoverageTestResult


@dataclass(frozen=True)
class MarketBacktestResult:
    """Complete market-data VaR backtesting result."""

    market_data: MarketData
    forecasts: RollingVaRForecasts
    historical_model: ModelBacktestResult
    parametric_model: ModelBacktestResult


def _evaluate_model(
    model_name: str,
    actual_losses: np.ndarray,
    var_forecasts: np.ndarray,
    confidence_level: float,
    significance_level: float,
) -> ModelBacktestResult:
    """Evaluate one sequence of rolling VaR forecasts."""
    backtest = backtest_value_at_risk(
        actual_losses=actual_losses,
        var_forecasts=var_forecasts,
        confidence_level=confidence_level,
        significance_level=significance_level,
    )

    conditional_coverage = conditional_coverage_test(
        exceptions=backtest.exceptions,
        confidence_level=confidence_level,
        significance_level=significance_level,
    )

    return ModelBacktestResult(
        model_name=model_name,
        average_var=float(np.mean(var_forecasts)),
        minimum_var=float(np.min(var_forecasts)),
        maximum_var=float(np.max(var_forecasts)),
        backtest=backtest,
        conditional_coverage=conditional_coverage,
    )


def run_market_backtest(
    ticker: str,
    start_date: str,
    end_date: str,
    portfolio_value: float = 1_000_000.0,
    window_size: int = 250,
    confidence_level: float = 0.99,
    significance_level: float = 0.05,
) -> MarketBacktestResult:
    """Run rolling historical and parametric VaR backtests."""
    market_data = load_market_data(
        ticker=ticker,
        start_date=start_date,
        end_date=end_date,
    )

    returns = market_data.returns.to_numpy(
        dtype=float
    )

    forecasts = calculate_rolling_var_forecasts(
        returns=returns,
        portfolio_value=portfolio_value,
        window_size=window_size,
        confidence_level=confidence_level,
    )

    historical_model = _evaluate_model(
        model_name="Historical rolling VaR",
        actual_losses=forecasts.actual_losses,
        var_forecasts=forecasts.historical_var,
        confidence_level=confidence_level,
        significance_level=significance_level,
    )

    parametric_model = _evaluate_model(
        model_name="Parametric rolling VaR",
        actual_losses=forecasts.actual_losses,
        var_forecasts=forecasts.parametric_var,
        confidence_level=confidence_level,
        significance_level=significance_level,
    )

    return MarketBacktestResult(
        market_data=market_data,
        forecasts=forecasts,
        historical_model=historical_model,
        parametric_model=parametric_model,
    )


def _print_model_result(
    result: ModelBacktestResult,
) -> None:
    """Print one model's VaR backtesting summary."""
    backtest = result.backtest
    independence = (
        result
        .conditional_coverage
        .independence_test
    )
    combined = result.conditional_coverage

    print(f"\n{result.model_name}")
    print(f"Average VaR: ${result.average_var:,.2f}")
    print(f"Minimum VaR: ${result.minimum_var:,.2f}")
    print(f"Maximum VaR: ${result.maximum_var:,.2f}")
    print(
        f"Expected exceptions: "
        f"{backtest.expected_exceptions:.2f}"
    )
    print(
        f"Actual exceptions: "
        f"{backtest.actual_exceptions}"
    )
    print(
        f"Exception rate: "
        f"{backtest.exception_rate:.2%}"
    )

    print("\nKupiec unconditional coverage")
    print(
        f"Statistic: "
        f"{backtest.kupiec_test.likelihood_ratio:.6f}"
    )
    print(
        f"P-value: "
        f"{backtest.kupiec_test.p_value:.6f}"
    )
    print(
        f"Rejected: "
        f"{backtest.kupiec_test.rejected}"
    )

    print("\nChristoffersen independence")
    print(
        f"Statistic: "
        f"{independence.likelihood_ratio:.6f}"
    )
    print(
        f"P-value: "
        f"{independence.p_value:.6f}"
    )
    print(
        f"Rejected: "
        f"{independence.rejected}"
    )

    print("\nConditional coverage")
    print(
        f"Statistic: "
        f"{combined.likelihood_ratio:.6f}"
    )
    print(f"P-value: {combined.p_value:.6f}")
    print(f"Rejected: {combined.rejected}")


def main() -> None:
    """Run a real-market SPY VaR backtest."""
    result = run_market_backtest(
        ticker="SPY",
        start_date="2021-01-01",
        end_date="2026-01-01",
        portfolio_value=1_000_000.0,
        window_size=250,
        confidence_level=0.99,
        significance_level=0.05,
    )

    prices = result.market_data.prices

    print("Financial risk backtesting engine")
    print(f"Asset: {result.market_data.ticker}")
    print(
        f"Market period: "
        f"{prices.index.min().date()} "
        f"to {prices.index.max().date()}"
    )
    print(
        f"Price observations: "
        f"{len(prices):,}"
    )
    print(
        f"Out-of-sample forecasts: "
        f"{len(result.forecasts.actual_losses):,}"
    )
    print(
        f"Portfolio value: "
        f"${result.forecasts.portfolio_value:,.2f}"
    )
    print(
        f"Confidence level: "
        f"{result.forecasts.confidence_level:.2%}"
    )
    print(
        f"Rolling window: "
        f"{result.forecasts.window_size} days"
    )

    _print_model_result(
        result.historical_model
    )

    _print_model_result(
        result.parametric_model
    )


if __name__ == "__main__":
    main()