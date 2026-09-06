"""Download and prepare historical market data."""

from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf


YFINANCE_CACHE_DIRECTORY = (
    Path(__file__).resolve().parents[1]
    / ".yfinance_cache"
)

yf.set_tz_cache_location(
    str(YFINANCE_CACHE_DIRECTORY)
)


@dataclass(frozen=True)
class MarketData:
    """Historical prices and daily returns for one asset."""

    ticker: str
    prices: pd.Series
    returns: pd.Series
    start_date: str
    end_date: str


def _validate_ticker(ticker: str) -> str:
    """Validate and normalize a market ticker."""
    if not isinstance(ticker, str):
        raise TypeError(
            "Ticker must be a string."
        )

    normalized_ticker = ticker.strip().upper()

    if not normalized_ticker:
        raise ValueError(
            "Ticker cannot be empty."
        )

    return normalized_ticker


def _normalize_date(
    value: str | date | datetime,
    field_name: str,
) -> str:
    """Validate and convert a date to ISO format."""
    try:
        normalized_date = pd.Timestamp(value)
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"{field_name} must be a valid date."
        ) from error

    if pd.isna(normalized_date):
        raise ValueError(
            f"{field_name} must be a valid date."
        )

    return normalized_date.strftime("%Y-%m-%d")


def _validate_date_range(
    start_date: str | date | datetime,
    end_date: str | date | datetime,
) -> tuple[str, str]:
    """Validate the historical-data date range."""
    normalized_start = _normalize_date(
        start_date,
        "Start date",
    )

    normalized_end = _normalize_date(
        end_date,
        "End date",
    )

    if normalized_start >= normalized_end:
        raise ValueError(
            "Start date must be earlier than end date."
        )

    return normalized_start, normalized_end


def _extract_close_prices(
    downloaded_data: pd.DataFrame,
    ticker: str,
) -> pd.Series:
    """Extract a clean closing-price series."""
    if downloaded_data.empty:
        raise RuntimeError(
            f"No market data was found for {ticker}."
        )

    if isinstance(
        downloaded_data.columns,
        pd.MultiIndex,
    ):
        first_level = (
            downloaded_data
            .columns
            .get_level_values(0)
        )

        if "Close" not in first_level:
            raise RuntimeError(
                "Downloaded data does not contain "
                "Close prices."
            )

        close_data = downloaded_data["Close"]

        if isinstance(close_data, pd.DataFrame):
            if ticker in close_data.columns:
                prices = close_data[ticker]
            elif close_data.shape[1] == 1:
                prices = close_data.iloc[:, 0]
            else:
                raise RuntimeError(
                    f"Close prices for {ticker} "
                    "were not found."
                )
        else:
            prices = close_data

    else:
        if "Close" not in downloaded_data.columns:
            raise RuntimeError(
                "Downloaded data does not contain "
                "Close prices."
            )

        prices = downloaded_data["Close"]

    normalized_prices = pd.Series(
        prices,
        index=prices.index,
        name=ticker,
        dtype=float,
    )

    normalized_prices = (
        normalized_prices
        .replace(
            [np.inf, -np.inf],
            np.nan,
        )
        .dropna()
        .sort_index()
    )

    if normalized_prices.empty:
        raise RuntimeError(
            f"No valid closing prices were found "
            f"for {ticker}."
        )

    if np.any(normalized_prices <= 0.0):
        raise RuntimeError(
            "Closing prices must be positive."
        )

    return normalized_prices


def download_adjusted_prices(
    ticker: str,
    start_date: str | date | datetime,
    end_date: str | date | datetime,
) -> pd.Series:
    """Download adjusted historical closing prices."""
    normalized_ticker = _validate_ticker(ticker)

    normalized_start, normalized_end = (
        _validate_date_range(
            start_date=start_date,
            end_date=end_date,
        )
    )

    try:
        downloaded_data = yf.download(
            tickers=normalized_ticker,
            start=normalized_start,
            end=normalized_end,
            auto_adjust=True,
            progress=False,
            threads=False,
        )
    except Exception as error:
        raise RuntimeError(
            "Unable to download market data "
            f"for {normalized_ticker}."
        ) from error

    return _extract_close_prices(
        downloaded_data=downloaded_data,
        ticker=normalized_ticker,
    )


def calculate_simple_returns(
    prices: pd.Series,
) -> pd.Series:
    """Calculate daily percentage returns from prices."""
    if not isinstance(prices, pd.Series):
        raise TypeError(
            "Prices must be a pandas Series."
        )

    normalized_prices = pd.to_numeric(
        prices,
        errors="coerce",
    )

    if len(normalized_prices) < 2:
        raise ValueError(
            "At least two price observations "
            "are required."
        )

    if normalized_prices.isna().any():
        raise ValueError(
            "Prices must contain only finite "
            "numeric values."
        )

    price_values = normalized_prices.to_numpy(
        dtype=float
    )

    if not np.all(np.isfinite(price_values)):
        raise ValueError(
            "Prices must contain only finite "
            "numeric values."
        )

    if np.any(price_values <= 0.0):
        raise ValueError(
            "Prices must be positive."
        )

    returns = (
        normalized_prices
        .pct_change()
        .dropna()
    )

    returns.name = (
        prices.name
        if prices.name is not None
        else "returns"
    )

    return returns.astype(float)


def load_market_data(
    ticker: str,
    start_date: str | date | datetime,
    end_date: str | date | datetime,
) -> MarketData:
    """Download prices and calculate aligned returns."""
    normalized_ticker = _validate_ticker(ticker)

    normalized_start, normalized_end = (
        _validate_date_range(
            start_date=start_date,
            end_date=end_date,
        )
    )

    prices = download_adjusted_prices(
        ticker=normalized_ticker,
        start_date=normalized_start,
        end_date=normalized_end,
    )

    returns = calculate_simple_returns(prices)

    return MarketData(
        ticker=normalized_ticker,
        prices=prices,
        returns=returns,
        start_date=normalized_start,
        end_date=normalized_end,
    )


def main() -> None:
    """Download SPY data and display a summary."""
    market_data = load_market_data(
        ticker="SPY",
        start_date="2021-01-01",
        end_date="2026-01-01",
    )

    print("Historical market data")
    print(f"Ticker: {market_data.ticker}")
    print(
        "Period: "
        f"{market_data.prices.index.min().date()} "
        "to "
        f"{market_data.prices.index.max().date()}"
    )
    print(
        "Price observations: "
        f"{len(market_data.prices):,}"
    )
    print(
        "Return observations: "
        f"{len(market_data.returns):,}"
    )

    print("\nLatest adjusted prices")
    print(market_data.prices.tail())

    print("\nReturn statistics")
    print(
        "Average daily return: "
        f"{market_data.returns.mean():.4%}"
    )
    print(
        "Daily volatility: "
        f"{market_data.returns.std(ddof=1):.4%}"
    )
    print(
        "Worst daily return: "
        f"{market_data.returns.min():.4%}"
    )
    print(
        "Best daily return: "
        f"{market_data.returns.max():.4%}"
    )


if __name__ == "__main__":
    main()