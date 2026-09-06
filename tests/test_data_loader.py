"""Tests for the historical market-data loader."""

import numpy as np
import pandas as pd
import pytest

import src.data_loader as data_loader
from src.data_loader import (
    MarketData,
    _extract_close_prices,
    calculate_simple_returns,
    download_adjusted_prices,
    load_market_data,
)


def sample_prices() -> pd.Series:
    """Create a deterministic price series."""
    index = pd.date_range(
        "2025-01-01",
        periods=3,
        freq="D",
    )

    return pd.Series(
        [100.0, 110.0, 99.0],
        index=index,
        name="SPY",
    )


def test_calculate_simple_returns() -> None:
    """Percentage returns should match known values."""
    result = calculate_simple_returns(
        sample_prices()
    )

    assert result.to_numpy() == pytest.approx(
        np.array([0.10, -0.10])
    )


def test_returns_preserve_series_name() -> None:
    """The ticker name should be preserved."""
    result = calculate_simple_returns(
        sample_prices()
    )

    assert result.name == "SPY"


def test_extract_flat_close_prices() -> None:
    """Close prices should be extracted from flat columns."""
    frame = pd.DataFrame(
        {
            "Open": [99.0, 109.0],
            "Close": [100.0, 110.0],
        },
        index=pd.date_range(
            "2025-01-01",
            periods=2,
        ),
    )

    result = _extract_close_prices(
        downloaded_data=frame,
        ticker="SPY",
    )

    assert result.name == "SPY"
    assert result.to_numpy() == pytest.approx(
        np.array([100.0, 110.0])
    )


def test_extract_multiindex_close_prices() -> None:
    """Yahoo MultiIndex columns should be supported."""
    columns = pd.MultiIndex.from_tuples(
        [("Close", "SPY")]
    )

    frame = pd.DataFrame(
        [[100.0], [101.0]],
        columns=columns,
        index=pd.date_range(
            "2025-01-01",
            periods=2,
        ),
    )

    result = _extract_close_prices(
        downloaded_data=frame,
        ticker="SPY",
    )

    assert result.to_numpy() == pytest.approx(
        np.array([100.0, 101.0])
    )


def test_download_normalizes_ticker(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Ticker should be stripped and converted to uppercase."""
    captured_arguments: dict[str, object] = {}

    def fake_download(**kwargs: object) -> pd.DataFrame:
        captured_arguments.update(kwargs)

        return pd.DataFrame(
            {"Close": [100.0, 101.0]},
            index=pd.date_range(
                "2025-01-01",
                periods=2,
            ),
        )

    monkeypatch.setattr(
        data_loader.yf,
        "download",
        fake_download,
    )

    result = download_adjusted_prices(
        ticker=" spy ",
        start_date="2025-01-01",
        end_date="2025-02-01",
    )

    assert captured_arguments["tickers"] == "SPY"
    assert captured_arguments["auto_adjust"] is True
    assert captured_arguments["progress"] is False
    assert captured_arguments["threads"] is False
    assert result.name == "SPY"


def test_load_market_data(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The combined loader should return prices and returns."""
    prices = sample_prices()

    def fake_download_adjusted_prices(
        ticker: str,
        start_date: str,
        end_date: str,
    ) -> pd.Series:
        assert ticker == "SPY"
        assert start_date == "2025-01-01"
        assert end_date == "2025-02-01"
        return prices

    monkeypatch.setattr(
        data_loader,
        "download_adjusted_prices",
        fake_download_adjusted_prices,
    )

    result = load_market_data(
        ticker="spy",
        start_date="2025-01-01",
        end_date="2025-02-01",
    )

    assert isinstance(result, MarketData)
    assert result.ticker == "SPY"
    assert result.prices.equals(prices)
    assert len(result.returns) == 2
    assert result.start_date == "2025-01-01"
    assert result.end_date == "2025-02-01"


def test_empty_download_raises_runtime_error() -> None:
    """An empty download should be rejected."""
    with pytest.raises(
        RuntimeError,
        match="No market data",
    ):
        _extract_close_prices(
            downloaded_data=pd.DataFrame(),
            ticker="SPY",
        )


def test_missing_close_column_raises_runtime_error() -> None:
    """Downloaded data must contain closing prices."""
    frame = pd.DataFrame(
        {"Open": [100.0, 101.0]}
    )

    with pytest.raises(
        RuntimeError,
        match="does not contain Close",
    ):
        _extract_close_prices(
            downloaded_data=frame,
            ticker="SPY",
        )


@pytest.mark.parametrize(
    "invalid_price",
    [0.0, -1.0],
)
def test_nonpositive_downloaded_price_raises_runtime_error(
    invalid_price: float,
) -> None:
    """Downloaded closing prices must be positive."""
    frame = pd.DataFrame(
        {"Close": [100.0, invalid_price]}
    )

    with pytest.raises(
        RuntimeError,
        match="must be positive",
    ):
        _extract_close_prices(
            downloaded_data=frame,
            ticker="SPY",
        )


def test_non_series_prices_raise_type_error() -> None:
    """Return calculation requires a pandas Series."""
    with pytest.raises(
        TypeError,
        match="pandas Series",
    ):
        calculate_simple_returns(
            [100.0, 101.0]  # type: ignore[arg-type]
        )


def test_single_price_raises_value_error() -> None:
    """At least two prices should be required."""
    with pytest.raises(
        ValueError,
        match="At least two",
    ):
        calculate_simple_returns(
            pd.Series([100.0])
        )


def test_nan_price_raises_value_error() -> None:
    """Missing prices should be rejected."""
    with pytest.raises(
        ValueError,
        match="finite numeric",
    ):
        calculate_simple_returns(
            pd.Series([100.0, np.nan])
        )


def test_infinite_price_raises_value_error() -> None:
    """Infinite prices should be rejected."""
    with pytest.raises(
        ValueError,
        match="finite numeric",
    ):
        calculate_simple_returns(
            pd.Series([100.0, np.inf])
        )


@pytest.mark.parametrize(
    "invalid_price",
    [0.0, -100.0],
)
def test_nonpositive_price_raises_value_error(
    invalid_price: float,
) -> None:
    """Input prices must be positive."""
    with pytest.raises(
        ValueError,
        match="must be positive",
    ):
        calculate_simple_returns(
            pd.Series([100.0, invalid_price])
        )


@pytest.mark.parametrize(
    "ticker",
    ["", "   "],
)
def test_empty_ticker_raises_value_error(
    ticker: str,
) -> None:
    """Ticker cannot be empty."""
    with pytest.raises(
        ValueError,
        match="cannot be empty",
    ):
        download_adjusted_prices(
            ticker=ticker,
            start_date="2025-01-01",
            end_date="2025-02-01",
        )


def test_non_string_ticker_raises_type_error() -> None:
    """Ticker must be text."""
    with pytest.raises(
        TypeError,
        match="must be a string",
    ):
        download_adjusted_prices(
            ticker=123,  # type: ignore[arg-type]
            start_date="2025-01-01",
            end_date="2025-02-01",
        )


def test_reversed_date_range_raises_value_error() -> None:
    """Start date must occur before end date."""
    with pytest.raises(
        ValueError,
        match="earlier than end date",
    ):
        download_adjusted_prices(
            ticker="SPY",
            start_date="2025-02-01",
            end_date="2025-01-01",
        )


def test_equal_dates_raise_value_error() -> None:
    """Start and end dates cannot be equal."""
    with pytest.raises(
        ValueError,
        match="earlier than end date",
    ):
        download_adjusted_prices(
            ticker="SPY",
            start_date="2025-01-01",
            end_date="2025-01-01",
        )


def test_invalid_date_raises_value_error() -> None:
    """Invalid calendar dates should be rejected."""
    with pytest.raises(
        ValueError,
        match="valid date",
    ):
        download_adjusted_prices(
            ticker="SPY",
            start_date="not-a-date",
            end_date="2025-02-01",
        )


def test_download_failure_raises_runtime_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Provider errors should become clear runtime errors."""
    def fake_download(**kwargs: object) -> pd.DataFrame:
        raise OSError("Network unavailable")

    monkeypatch.setattr(
        data_loader.yf,
        "download",
        fake_download,
    )

    with pytest.raises(
        RuntimeError,
        match="Unable to download",
    ):
        download_adjusted_prices(
            ticker="SPY",
            start_date="2025-01-01",
            end_date="2025-02-01",
        )