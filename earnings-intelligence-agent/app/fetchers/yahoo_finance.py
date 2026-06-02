"""Yahoo Finance fetcher — wraps yfinance to pull a standardised financial snapshot.

yfinance is a third-party wrapper; it may break when Yahoo changes its API.
All fields are Optional so a partial failure doesn't abort the whole pipeline.
"""

from typing import Optional

import yfinance as yf

from app.models.schemas import FinancialSnapshot


def _safe_get(info: dict, key: str) -> Optional[float | str]:
    """Return info[key] or None if missing / non-finite."""
    val = info.get(key)
    if val is None:
        return None
    if isinstance(val, float) and (val != val):  # NaN check
        return None
    return val


def get_financial_snapshot(ticker: str) -> FinancialSnapshot:
    """Fetch key financial metrics for a ticker via Yahoo Finance.

    Pulls from the yfinance Ticker.info dict. All numeric fields are Optional;
    missing data is represented as None rather than raising an error.

    Args:
        ticker: Stock ticker symbol (e.g. "AAPL").

    Returns:
        A FinancialSnapshot populated with whatever Yahoo Finance provides.

    Raises:
        ValueError: If yfinance returns an empty info dict (unknown ticker).
    """
    yticker = yf.Ticker(ticker.upper())
    info: dict = yticker.info

    if not info or info.get("quoteType") is None:
        raise ValueError(f"No Yahoo Finance data found for ticker '{ticker}'.")

    return FinancialSnapshot(
        ticker=ticker.upper(),
        company_name=_safe_get(info, "longName"),
        sector=_safe_get(info, "sector"),
        industry=_safe_get(info, "industry"),
        market_cap=_safe_get(info, "marketCap"),
        pe_ratio=_safe_get(info, "trailingPE"),
        forward_pe=_safe_get(info, "forwardPE"),
        price_to_book=_safe_get(info, "priceToBook"),
        revenue_ttm=_safe_get(info, "totalRevenue"),
        net_income_ttm=_safe_get(info, "netIncomeToCommon"),
        eps_ttm=_safe_get(info, "trailingEps"),
        dividend_yield=_safe_get(info, "dividendYield"),
        fifty_two_week_high=_safe_get(info, "fiftyTwoWeekHigh"),
        fifty_two_week_low=_safe_get(info, "fiftyTwoWeekLow"),
        current_price=_safe_get(info, "currentPrice") or _safe_get(info, "regularMarketPrice"),
        analyst_target_price=_safe_get(info, "targetMeanPrice"),
        recommendation=_safe_get(info, "recommendationKey"),
    )
