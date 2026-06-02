"""Smoke tests for the SEC EDGAR and Yahoo Finance fetchers.

Run with: pytest tests/ -v
These hit live APIs — use sparingly and cache results locally for CI.
"""

import pytest

from app.fetchers.sec_edgar import get_filings, resolve_cik
from app.fetchers.yahoo_finance import get_financial_snapshot


@pytest.mark.asyncio
async def test_resolve_cik_apple():
    cik = await resolve_cik("AAPL")
    assert cik == "0000320193"


@pytest.mark.asyncio
async def test_get_filings_returns_10k():
    filings = await get_filings("AAPL", form_type="10-K", num_filings=1)
    assert len(filings) == 1
    assert filings[0].form_type == "10-K"
    assert filings[0].ticker == "AAPL"
    assert filings[0].document_url is not None


def test_financial_snapshot_apple():
    snapshot = get_financial_snapshot("AAPL")
    assert snapshot.ticker == "AAPL"
    assert snapshot.company_name is not None
    assert snapshot.current_price is not None
    assert snapshot.market_cap is not None
