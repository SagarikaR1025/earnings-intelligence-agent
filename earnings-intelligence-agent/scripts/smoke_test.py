"""Quick CLI smoke test — run directly to verify fetchers work end-to-end.

Usage:
    python scripts/smoke_test.py AAPL
"""

import asyncio
import sys

from app.fetchers.sec_edgar import fetch_filing_text, get_filings
from app.fetchers.yahoo_finance import get_financial_snapshot


async def main(ticker: str) -> None:
    print(f"\n--- Yahoo Finance snapshot for {ticker} ---")
    snap = get_financial_snapshot(ticker)
    print(f"  Company   : {snap.company_name}")
    print(f"  Price     : ${snap.current_price}")
    print(f"  Market cap: ${snap.market_cap:,.0f}" if snap.market_cap else "  Market cap: N/A")
    print(f"  P/E (TTM) : {snap.pe_ratio}")
    print(f"  Recommend : {snap.recommendation}")

    print(f"\n--- SEC EDGAR latest 10-K for {ticker} ---")
    filings = await get_filings(ticker, form_type="10-K", num_filings=1)
    if not filings:
        print("  No filings found.")
        return

    filing = filings[0]
    print(f"  Filed     : {filing.filed_date}")
    print(f"  Accession : {filing.accession_number}")
    print(f"  Doc URL   : {filing.document_url}")

    print("\n  Fetching filing text (first 500 chars)...")
    filing = await fetch_filing_text(filing, max_chars=500)
    if filing.text_content:
        print(filing.text_content[:500])


if __name__ == "__main__":
    ticker = sys.argv[1] if len(sys.argv) > 1 else "AAPL"
    asyncio.run(main(ticker))
