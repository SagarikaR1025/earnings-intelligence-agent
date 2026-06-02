"""SEC EDGAR fetcher — resolves a ticker to CIK, lists filings, and downloads document text.

EDGAR requires a descriptive User-Agent header identifying the app and a contact email.
See: https://www.sec.gov/os/accessing-edgar-data
"""

import re
from datetime import date
from typing import Optional

import httpx
from bs4 import BeautifulSoup

from app.config import settings
from app.models.schemas import SECFiling

_BASE = "https://data.sec.gov"
_SUBMISSIONS_URL = _BASE + "/submissions/CIK{cik}.json"
_FILING_INDEX_URL = "https://www.sec.gov/Archives/edgar/full-index/{year}/{quarter}/company.idx"
_ARCHIVE_URL = "https://www.sec.gov/Archives/edgar/{accession_path}/{doc}"

_HEADERS = {
    "User-Agent": settings.sec_edgar_user_agent,
    "Accept-Encoding": "gzip, deflate",
    "Host": "data.sec.gov",
}

_ARCHIVE_HEADERS = {**_HEADERS, "Host": "www.sec.gov"}


def _cik_padded(cik: str) -> str:
    """Left-pad a CIK to 10 digits as EDGAR expects."""
    return cik.zfill(10)


async def resolve_cik(ticker: str) -> str:
    """Return the 10-digit CIK for a given ticker symbol.

    Args:
        ticker: Uppercase stock ticker (e.g. "AAPL").

    Returns:
        Zero-padded 10-digit CIK string.

    Raises:
        ValueError: If the ticker is not found in EDGAR.
    """
    tickers_url = "https://www.sec.gov/files/company_tickers.json"
    async with httpx.AsyncClient(headers=_ARCHIVE_HEADERS, timeout=15) as client:
        resp = await client.get(tickers_url)
        resp.raise_for_status()
        data = resp.json()

    for entry in data.values():
        if entry["ticker"].upper() == ticker.upper():
            return _cik_padded(str(entry["cik_str"]))

    raise ValueError(f"Ticker '{ticker}' not found in SEC EDGAR company list.")


async def get_filings(
    ticker: str,
    form_type: str = "10-K",
    num_filings: int = 1,
) -> list[SECFiling]:
    """Fetch the most recent SEC filings of a given type for a ticker.

    Args:
        ticker: Stock ticker symbol.
        form_type: SEC form type string (e.g. "10-K", "10-Q", "8-K").
        num_filings: Maximum number of filings to return.

    Returns:
        List of SECFiling objects (newest first), up to num_filings.
    """
    cik = await resolve_cik(ticker)
    submissions_url = _SUBMISSIONS_URL.format(cik=cik)

    async with httpx.AsyncClient(headers=_HEADERS, timeout=15) as client:
        resp = await client.get(submissions_url)
        resp.raise_for_status()
        data = resp.json()

    filings_data = data.get("filings", {}).get("recent", {})
    forms = filings_data.get("form", [])
    accessions = filings_data.get("accessionNumber", [])
    filed_dates = filings_data.get("filingDate", [])
    primary_docs = filings_data.get("primaryDocument", [])

    results: list[SECFiling] = []
    for form, accession, filed, primary_doc in zip(forms, accessions, filed_dates, primary_docs):
        if form.upper() != form_type.upper():
            continue

        accession_clean = accession.replace("-", "")
        # EDGAR archive URLs use /data/{cik_int}/ — the padded CIK is for the submissions API only
        cik_int = str(int(cik))
        accession_path = f"data/{cik_int}/{accession_clean}"
        filing_url = f"https://www.sec.gov/Archives/edgar/{accession_path}/{accession}.txt"
        document_url = f"https://www.sec.gov/Archives/edgar/{accession_path}/{primary_doc}"

        results.append(
            SECFiling(
                ticker=ticker.upper(),
                cik=cik,
                form_type=form,
                filed_date=date.fromisoformat(filed),
                accession_number=accession,
                filing_url=filing_url,
                document_url=document_url,
            )
        )
        if len(results) >= num_filings:
            break

    return results


async def fetch_filing_text(filing: SECFiling, max_chars: int = 80_000) -> SECFiling:
    """Download and extract plain text from the primary document of a filing.

    Strips HTML tags when the document is HTML/XBRL. Truncates to max_chars
    to stay within reasonable LLM context limits.

    Args:
        filing: An SECFiling with a populated document_url.
        max_chars: Maximum number of characters to keep from the document.

    Returns:
        The same SECFiling with text_content populated.
    """
    if not filing.document_url:
        return filing

    async with httpx.AsyncClient(headers=_ARCHIVE_HEADERS, timeout=30, follow_redirects=True) as client:
        resp = await client.get(filing.document_url)
        resp.raise_for_status()
        raw = resp.text

    # Strip HTML/XML markup. Check XML/XBRL before HTML — XHTML docs contain both declarations.
    preamble = raw[:500].lower()
    if "<?xml" in preamble or "<xbrl" in preamble:
        parser = "lxml-xml"
    elif "<html" in preamble or "<!doctype" in preamble:
        parser = "lxml"
    else:
        parser = None

    if parser:
        soup = BeautifulSoup(raw, parser)
        for tag in soup(["script", "style", "ix:header", "ix:nonnumeric"]):
            tag.decompose()
        text = soup.get_text(separator="\n")
    else:
        text = raw

    # Collapse excessive whitespace
    text = re.sub(r"\n{3,}", "\n\n", text).strip()

    filing.text_content = text[:max_chars]
    return filing
