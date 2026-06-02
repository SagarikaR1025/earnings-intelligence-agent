"""FastAPI route definitions for the Earnings Intelligence Agent.

Pipeline for POST /analyze/{ticker}:
    1. Concurrently fetch SEC EDGAR filing list + Yahoo Finance snapshot.
    2. Download the filing's primary document text from EDGAR.
    3. Run SentimentAnalyzer on the filing text (CPU-bound, offloaded to a thread).
    4. Call BriefGenerator.generate_async() to produce the structured brief via Claude.
    5. Return a clean AnalysisResponse with filing metadata, sentiment, and brief sections.

Singletons (_analyzer, _generator) are initialized lazily on first use via
_get_analyzer() / _get_generator() so the module can be imported without
triggering the sklearn training or Anthropic client construction at import time.
Call warm_up() from the FastAPI lifespan to pre-train eagerly at startup instead.
"""

import asyncio
import logging
from typing import Annotated, Optional

import anthropic
from fastapi import APIRouter, HTTPException, Query

from app.analysis.brief import BriefGenerator
from app.analysis.sentiment import SentimentAnalyzer
from app.fetchers.sec_edgar import fetch_filing_text, get_filings
from app.fetchers.yahoo_finance import get_financial_snapshot
from app.models.schemas import (
    AnalysisResponse,
    BriefSections,
    FilingMeta,
    FinancialSnapshot,
    SECFiling,
    SentimentInfo,
)

log = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1", tags=["earnings"])

# ── Lazy-initialized singletons ───────────────────────────────────────────────

_analyzer: Optional[SentimentAnalyzer] = None
_generator: Optional[BriefGenerator] = None


def _get_analyzer() -> SentimentAnalyzer:
    """Return the shared SentimentAnalyzer, training it on first call."""
    global _analyzer
    if _analyzer is None:
        log.info("Fitting SentimentAnalyzer on internal finance corpus…")
        _analyzer = SentimentAnalyzer().fit()
        log.info("SentimentAnalyzer ready.")
    return _analyzer


def _get_generator() -> BriefGenerator:
    """Return the shared BriefGenerator, constructing it on first call."""
    global _generator
    if _generator is None:
        log.info("Initializing BriefGenerator…")
        _generator = BriefGenerator()
        log.info("BriefGenerator ready.")
    return _generator


async def warm_up() -> None:
    """Pre-initialize ML components during server startup.

    Called by the FastAPI lifespan handler in main.py so the first real
    request does not pay the one-time initialization cost.
    """
    await asyncio.to_thread(_get_analyzer)
    _get_generator()


# ── Utility routes ────────────────────────────────────────────────────────────

@router.get("/health")
async def health() -> dict:
    """Liveness check."""
    return {"status": "ok"}


@router.get("/financials/{ticker}", response_model=FinancialSnapshot)
async def financials(ticker: str) -> FinancialSnapshot:
    """Return a financial snapshot for a ticker from Yahoo Finance.

    Args:
        ticker: Stock ticker symbol (e.g. AAPL).
    """
    try:
        return await asyncio.to_thread(get_financial_snapshot, ticker)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.get("/filings/{ticker}", response_model=list[SECFiling])
async def filings(
    ticker: str,
    form_type: Annotated[str, Query(description="SEC form type")] = "10-K",
    num_filings: Annotated[int, Query(ge=1, le=5)] = 1,
    include_text: Annotated[bool, Query(description="Download and attach filing text")] = False,
) -> list[SECFiling]:
    """List recent SEC filings for a ticker.

    Args:
        ticker: Stock ticker symbol.
        form_type: Form type to filter by (10-K, 10-Q, 8-K …).
        num_filings: How many filings to return (max 5).
        include_text: If true, also fetches and attaches the full document text.
    """
    try:
        results = await get_filings(ticker, form_type=form_type, num_filings=num_filings)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    if include_text:
        results = list(await asyncio.gather(*[fetch_filing_text(f) for f in results]))

    return results


# ── Core pipeline ─────────────────────────────────────────────────────────────

@router.post("/analyze/{ticker}", response_model=AnalysisResponse)
async def analyze(
    ticker: str,
    form_type: Annotated[
        str, Query(description="SEC form type to retrieve (10-K, 10-Q, 8-K)")
    ] = "10-K",
) -> AnalysisResponse:
    """Run the full Earnings Intelligence pipeline for a stock ticker.

    Fetches the most recent SEC filing of the requested type, downloads its
    text, runs sentiment analysis, and generates a Goldman Sachs–style
    investment research brief via the Claude API.

    Args:
        ticker:    Stock ticker symbol (case-insensitive, e.g. AAPL).
        form_type: SEC filing type — 10-K (annual), 10-Q (quarterly), 8-K (current).

    Returns:
        AnalysisResponse with filing metadata, sentiment scores, and all
        seven structured sections of the research brief.

    Raises:
        404: Ticker not found in SEC EDGAR or Yahoo Finance.
        401: Invalid Anthropic API key.
        429: Claude API rate limit exceeded.
        502: Upstream data source (EDGAR / Claude) returned an error.
        500: Unexpected internal error.
    """
    ticker = ticker.upper()
    log.info("━━ ANALYZE %s (%s) ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━", ticker, form_type)

    # ── Step 1: Fetch SEC filing list + market data concurrently ─────────────
    log.info("[1/4] Fetching SEC EDGAR filings and Yahoo Finance snapshot…")
    try:
        filings_list, snapshot = await asyncio.gather(
            get_filings(ticker, form_type=form_type, num_filings=1),
            asyncio.to_thread(get_financial_snapshot, ticker),
        )
    except ValueError as exc:
        log.warning("Ticker lookup failed for %s: %s", ticker, exc)
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        log.error("Data fetch error for %s: %s", ticker, exc, exc_info=True)
        raise HTTPException(
            status_code=502,
            detail=f"Failed to fetch data for {ticker}: {exc}",
        ) from exc

    if not filings_list:
        log.warning("No %s filings found for %s in EDGAR.", form_type, ticker)
        raise HTTPException(
            status_code=404,
            detail=f"No {form_type} filings found for {ticker} in SEC EDGAR.",
        )

    filing = filings_list[0]
    log.info(
        "[1/4] ✓ %s filed %s  |  %s  |  price=$%.2f  mktcap=$%.1fB",
        filing.form_type,
        filing.filed_date,
        snapshot.company_name or ticker,
        snapshot.current_price or 0.0,
        (snapshot.market_cap or 0.0) / 1e9,
    )

    # ── Step 2: Download filing document text ─────────────────────────────────
    log.info("[2/4] Downloading filing text from EDGAR (%s)…", filing.document_url)
    try:
        filing = await fetch_filing_text(filing)
        text_len = len(filing.text_content or "")
        log.info("[2/4] ✓ Filing text fetched: %d chars", text_len)
    except Exception as exc:
        # Non-fatal: proceed with empty text; sentiment and brief will reflect
        # the missing content through their own handling.
        log.warning(
            "[2/4] Filing text fetch failed for %s (%s). Continuing with empty text: %s",
            ticker,
            filing.accession_number,
            exc,
        )

    filing_text = filing.text_content or ""

    # ── Step 3: Sentiment analysis ────────────────────────────────────────────
    log.info("[3/4] Running sentiment analysis on %d chars of filing text…", len(filing_text))
    try:
        # _get_analyzer() trains sklearn on first call — offload to a thread.
        analyzer = await asyncio.to_thread(_get_analyzer)
        sentiment_result = await asyncio.to_thread(analyzer.analyze_text, filing_text)
    except Exception as exc:
        log.error("[3/4] Sentiment analysis error for %s: %s", ticker, exc, exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Sentiment analysis failed: {exc}",
        ) from exc

    log.info(
        "[3/4] ✓ Sentiment: score=%+.4f  label=%-8s  confidence=%.4f",
        sentiment_result.score,
        sentiment_result.label,
        sentiment_result.confidence,
    )

    # ── Step 4: Generate research brief via Claude ────────────────────────────
    generator = _get_generator()
    log.info("[4/4] Generating research brief via Claude (model=%s)…", generator._model)
    try:
        brief = await generator.generate_async(filing, snapshot, sentiment_result)
    except anthropic.AuthenticationError as exc:
        log.error("[4/4] Anthropic authentication error — check ANTHROPIC_API_KEY.")
        raise HTTPException(
            status_code=401,
            detail="Invalid Anthropic API key. Set ANTHROPIC_API_KEY in .env.",
        ) from exc
    except anthropic.RateLimitError as exc:
        log.warning("[4/4] Claude rate limit hit for %s.", ticker)
        raise HTTPException(
            status_code=429,
            detail="Claude API rate limit exceeded. Please retry in a moment.",
        ) from exc
    except anthropic.APIStatusError as exc:
        log.error(
            "[4/4] Claude API status error %d for %s: %s (request_id=%s)",
            exc.status_code,
            ticker,
            exc.message,
            exc._request_id,
        )
        raise HTTPException(
            status_code=502,
            detail=f"Claude API error ({exc.status_code}): {exc.message}",
        ) from exc
    except anthropic.APIConnectionError as exc:
        log.error("[4/4] Claude connection error for %s: %s", ticker, exc)
        raise HTTPException(
            status_code=502,
            detail="Could not connect to the Claude API. Check network and retry.",
        ) from exc
    except Exception as exc:
        log.error("[4/4] Brief generation error for %s: %s", ticker, exc, exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Brief generation failed: {exc}",
        ) from exc

    log.info(
        "[4/4] ✓ Brief complete — in=%s out=%s cache_read=%s cache_write=%s",
        brief.input_tokens,
        brief.output_tokens,
        brief.cache_read_tokens,
        brief.cache_creation_tokens,
    )
    log.info(
        "━━ DONE %s ── outlook: %s ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
        ticker,
        brief.investment_outlook[:80].replace("\n", " ") + "…" if brief.investment_outlook else "N/A",
    )

    # ── Build and return structured response ──────────────────────────────────
    return AnalysisResponse(
        ticker=ticker,
        company_name=snapshot.company_name,
        filing=FilingMeta(
            form_type=filing.form_type,
            filed_date=filing.filed_date,
            accession_number=filing.accession_number,
            document_url=filing.document_url,
        ),
        sentiment=SentimentInfo(
            score=sentiment_result.score,
            label=sentiment_result.label,
            confidence=sentiment_result.confidence,
        ),
        brief=BriefSections(
            company_overview=brief.company_overview,
            earnings_analysis=brief.earnings_analysis,
            sentiment_assessment=brief.sentiment_assessment,
            bull_case=brief.bull_case,
            bear_case=brief.bear_case,
            key_risks=brief.key_risks,
            investment_outlook=brief.investment_outlook,
            model_used=brief.model,
            generated_at=brief.generated_at,
        ),
    )
