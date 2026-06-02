"""Pydantic schemas shared across the application."""

from datetime import date, datetime
from typing import Optional

from pydantic import BaseModel, Field


# ── SEC / EDGAR ───────────────────────────────────────────────────────────────

class SECFiling(BaseModel):
    """A single SEC EDGAR filing record (returned by /filings endpoint)."""

    ticker: str
    cik: str
    form_type: str = Field(description="e.g. 10-K, 10-Q, 8-K")
    filed_date: date
    accession_number: str
    filing_url: str
    document_url: Optional[str] = None
    text_content: Optional[str] = Field(
        default=None, description="Extracted plain text of the primary document"
    )


class FilingMeta(BaseModel):
    """Lightweight filing metadata embedded in analysis responses.

    Excludes the raw text_content to keep response payloads small.
    """

    form_type: str
    filed_date: date
    accession_number: str
    document_url: Optional[str] = None


# ── Yahoo Finance ─────────────────────────────────────────────────────────────

class FinancialSnapshot(BaseModel):
    """Key financial metrics pulled from Yahoo Finance."""

    ticker: str
    company_name: Optional[str] = None
    sector: Optional[str] = None
    industry: Optional[str] = None
    market_cap: Optional[float] = None
    pe_ratio: Optional[float] = None
    forward_pe: Optional[float] = None
    price_to_book: Optional[float] = None
    revenue_ttm: Optional[float] = None
    net_income_ttm: Optional[float] = None
    eps_ttm: Optional[float] = None
    dividend_yield: Optional[float] = None
    fifty_two_week_high: Optional[float] = None
    fifty_two_week_low: Optional[float] = None
    current_price: Optional[float] = None
    analyst_target_price: Optional[float] = None
    recommendation: Optional[str] = None


# ── Analysis pipeline ─────────────────────────────────────────────────────────

class SentimentInfo(BaseModel):
    """Quantitative sentiment result for the SEC filing text."""

    score: float = Field(description="Compound score in [-1.0, 1.0]; positive = bullish")
    label: str = Field(description="'positive', 'neutral', or 'negative'")
    confidence: float = Field(description="Model confidence [0.0, 1.0]")


class BriefSections(BaseModel):
    """Structured sections of the Claude-generated investment research brief."""

    company_overview: str = ""
    earnings_analysis: str = ""
    sentiment_assessment: str = ""
    bull_case: str = ""
    bear_case: str = ""
    key_risks: str = ""
    investment_outlook: str = ""
    model_used: str = Field(description="Claude model ID that generated this brief")
    generated_at: datetime


# ── Request / response ────────────────────────────────────────────────────────

class AnalysisRequest(BaseModel):
    """Request body for the legacy POST /analyze body-param endpoint (kept for compat)."""

    ticker: str = Field(description="Stock ticker symbol, e.g. AAPL")
    form_type: str = Field(default="10-K", description="SEC form type to retrieve")
    num_filings: int = Field(default=1, ge=1, le=5)


class AnalysisResponse(BaseModel):
    """Full pipeline result returned by POST /analyze/{ticker}."""

    ticker: str
    company_name: Optional[str] = None
    filing: Optional[FilingMeta] = None
    sentiment: Optional[SentimentInfo] = None
    brief: Optional[BriefSections] = None
    error: Optional[str] = None
