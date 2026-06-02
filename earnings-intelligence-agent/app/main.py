"""Application entry point — creates and configures the FastAPI app.

Lifespan tasks:
    startup  — configure logging, pre-warm the SentimentAnalyzer and BriefGenerator
    shutdown — log a clean exit message
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api.routes import router, warm_up

_LOG_FORMAT = "%(asctime)s  %(levelname)-8s  %(name)s  %(message)s"
_LOG_DATE   = "%H:%M:%S"

logging.basicConfig(format=_LOG_FORMAT, datefmt=_LOG_DATE, level=logging.INFO)
log = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Configure startup and shutdown tasks."""
    log.info("━━ Earnings Intelligence Agent — starting up ━━━━━━━━━━━━━━━━")
    log.info("Pre-warming ML pipeline components…")
    await warm_up()
    log.info("All components ready.  API available at /docs")
    log.info("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

    yield  # server is running

    log.info("Earnings Intelligence Agent — shutting down.")


app = FastAPI(
    title="Earnings Intelligence Agent",
    description=(
        "Agentic system for SEC EDGAR + Yahoo Finance analysis powered by Claude. "
        "POST /api/v1/analyze/{ticker} runs the full pipeline: EDGAR → yfinance → "
        "sentiment analysis → Claude research brief."
    ),
    version="0.1.0",
    lifespan=lifespan,
)

app.include_router(router)
