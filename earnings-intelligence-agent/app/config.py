"""Application configuration loaded from environment variables."""

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

# Resolve the project root from this file's location so the .env path is
# absolute — pydantic-settings resolves relative paths from CWD, which
# breaks when uvicorn is launched from outside the project root.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    anthropic_api_key: str
    sec_edgar_user_agent: str = "EarningsIntelligenceAgent contact@example.com"

    model_config = SettingsConfigDict(
        env_file=str(_PROJECT_ROOT / ".env"),
        env_file_encoding="utf-8",
    )


settings = Settings()
