"""Centralized configuration loaded from environment variables (.env)."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    """Typed container for all environment-driven configuration values."""

    notion_api_key: str
    notion_root_page_id: str

    voyage_api_key: str
    voyage_model: str

    anthropic_api_key: str
    anthropic_model: str

    postgres_user: str
    postgres_password: str
    postgres_db: str
    postgres_host: str
    postgres_port: int

    top_k: int


def load_settings() -> Settings:
    """Load .env (via python-dotenv) and return a populated Settings instance.

    Raises:
        ValueError: if a required environment variable is missing.
    """
    raise NotImplementedError


def get_database_url(settings: Settings) -> str:
    """Build a psycopg-compatible connection string from Settings."""
    raise NotImplementedError
