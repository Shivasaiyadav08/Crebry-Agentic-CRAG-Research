"""Backend settings. Environment variables override the root and backend dotenv files."""

from pathlib import Path
from pydantic import AliasChoices, Field

from pydantic_settings import BaseSettings, SettingsConfigDict

BACKEND_DIR = Path(__file__).resolve().parents[2]


class Config(BaseSettings):
    """Credentials, provider choices, and bounded resource budgets."""

    model_config = SettingsConfigDict(
        env_file=(BACKEND_DIR.parent / ".env", BACKEND_DIR / ".env"),
        env_ignore_empty=True,
        extra="ignore",
    )
    gemini_api_key: str = Field(default="", validation_alias=AliasChoices("GEMINI_API_KEY", "GOOGLE_API_KEY"))
    tavily_api_key: str = ""
    clerk_secret_key: str = ""
    clerk_jwt_key: str = ""
    clerk_issuer: str = ""
    clerk_authorized_parties: str = "http://localhost:3000,http://127.0.0.1:3000"
    gemini_model: str = "auto"
    gemini_project_id: str = "local"
    gemini_retry_rounds: int = Field(default=1, ge=0, le=1)
    gemini_max_concurrency: int = Field(default=2, ge=1, le=8)
    gemini_free_rpm: int = Field(default=5, ge=1, le=1000)
    gemini_free_rpd: int = Field(default=20, ge=1, le=10000)
    gemini_free_tpm: int = Field(default=125000, ge=8192, le=2000000)
    research_timeout_seconds: float = Field(default=240, ge=30, le=600)
    max_research_jobs: int = Field(default=2, ge=1, le=8)
    user_requests_per_minute: int = Field(default=120, ge=10, le=1000)
    database_url: str = ""
    vector_provider: str = Field(default="chroma", pattern=r"^(chroma|supabase)$")
    storage_provider: str = Field(default="auto", pattern=r"^(auto|local|r2|supabase)$")
    supabase_url: str = ""
    supabase_secret_key: str = Field(
        default="", validation_alias=AliasChoices("SUPABASE_SECRET_KEY", "SUPABASE_SERVICE_ROLE_KEY")
    )
    supabase_storage_bucket: str = "crebry-documents"
    environment: str = Field(default="development", pattern=r"^(development|test|production)$")
    persistent_data_root: Path | None = None
    gemini_free_project_confirmed: bool = False
    chroma_host: str = ""
    chroma_port: int = Field(default=8000, ge=1, le=65535)
    chroma_ssl: bool = True
    chroma_auth_token: str = ""
    gemini_max_model_attempts: int = Field(default=3, ge=1, le=8)
    gemini_operation_timeout_seconds: float = Field(default=75, ge=1, le=180)
    gemini_structured_max_output_tokens: int = Field(default=2048, ge=256, le=8192)
    data_dir: Path = Path(__file__).resolve().parents[2] / "data"
    frontend_origin: str = "http://localhost:3000"
    max_upload_mb: int = Field(default=10, ge=1, le=100)
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_batch_size: int = Field(default=32, ge=1, le=128)
    max_concurrent_embeddings: int = Field(default=1, ge=1, le=8)
    max_extracted_chars: int = Field(default=2_000_000, ge=10_000, le=10_000_000)
    enable_memory_diagnostics: bool = Field(default=True)
    gc_threshold_mb: float = Field(default=400.0, ge=100.0, le=2000.0)
    r2_account_id: str = ""
    r2_access_key_id: str = ""
    r2_secret_access_key: str = ""
    r2_bucket_name: str = ""
    # Auto uses Gemini when a key exists, otherwise optional local embeddings.
    embedding_provider: str = Field(default="auto", pattern=r"^(auto|local|gemini)$")
    gemini_ocr_enabled: bool = Field(default=True)
    gemini_ocr_model: str = "gemini-3.1-flash-lite"


config = Config()
