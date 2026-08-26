from dataclasses import dataclass
import os
from pathlib import Path

try:
    from dotenv import load_dotenv
except ModuleNotFoundError:  # Keeps the health endpoint usable before backend dependencies are installed.
    def load_dotenv(path: Path) -> None:
        if not path.exists():
            return
        for raw_line in path.read_text().splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


load_dotenv(Path(__file__).resolve().parents[1] / ".env")


@dataclass(frozen=True)
class Settings:
    mongodb_uri: str | None = os.getenv("MONGODB_URI")
    mongodb_database: str = os.getenv("DB_NAME", "ilumos")
    mongodb_gridfs_bucket: str = os.getenv("MONGODB_GRIDFS_BUCKET", "ilumos_files")
    app_origin: str = os.getenv("APP_ORIGIN", "http://localhost:3006")
    # The backend calls Ollama Cloud directly; no local Ollama daemon is required.
    ollama_base_url: str = os.getenv("OLLAMA_BASE_URL", "https://ollama.com/v1")
    ollama_model: str = os.getenv("OLLAMA_MODEL", "nemotron-3-nano:30b-cloud")
    ollama_api_key: str | None = os.getenv("OLLAMA_API_KEY")
    ollama_timeout_seconds: float = float(os.getenv("OLLAMA_TIMEOUT_SECONDS", "300"))
    ollama_max_input_chars: int = int(os.getenv("OLLAMA_MAX_INPUT_CHARS", "400000"))

    @property
    def database_configured(self) -> bool:
        return bool(self.mongodb_uri)
