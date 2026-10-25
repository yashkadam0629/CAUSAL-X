"""Environment-driven configuration. No secrets are embedded in source."""
from __future__ import annotations

import logging
import os
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]  # repository root


def _load_dotenv() -> None:
    env = ROOT / ".env"
    if not env.exists():
        return
    for line in env.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


_load_dotenv()


@dataclass(frozen=True)
class Settings:
    app_name: str = "CAUSAL-X"
    version: str = "0.1.0"
    database_url: str = os.environ.get("CAUSALX_DATABASE_URL", f"sqlite:///{(ROOT / 'data' / 'causalx.db').as_posix()}")
    data_dir: Path = Path(os.environ.get("CAUSALX_DATA_DIR", str(ROOT / "data")))
    artifacts_dir: Path = Path(os.environ.get("CAUSALX_ARTIFACTS_DIR", str(ROOT / "artifacts")))
    max_upload_mb: int = int(os.environ.get("CAUSALX_MAX_UPLOAD_MB", "50"))
    default_seed: int = int(os.environ.get("CAUSALX_SEED", "42"))
    default_n_customers: int = int(os.environ.get("CAUSALX_N_CUSTOMERS", "40000"))
    cors_origins: tuple = tuple(os.environ.get("CAUSALX_CORS_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173").split(","))
    log_level: str = os.environ.get("CAUSALX_LOG_LEVEL", "INFO")

    @property
    def synthetic_dir(self) -> Path:
        return self.data_dir / "synthetic"

    @property
    def upload_dir(self) -> Path:
        return self.data_dir / "uploads"

    @property
    def is_sqlite(self) -> bool:
        return self.database_url.startswith("sqlite")


settings = Settings()
for _p in (settings.synthetic_dir, settings.upload_dir, settings.artifacts_dir / "models",
           settings.artifacts_dir / "reports", settings.artifacts_dir / "experiment_results"):
    _p.mkdir(parents=True, exist_ok=True)

logging.basicConfig(level=settings.log_level, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("causalx")


@contextmanager
def timed(label: str):
    """Timing log for major analytical operations."""
    t0 = time.perf_counter()
    try:
        yield
    finally:
        log.info("timing %s: %.2fs", label, time.perf_counter() - t0)
