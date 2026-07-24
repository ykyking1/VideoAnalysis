"""Ortam değişkenlerinden okunan bağlantı ayarları (bkz. .env.example).

Yerel Docker Compose stack'i (docker-compose.yml) ile eşleşir.
"""
import os


def _env(name: str, default: str) -> str:
    return os.environ.get(name, default)


MINIO_ENDPOINT = _env("MINIO_ENDPOINT", "localhost:9000")
MINIO_ROOT_USER = _env("MINIO_ROOT_USER", "minioadmin")
MINIO_ROOT_PASSWORD = _env("MINIO_ROOT_PASSWORD", "minioadmin123")
MINIO_BUCKET_RAW = _env("MINIO_BUCKET_RAW", "raw-videos")
MINIO_BUCKET_PROXY = _env("MINIO_BUCKET_PROXY", "proxy-videos")

POSTGRES_HOST = _env("POSTGRES_HOST", "localhost")
POSTGRES_PORT = int(_env("POSTGRES_PORT", "5432"))
POSTGRES_USER = _env("POSTGRES_USER", "app")
POSTGRES_PASSWORD = _env("POSTGRES_PASSWORD", "app")
POSTGRES_DB = _env("POSTGRES_DB", "videoanalysis")

CLICKHOUSE_HOST = _env("CLICKHOUSE_HOST", "localhost")
CLICKHOUSE_PORT = int(_env("CLICKHOUSE_PORT", "8123"))
CLICKHOUSE_USER = _env("CLICKHOUSE_USER", "app")
CLICKHOUSE_PASSWORD = _env("CLICKHOUSE_PASSWORD", "app")
CLICKHOUSE_DB = _env("CLICKHOUSE_DB", "videoanalysis")

OLLAMA_HOST = _env("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_PARSE_MODEL = _env("OLLAMA_PARSE_MODEL", "qwen2.5:3b")
OLLAMA_CAPTION_MODEL = _env("OLLAMA_CAPTION_MODEL", "moondream")


def postgres_dsn() -> str:
    return (
        f"host={POSTGRES_HOST} port={POSTGRES_PORT} "
        f"user={POSTGRES_USER} password={POSTGRES_PASSWORD} dbname={POSTGRES_DB}"
    )
