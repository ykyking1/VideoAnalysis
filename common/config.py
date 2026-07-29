"""Ortam değişkenlerinden okunan yapılandırma (bkz. .env.example).

Yerel Docker Compose stack'i (docker-compose.yml) ile eşleşir. Tüm değerler
env ile geçersiz kılınabilir - kod içinde sabit bağlantı bilgisi yok.
"""
import os


def _env(name: str, default: str) -> str:
    return os.environ.get(name, default)


def _env_int(name: str, default: int) -> int:
    return int(os.environ.get(name, str(default)))


def _env_float(name: str, default: float) -> float:
    return float(os.environ.get(name, str(default)))


def _env_bool(name: str, default: bool) -> bool:
    return os.environ.get(name, str(default)).strip().lower() in ("1", "true", "yes", "on")


# --- Nesne deposu (ham video + model proxy'leri) ---
MINIO_ENDPOINT = _env("MINIO_ENDPOINT", "localhost:9000")
MINIO_ROOT_USER = _env("MINIO_ROOT_USER", "minioadmin")
MINIO_ROOT_PASSWORD = _env("MINIO_ROOT_PASSWORD", "minioadmin123")
MINIO_SECURE = _env_bool("MINIO_SECURE", False)
MINIO_BUCKET_RAW = _env("MINIO_BUCKET_RAW", "raw-videos")
MINIO_BUCKET_PROXY = _env("MINIO_BUCKET_PROXY", "proxy-videos")

# --- Durum takibi (Temporal workflow state machine, proje-ozeti.md §2) ---
# NOT: Postgres SADECE ingest durum takibi icin. Telemetri/metadata arama
# tarafinda Qdrant payload'inda tutuluyor - iki sisteme cift yazip senkron
# riski almamak icin bilincli bir karar (bkz. docs/worklog).
POSTGRES_HOST = _env("POSTGRES_HOST", "localhost")
POSTGRES_PORT = _env_int("POSTGRES_PORT", 5432)
POSTGRES_USER = _env("POSTGRES_USER", "app")
POSTGRES_PASSWORD = _env("POSTGRES_PASSWORD", "app")
POSTGRES_DB = _env("POSTGRES_DB", "videoanalysis")

# --- Vektör deposu (arama katmanı) ---
# Docker'siz (gomulu) mod: sunucu yerine yerel bir dizin kullanilir.
# Colab/Kaggle gibi Docker calistirilamayan ortamlar icin.
# DIKKAT: gomulu mod qdrant-client'in saf Python implementasyonudur, gercek
# Rust HNSW motorunu KULLANMAZ - tam (exact) arama yapar. Islevsel testler
# gecerli, PERFORMANS OLCUMLERI GECERSIZ. Uretimde asla kullanmayin.
QDRANT_LOCAL_PATH = os.environ.get("QDRANT_LOCAL_PATH") or None
QDRANT_HOST = _env("QDRANT_HOST", "localhost")
QDRANT_PORT = _env_int("QDRANT_PORT", 6333)
QDRANT_GRPC_PORT = _env_int("QDRANT_GRPC_PORT", 6334)
QDRANT_PREFER_GRPC = _env_bool("QDRANT_PREFER_GRPC", True)
QDRANT_API_KEY = os.environ.get("QDRANT_API_KEY") or None
QDRANT_TIMEOUT_S = _env_int("QDRANT_TIMEOUT_S", 120)
QDRANT_COLLECTION = _env("QDRANT_COLLECTION", "clips")

# Uretim olceginde (bkz. proje-ozeti.md §8: ~300.000 video x 3-5sa -> ~540M
# pencere) vektorler RAM'e sigmaz. on_disk=True vektorleri diskte tutar, HNSW
# grafi RAM'de kalir. Scalar (int8) kuantizasyon 4x kucultup always_ram ile
# grafi hizli tutar; orijinaller diskte kalip rescoring'de kullanilir.
QDRANT_ON_DISK = _env_bool("QDRANT_ON_DISK", True)
QDRANT_QUANTIZATION = _env("QDRANT_QUANTIZATION", "scalar")  # "none" | "scalar"
QDRANT_HNSW_M = _env_int("QDRANT_HNSW_M", 16)
QDRANT_HNSW_EF_CONSTRUCT = _env_int("QDRANT_HNSW_EF_CONSTRUCT", 128)
QDRANT_SEARCH_HNSW_EF = _env_int("QDRANT_SEARCH_HNSW_EF", 128)

# --- Embedding modeli (proje-ozeti.md §5) ---
# Qwen3-VL-Embedding-2B: Apache-2.0. VideoCLIP-XL ve EBind CC-BY-NC-SA
# (ticari/savunma kullanimina kapali) oldugu icin elendi - bkz. docs/worklog.
EMBEDDING_MODEL = _env("EMBEDDING_MODEL", "Qwen/Qwen3-VL-Embedding-2B")
EMBEDDING_MODEL_DIR = os.environ.get("EMBEDDING_MODEL_DIR") or None  # yerel kopya (opsiyonel)
EMBEDDING_DIM = _env_int("EMBEDDING_DIM", 2048)  # MRL ile 64-2048 arasi kisaltilabilir
EMBEDDING_NUM_FRAMES = _env_int("EMBEDDING_NUM_FRAMES", 8)
EMBEDDING_BATCH_SIZE = _env_int("EMBEDDING_BATCH_SIZE", 8)
EMBEDDING_DTYPE = _env("EMBEDDING_DTYPE", "auto")  # auto | float16 | bfloat16 | float32

# --- Pencereleme (proje-ozeti.md §3.1 madde 2) ---
# 8sn/8sn ortusmesiz: gercek envanterde (§8) %50 ortusmeli 4sn kaydirma ~1
# milyar vektor demekti; kaydirmayi pencereye esitlemek bunu yariya indiriyor.
WINDOW_S = _env_float("WINDOW_S", 8.0)
STRIDE_S = _env_float("STRIDE_S", 8.0)

# --- Proxy üretimi (proje-ozeti.md §3.1 madde 1, §4) ---
PROXY_HEIGHT = _env_int("PROXY_HEIGHT", 360)
PROXY_BITRATE = _env("PROXY_BITRATE", "800k")
PROXY_USE_NVDEC = _env_bool("PROXY_USE_NVDEC", True)
PROXY_USE_NVENC = _env_bool("PROXY_USE_NVENC", True)

# --- Görsel alanlar (proje-ozeti.md §3.1 madde 4) ---
YOLO_MODEL = _env("YOLO_MODEL", "yolo26n.pt")
YOLO_SAMPLE_FRAMES = _env_int("YOLO_SAMPLE_FRAMES", 3)
YOLO_CONF = _env_float("YOLO_CONF", 0.25)

# --- vLLM (sorgu ayrıştırma + caption + rerank, proje-ozeti.md §3.1/§3.2) ---
# OpenAI-uyumlu sunucu. Ayni sunucuda iki ayri model calistirilabilir ya da
# VLM_BASE_URL farkli bir porta yonlendirilebilir.
VLLM_BASE_URL = _env("VLLM_BASE_URL", "http://localhost:8000/v1")
VLLM_API_KEY = _env("VLLM_API_KEY", "EMPTY")
# Uretim hedefi Qwen 14B (§3.2); 4060 sinifi tek GPU'da 7B-AWQ pratik siniridir.
PARSE_MODEL = _env("PARSE_MODEL", "Qwen/Qwen2.5-7B-Instruct-AWQ")
VLM_BASE_URL = _env("VLM_BASE_URL", VLLM_BASE_URL)
VLM_MODEL = _env("VLM_MODEL", "Qwen/Qwen2.5-VL-7B-Instruct-AWQ")
LLM_TIMEOUT_S = _env_int("LLM_TIMEOUT_S", 120)

# --- Caption (proje-ozeti.md §3.1 madde 5) ---
CAPTION_ENABLED = _env_bool("CAPTION_ENABLED", True)
CAPTION_WINDOW_FRACTION = _env_float("CAPTION_WINDOW_FRACTION", 0.10)

# --- Sorgu hattı ---
SEARCH_TOP_K = _env_int("SEARCH_TOP_K", 20)
# Hard filtre sonucu bu esigin altina duserse filtre otomatik gevsetilir
# (bkz. query/filter_relaxation.py). Olctugumuz gercek risk: dar bir filtre
# dogru cevabi yapisal olarak disliyor - bkz. docs/worklog.
SEARCH_MIN_RESULTS = _env_int("SEARCH_MIN_RESULTS", 5)
INTERVAL_GAP_TOLERANCE_S = _env_float("INTERVAL_GAP_TOLERANCE_S", 10.0)
RERANK_ENABLED = _env_bool("RERANK_ENABLED", False)
RERANK_CANDIDATES = _env_int("RERANK_CANDIDATES", 10)

# --- Kafka (ingest tetikleyici, proje-ozeti.md §3.1) ---
KAFKA_BOOTSTRAP_SERVERS = _env("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
KAFKA_TOPIC = _env("KAFKA_TOPIC", "video-ingest")
KAFKA_GROUP_ID = _env("KAFKA_GROUP_ID", "video-ingest-workers")

# --- Temporal (orkestrasyon) ---
TEMPORAL_HOST = _env("TEMPORAL_HOST", "localhost:7233")
TEMPORAL_NAMESPACE = _env("TEMPORAL_NAMESPACE", "default")
TEMPORAL_TASK_QUEUE = _env("TEMPORAL_TASK_QUEUE", "video-ingest")

# --- Telemetri (proje-ozeti.md §3.1 madde 2) ---
# Kiyi poligonu (Shapely/GeoPandas) - over_sea hesabi icin. Verilmezse
# over_sea None kalir (filtre bu alani atlar).
COASTLINE_GEOJSON = os.environ.get("COASTLINE_GEOJSON") or None
SUNSET_ELEVATION_RANGE = (-6.0, 6.0)
NIGHT_ELEVATION_MAX = -6.0


def postgres_dsn() -> str:
    return (
        f"host={POSTGRES_HOST} port={POSTGRES_PORT} "
        f"user={POSTGRES_USER} password={POSTGRES_PASSWORD} dbname={POSTGRES_DB}"
    )
