-- Video/görev kimlik kayıtları ve ingest durum takibi (proje-ozeti.md §2).
-- Durum makinesi geçişleri Temporal workflow tarafından güncellenir.

CREATE TYPE ingest_status AS ENUM (
    'discovered',       -- MinIO bucket notification alındı
    'proxy_pending',
    'proxy_done',
    'telemetry_pending',
    'telemetry_done',
    'embedding_pending',
    'embedding_done',
    'caption_pending',
    'caption_done',
    'written',          -- ClickHouse'a yazıldı
    'failed'
);

CREATE TABLE IF NOT EXISTS videos
(
    video_id        TEXT PRIMARY KEY,
    source_path     TEXT NOT NULL,           -- MinIO nesne yolu (ham video)
    proxy_path      TEXT,                    -- MinIO nesne yolu (model proxy), kalıcı tutulmuyorsa NULL
    duration_s      DOUBLE PRECISION,
    telemetry_path  TEXT,                    -- .tlog / MAVLink log yolu, varsa
    status          ingest_status NOT NULL DEFAULT 'discovered',
    error_message   TEXT,
    discovered_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS videos_status_idx ON videos (status);
