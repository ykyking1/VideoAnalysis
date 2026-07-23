-- Klip düzeyinde arama/analitik tablosu (proje-ozeti.md §3.1 madde 6).
-- Vektör boyutu (512 vs 768) ve embedding modeli §5'te kesinleşince güncellenmeli.

CREATE TABLE IF NOT EXISTS clips
(
    video_id        String,
    t_start         Float64,                 -- saniye
    t_end           Float64,                 -- saniye

    -- semantik arama
    embedding       Array(Float32),          -- boyut model seçimine bağlı, bkz. §5
    caption         String,                  -- Qwen2.5-VL seçici caption, bkz. §3.1 madde 5

    -- telemetriden türetilmiş deterministik alanlar (§3.1 madde 2)
    sensor_type     LowCardinality(String),
    avg_speed_kmh   Float32,
    agl_m           Float32,
    sun_elevation   Float32,
    over_sea        Bool,

    -- terfi etmiş görsel alanlar (§3.1 madde 4, kullanım verisine göre büyür)
    vehicle_count   UInt16,

    -- ham telemetri özeti (öngörülmeyen gelecekteki filtreler için, §3.1 madde 2)
    lat             Float64,
    lon             Float64,
    heading_deg     Float32,
    gimbal_pitch_deg Float32,

    ingested_at     DateTime DEFAULT now()
)
ENGINE = MergeTree
ORDER BY (video_id, t_start)
-- HNSW vektör indeksi: gerçek ClickHouse sürümü ve boyut kesinleşince parametreler
-- (M, ef_construction, distance function) Adım 0 ölçümüyle ayarlanmalı (bkz. §6 not).
;
-- ALTER TABLE clips ADD INDEX clips_embedding_idx embedding TYPE vector_similarity(...) GRANULARITY ...;
