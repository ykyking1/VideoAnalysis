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
    -- Nullable: kaynak veri setinde (örn. SeaDronesSee) uçuş telemetrisi yoksa NULL kalır.
    sensor_type     LowCardinality(String) DEFAULT 'unknown',
    avg_speed_kmh   Nullable(Float32),
    agl_m           Nullable(Float32),
    sun_elevation   Nullable(Float32),
    over_sea        Nullable(Bool),

    -- terfi etmiş görsel alanlar (§3.1 madde 4, kullanım verisine göre büyür)
    vehicle_count   UInt16 DEFAULT 0,

    -- ham telemetri özeti (öngörülmeyen gelecekteki filtreler için, §3.1 madde 2)
    lat             Nullable(Float64),
    lon             Nullable(Float64),
    heading_deg     Nullable(Float32),
    gimbal_pitch_deg Nullable(Float32),

    ingested_at     DateTime DEFAULT now()
)
ENGINE = MergeTree
ORDER BY (video_id, t_start);

-- HNSW vektör indeksi (ClickHouse 25.8+'da GA, deneysel flag gerekmiyor - canlı
-- instance'ta doğrulandı: 26.7.1). Boyut (512) şu an X-CLIP-base-patch32'ye
-- sabit - model değişirse (bkz. §5) bu satır güncellenmeli. Yeni INSERT edilen
-- part'lar otomatik index'lenir. MATERIALIZE sadece index eklenmeden önce
-- yazılmış satırları geriye dönük doldurmak için gerekir (bkz. scripts/init_schema.py).
ALTER TABLE clips ADD INDEX IF NOT EXISTS clips_embedding_idx embedding
    TYPE vector_similarity('hnsw', 'cosineDistance', 512);
