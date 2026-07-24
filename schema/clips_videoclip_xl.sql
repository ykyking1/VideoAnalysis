-- Model karşılaştırması için: VideoCLIP-XL (768d) embedding'leri, aynı
-- pencereler/caption'lar üzerinde, ana `clips` tablosundan (X-CLIP, 512d) ayrı
-- tutuluyor (proje-ozeti.md §5 model seçimi karşılaştırması).

CREATE TABLE IF NOT EXISTS clips_videoclip_xl
(
    video_id    String,
    t_start     Float64,
    t_end       Float64,
    embedding   Array(Float32),  -- 768d, ViT-L/14
    caption     String
)
ENGINE = MergeTree
ORDER BY (video_id, t_start);

-- Bkz. schema/clickhouse_clips.sql'deki HNSW index yorumu - aynı mantık, 768d.
ALTER TABLE clips_videoclip_xl ADD INDEX IF NOT EXISTS clips_videoclip_xl_embedding_idx embedding
    TYPE vector_similarity('hnsw', 'cosineDistance', 768);
