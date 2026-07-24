"""Model karşılaştırması: mevcut korpustaki (95 klip, 6 video) pencere/caption
verisini yeniden kullanarak VideoCLIP-XL ile embedding üretir, ClickHouse'a
`clips_videoclip_xl` tablosuna yazar, sonra aynı self-retrieval metoduyla
(scripts/eval_retrieval.py mantığı) X-CLIP-base-patch32 ile yan yana karşılaştırır.

xuguohai/X-CLIP (proje-ozeti.md §5'in asıl lider adayı) karşılaştırmaya dahil
DEĞİL - GitHub deposu hazır bir retrieval checkpoint'i dağıtmıyor, sadece ham
CLIP ağırlıkları + eğitim script'leri var; gerçek modeli kullanmak MSR-VTT gibi
bir veri setinde sıfırdan eğitim gerektiriyor (GT1030'da günler sürecek,
gerçekçi değil). Bu, projenin kendi §5 notunun ötesinde yeni bir bulgu.

Kullanım: python poc/compare_embedding_models.py
"""
import time

import clickhouse_connect

from common import config
from common.minio_client import download_temp
from ingest.activities.embedding_videoclip_xl import embed_text as vxl_embed_text
from ingest.activities.embedding_videoclip_xl import embed_window as vxl_embed_window
from ingest.activities.embedding_videoclip_xl import unload_model as vxl_unload

VIDEOCLIP_XL_SCHEMA = """
CREATE TABLE IF NOT EXISTS clips_videoclip_xl
(
    video_id    String,
    t_start     Float64,
    t_end       Float64,
    embedding   Array(Float32),
    caption     String
)
ENGINE = MergeTree
ORDER BY (video_id, t_start)
"""


def _get_client():
    return clickhouse_connect.get_client(
        host=config.CLICKHOUSE_HOST, port=config.CLICKHOUSE_PORT,
        username=config.CLICKHOUSE_USER, password=config.CLICKHOUSE_PASSWORD,
        database=config.CLICKHOUSE_DB,
    )


def embed_corpus_with_videoclip_xl() -> tuple[int, float]:
    """Mevcut `clips` tablosundaki tüm pencereleri VideoCLIP-XL ile yeniden
    embed edip `clips_videoclip_xl`'e yazar. (video_id, proxy MinIO'dan bir kez
    indirilir, o videonun tüm pencereleri için tekrar kullanılır.)"""
    client = _get_client()
    client.command(VIDEOCLIP_XL_SCHEMA)
    client.command("TRUNCATE TABLE clips_videoclip_xl")

    rows = client.query(
        "SELECT video_id, t_start, t_end, caption FROM clips ORDER BY video_id, t_start"
    ).result_rows

    videos: dict[str, list] = {}
    for video_id, t_start, t_end, caption in rows:
        videos.setdefault(video_id, []).append((t_start, t_end, caption))

    total_windows = 0
    t0 = time.perf_counter()
    for video_id, windows in videos.items():
        proxy_path = f"{video_id}/proxy.mp4"
        print(f"[{video_id}] {len(windows)} pencere embed ediliyor (VideoCLIP-XL)...")
        with download_temp(config.MINIO_BUCKET_PROXY, proxy_path) as local_path:
            out_rows = []
            for t_start, t_end, caption in windows:
                embedding = vxl_embed_window(local_path, t_start, t_end)
                out_rows.append([video_id, t_start, t_end, embedding, caption])
            client.insert(
                "clips_videoclip_xl", out_rows,
                column_names=["video_id", "t_start", "t_end", "embedding", "caption"],
            )
        total_windows += len(windows)
    elapsed = time.perf_counter() - t0
    vxl_unload()
    return total_windows, elapsed


def eval_self_retrieval(table: str, embed_text_fn, top_k: int = 5) -> dict:
    client = _get_client()
    captioned = client.query(
        f"SELECT video_id, t_start, t_end, caption FROM {table} WHERE caption != ''"
    ).result_rows
    total_clips = client.query(f"SELECT count() FROM {table}").result_rows[0][0]

    if not captioned:
        return {"n": 0, "total": total_clips, "recall_1": 0.0, "recall_k": 0.0, "avg_rank": None}

    hits_1, hits_k, ranks = 0, 0, []
    for video_id, t_start, t_end, caption in captioned:
        query_embedding = embed_text_fn(caption)
        result = client.query(
            f"""
            SELECT video_id, t_start, t_end,
                   cosineDistance(embedding, {{qe:Array(Float32)}}) AS dist
            FROM {table}
            ORDER BY dist ASC
            LIMIT {{top_k:UInt32}}
            """,
            parameters={"qe": query_embedding, "top_k": max(top_k, total_clips)},
        ).result_rows
        rank = next(
            (i for i, row in enumerate(result, start=1)
             if row[0] == video_id and row[1] == t_start and row[2] == t_end),
            None,
        )
        if rank is not None:
            ranks.append(rank)
            hits_1 += rank == 1
            hits_k += rank <= top_k

    n = len(captioned)
    return {
        "n": n, "total": total_clips,
        "recall_1": hits_1 / n, "recall_k": hits_k / n,
        "avg_rank": sum(ranks) / len(ranks) if ranks else None,
    }


def storage_stats(table: str) -> dict:
    client = _get_client()
    row = client.query(
        """
        SELECT sum(rows), sum(bytes_on_disk)
        FROM system.parts
        WHERE database = {db:String} AND table = {table:String} AND active
        """,
        parameters={"db": config.CLICKHOUSE_DB, "table": table},
    ).result_rows[0]
    total_rows, total_bytes = row
    return {
        "rows": total_rows or 0,
        "bytes_per_row": (total_bytes / total_rows) if total_rows else 0,
    }


def main() -> None:
    from ingest.activities.clip_embedding import embed_text as xclip_embed_text

    n_windows, elapsed = embed_corpus_with_videoclip_xl()
    realtime_note = f"{elapsed:.1f}s / {n_windows} pencere ({elapsed / n_windows:.2f}s/pencere)"

    xclip_eval = eval_self_retrieval("clips", xclip_embed_text)
    vxl_eval = eval_self_retrieval("clips_videoclip_xl", vxl_embed_text)

    xclip_storage = storage_stats("clips")
    vxl_storage = storage_stats("clips_videoclip_xl")

    print("\n=== Model Karşılaştırması ===")
    print(f"VideoCLIP-XL embedding süresi: {realtime_note}\n")

    print(f"{'Metrik':<20}{'X-CLIP-base-patch32':<25}{'VideoCLIP-XL':<25}")
    print(f"{'Boyut':<20}{'512d':<25}{'768d':<25}")
    print(f"{'Byte/klip (disk)':<20}{xclip_storage['bytes_per_row']:<25.0f}{vxl_storage['bytes_per_row']:<25.0f}")
    print(f"{'Recall@1':<20}{xclip_eval['recall_1']*100:<24.1f}%{vxl_eval['recall_1']*100:<24.1f}%")
    print(f"{'Recall@5':<20}{xclip_eval['recall_k']*100:<24.1f}%{vxl_eval['recall_k']*100:<24.1f}%")
    print(f"{'Ort. sıra':<20}{xclip_eval['avg_rank']:<25.1f}{vxl_eval['avg_rank']:<25.1f}")
    print(f"{'N (caption sayısı)':<20}{xclip_eval['n']:<25}{vxl_eval['n']:<25}")
    print("\nUYARI: Golden-set değil, self-retrieval proxy metriği (bkz. scripts/eval_retrieval.py docstring).")
    print("xuguohai/X-CLIP karşılaştırmaya dahil değil: hazır retrieval checkpoint'i yok (bkz. dosya docstring'i).")


if __name__ == "__main__":
    main()
