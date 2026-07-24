"""Başarı oranı ölçümü: kendi caption'ına geri dönüş testi (self-retrieval).

Golden set (proje-ozeti.md §7) henüz yok - bu, etiketsiz bir proxy metrik:
her klip için moondream'in ürettiği caption'ı sorgu olarak kullanıp, embedding
modelinin doğru klibi (kendi caption'ının kaynağı) korpusun geri kalanı
arasından top-K'da bulup bulamadığını ölçer.

ZAYIF PROXY UYARISI: Bu, "model kendi ürettiği açıklamayı tanıyor mu" sorusunu
cevaplar - "kullanıcının doğal dil sorgusunu doğru anlıyor mu" sorusunu DEĞİL.
Gerçek golden set (§7: tam eşleşme/kısmi eşleşme/zor-negatif üçlü tasarım)
olmadan bu sayı model seçimi kararına (§5) temel oluşturmamalı, sadece
mekanik/kaba bir sağlık sinyali olarak kullanılmalı.

Kullanım: python scripts/eval_retrieval.py
"""
import clickhouse_connect

from common import config
from ingest.activities.clip_embedding import embed_text

TOP_K = 5


def _get_client():
    return clickhouse_connect.get_client(
        host=config.CLICKHOUSE_HOST, port=config.CLICKHOUSE_PORT,
        username=config.CLICKHOUSE_USER, password=config.CLICKHOUSE_PASSWORD,
        database=config.CLICKHOUSE_DB,
    )


def evaluate() -> None:
    client = _get_client()

    captioned = client.query(
        "SELECT video_id, t_start, t_end, caption FROM clips WHERE caption != ''"
    ).result_rows
    total_clips = client.query("SELECT count() FROM clips").result_rows[0][0]

    if not captioned:
        print("Değerlendirilecek caption bulunamadı (henüz hiç klip caption'lanmamış).")
        return

    hits_at_1 = 0
    hits_at_k = 0
    ranks = []

    for video_id, t_start, t_end, caption in captioned:
        query_embedding = embed_text(caption)
        result = client.query(
            """
            SELECT video_id, t_start, t_end,
                   cosineDistance(embedding, {query_embedding:Array(Float32)}) AS dist
            FROM clips
            ORDER BY dist ASC
            LIMIT {top_k:UInt32}
            """,
            parameters={"query_embedding": query_embedding, "top_k": max(TOP_K, total_clips)},
        ).result_rows

        rank = next(
            (i for i, row in enumerate(result, start=1)
             if row[0] == video_id and row[1] == t_start and row[2] == t_end),
            None,
        )
        if rank is not None:
            ranks.append(rank)
            if rank == 1:
                hits_at_1 += 1
            if rank <= TOP_K:
                hits_at_k += 1

    n = len(captioned)
    print("--- Başarı oranı (self-retrieval proxy) ---")
    print(f"Değerlendirilen caption sayısı: {n} / toplam klip: {total_clips}")
    print(f"Recall@1: {hits_at_1}/{n} ({100 * hits_at_1 / n:.1f}%)")
    print(f"Recall@{TOP_K}: {hits_at_k}/{n} ({100 * hits_at_k / n:.1f}%)")
    if ranks:
        print(f"Ortalama sıra: {sum(ranks) / len(ranks):.1f}")
    print("UYARI: Bu bir golden-set metriği değil, zayıf bir proxy'dir - bkz. dosya docstring'i.")


if __name__ == "__main__":
    evaluate()
