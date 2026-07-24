"""Hibrit arama: ClickHouse tek sorguda skip index'lerle filtre + küçültülmüş
kümede vektör karşılaştırması (proje-ozeti.md §3.2 madde 2).

HNSW (`vector_similarity`) indeksi artık kurulu (bkz. schema/clickhouse_clips.sql,
schema/clips_videoclip_xl.sql - ClickHouse 25.8+'da GA, deneysel flag gerekmiyor,
canlı instance'ta doğrulandı: 26.7.1). Sorgu şekli DEĞİŞMEDİ - `ORDER BY
cosineDistance(...) LIMIT k` deseni ClickHouse'un query planner'ı tarafından
otomatik olarak indeksten hızlandırılıyor (EXPLAIN ile doğrulandı, kod
tarafında ekstra bir şey yapmaya gerek yok).

ÖNEMLİ NÜANS: WHERE filtresi + vektör ORDER BY birlikte kullanıldığında
ClickHouse'un kendi `vector_search_filter_strategy` ayarı devreye giriyor -
'postfilter' (varsayılan) önce ANN indeksten en yakın komşuları bulup SONRA
filtreyi uyguluyor (LIMIT'ten az sonuç dönebilir), 'prefilter' önce filtreyi
uygulayıp SONRA brute-force arama yapıyor (ANN hızlanması kaybolur ama sonuç
tam). Şu an TELEMETRY_FILTERS_ENABLED=False olduğu için bu ayrım pratikte
devreye girmiyor; gerçek telemetri eklenip filtreler aktif olunca hangi
stratejinin daha iyi çalıştığı ölçülmeli, şimdilik ClickHouse varsayılanı
('auto'/postfilter) kullanılıyor.

sun_elevation eşikleri kaba yaklaşımlardır (gerçek astral hesaplama olmadığı
için): is_sunset ~ -6°..6°, is_night ~ < -6°.

TELEMETRY_FILTERS_ENABLED = False: kullanıcı isteğiyle (bkz. proje notu - "telemetri
kısmı yok, sadece embedding modelinin başarısına bakacağız") mevcut test korpusunda
avg_speed_kmh/agl_m/sun_elevation/over_sea hep NULL, vehicle_count hep 0 (YOLO
atlandı). LLM sorguda "deniz üzerinde"/"gece" gibi bir kavram gördüğünde bu alanlara
True/False atayabiliyor, ama filtre olarak uygulanırsa NULL/0 verilerle karşılaştığı
için (yön fark etmeksizin) sıfır sonuç dönüyor ve iyi bir semantik eşleşmeyi bile
gizliyor. Gerçek telemetri eklenince bu bayrak True yapılmalı.
"""
import clickhouse_connect

from common import config
from ingest.activities.clip_embedding import embed_text
from query.interval_merge import Match
from query.llm_parser import ParsedQuery

SUNSET_ELEVATION_RANGE = (-6.0, 6.0)
NIGHT_ELEVATION_MAX = -6.0
TELEMETRY_FILTERS_ENABLED = False


def _get_client():
    return clickhouse_connect.get_client(
        host=config.CLICKHOUSE_HOST, port=config.CLICKHOUSE_PORT,
        username=config.CLICKHOUSE_USER, password=config.CLICKHOUSE_PASSWORD,
        database=config.CLICKHOUSE_DB,
    )


def _build_where(filters) -> tuple[str, dict]:
    clauses = []
    params: dict = {}

    if filters.sensor_type is not None:
        clauses.append("sensor_type = {sensor_type:String}")
        params["sensor_type"] = filters.sensor_type

    if TELEMETRY_FILTERS_ENABLED:
        if filters.min_speed_kmh is not None:
            clauses.append("avg_speed_kmh >= {min_speed_kmh:Float32}")
            params["min_speed_kmh"] = filters.min_speed_kmh
        if filters.max_speed_kmh is not None:
            clauses.append("avg_speed_kmh <= {max_speed_kmh:Float32}")
            params["max_speed_kmh"] = filters.max_speed_kmh
        if filters.min_agl_m is not None:
            clauses.append("agl_m >= {min_agl_m:Float32}")
            params["min_agl_m"] = filters.min_agl_m
        if filters.max_agl_m is not None:
            clauses.append("agl_m <= {max_agl_m:Float32}")
            params["max_agl_m"] = filters.max_agl_m
        if filters.over_sea:
            clauses.append("over_sea = true")
        if filters.is_sunset:
            clauses.append("sun_elevation BETWEEN {sunset_lo:Float32} AND {sunset_hi:Float32}")
            params["sunset_lo"], params["sunset_hi"] = SUNSET_ELEVATION_RANGE
        if filters.is_night:
            clauses.append("sun_elevation < {night_max:Float32}")
            params["night_max"] = NIGHT_ELEVATION_MAX
        if filters.min_vehicle_count is not None:
            clauses.append("vehicle_count >= {min_vehicle_count:UInt16}")
            params["min_vehicle_count"] = filters.min_vehicle_count

    where = " AND ".join(clauses) if clauses else "1"
    return where, params


def search(parsed: ParsedQuery, top_k: int = 20) -> list[Match]:
    """ParsedQuery.filters'ı WHERE koşullarına, semantic_text'i embedding modeline
    çevirip ClickHouse `clips` tablosunda hibrit sorgu çalıştırır."""
    client = _get_client()
    where, params = _build_where(parsed.filters)

    if parsed.semantic_text.strip():
        query_embedding = embed_text(parsed.semantic_text)
        params["query_embedding"] = query_embedding
        order_by = "cosineDistance(embedding, {query_embedding:Array(Float32)}) ASC"
    else:
        order_by = "t_start ASC"

    result = client.query(
        f"""
        SELECT video_id, t_start, t_end
        FROM clips
        WHERE {where}
        ORDER BY {order_by}
        LIMIT {{top_k:UInt32}}
        """,
        parameters={**params, "top_k": top_k},
    )
    return [Match(video_id=row[0], t_start=row[1], t_end=row[2]) for row in result.result_rows]
