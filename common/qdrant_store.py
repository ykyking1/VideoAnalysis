"""Qdrant erişim katmanı: koleksiyon kurulumu, klip yazımı, filtreli arama.

NEDEN QDRANT (ClickHouse yerine): ClickHouse'ta `WHERE` + vektör `ORDER BY`
birlikte kullanıldığında `vector_search_filter_strategy` ikilemi doğuyor -
`prefilter` HNSW indeksini tamamen devre dışı bırakıp brute-force'a düşüyor
(`EXPLAIN indexes=1` ile doğrulandı: granül budama 72/72, yani hiç budama yok),
`postfilter` ise hızlı ama filtreyi aramadan SONRA uyguladığı için LIMIT'ten az
sonuç dönebiliyor. Qdrant filtreyi HNSW graf gezinmesinin İÇİNDE uyguluyor
(payload index sayesinde), yani tek bir yolda hem hızlı hem tam sonuç veriyor -
ölçümle doğrulandı (100K korpusta `exact=True` ile birebir aynı top-3, ~1.5x
daha hızlı). Detay: docs/worklog_2026-07-28.md.

Nokta kimliği `video_id + t_start`'tan türetilen deterministik UUID5 - aynı
pencere yeniden ingest edilirse yeni satır değil güncelleme olur (idempotent
backfill).
"""
import atexit
import uuid
from dataclasses import asdict, dataclass, field

from qdrant_client import QdrantClient
from qdrant_client.http import models as qm

from common import config

_NAMESPACE = uuid.UUID("6f9619ff-8b86-d011-b42d-00c04fc964ff")

# Filtrelenebilir alanlar -> Qdrant payload index tipi. Bu alanların indeksi
# olmadan Qdrant filtreyi HNSW gezinmesi sırasında uygulayamaz (tam tarama
# yapar), yani koleksiyonun tüm hız avantajı bu tabloya bağlı.
PAYLOAD_INDEXES: dict[str, qm.PayloadSchemaType] = {
    "video_id": qm.PayloadSchemaType.KEYWORD,
    "sensor_type": qm.PayloadSchemaType.KEYWORD,
    "over_sea": qm.PayloadSchemaType.BOOL,
    "avg_speed_kmh": qm.PayloadSchemaType.FLOAT,
    "agl_m": qm.PayloadSchemaType.FLOAT,
    "sun_elevation": qm.PayloadSchemaType.FLOAT,
    "vehicle_count": qm.PayloadSchemaType.INTEGER,
    "t_start": qm.PayloadSchemaType.FLOAT,
}


@dataclass
class ClipPayload:
    """ClickHouse `clips` satırının Qdrant payload karşılığı
    (proje-ozeti.md §3.1 madde 6)."""
    video_id: str
    t_start: float
    t_end: float
    sensor_type: str = "unknown"
    avg_speed_kmh: float | None = None
    agl_m: float | None = None
    sun_elevation: float | None = None
    over_sea: bool | None = None
    vehicle_count: int = 0
    caption: str = ""
    # Ham telemetri - öngörülmeyen gelecekteki filtreler için sigorta (§3.1)
    lat: float | None = None
    lon: float | None = None
    heading_deg: float | None = None
    gimbal_pitch_deg: float | None = None

    def to_dict(self) -> dict:
        # None degerleri ELEMIYORUZ: Qdrant'ta "alan yok" ile "alan null" farkli
        # davraniyor (IsNull kosulu sadece acikca null olanlari yakalar). Bilinmeyen
        # telemetriyi acikca null tutmak, sonradan "telemetrisi olmayan klipleri
        # bul" gibi sorgulari mumkun kiliyor.
        return asdict(self)


def point_id(video_id: str, t_start: float) -> str:
    """Aynı (video, pencere) için her zaman aynı kimlik - yeniden ingest
    idempotent olsun diye."""
    return str(uuid.uuid5(_NAMESPACE, f"{video_id}:{t_start:.3f}"))


_client: QdrantClient | None = None
_client_key: tuple | None = None


def _current_key() -> tuple:
    return (config.QDRANT_LOCAL_PATH, config.QDRANT_HOST, config.QDRANT_PORT,
            config.QDRANT_GRPC_PORT, config.QDRANT_PREFER_GRPC)


def get_client() -> QdrantClient:
    """QDRANT_LOCAL_PATH tanimliysa Docker'siz gomulu mod, aksi halde sunucu.

    ISTEMCI ONBELLEKLENIR. Gomulu mod bir dosya kilidi kullaniyor: ayni
    dizine ikinci bir istemci acmak "Storage folder is already accessed by
    another instance" hatasi veriyor (olculdu). Pipeline icinde write_clips
    ve search ayri ayri istemci istedigi icin, onbellek olmadan tek surecte
    (notebook, --local ingest) cakisma kaciniLmaz olurdu.

    Onbellek yapilandirmaya gore anahtarlanir - config reload edilip yol
    degisirse yeni istemci acilir (notebook'ta ortam degiskeni degistirmek
    bu yuzden guvenli).

    Gomulu mod (Colab/Kaggle icin) qdrant-client'in saf Python
    implementasyonudur: gercek Rust HNSW motorunu KULLANMAZ, tam (exact)
    arama yapar. Islevsel testler gecerli - hatta Recall daha yuksek cikar
    cunku yaklasiklik yok - ama GECIKME OLCUMLERI ANLAMSIZDIR ve buyuk
    korpusta kullanilamaz."""
    global _client, _client_key

    key = _current_key()
    if _client is not None and _client_key == key:
        return _client

    if _client is not None:
        try:
            _client.close()
        except Exception:  # noqa: BLE001 - kapanis hatasi yeni istemciyi engellemesin
            pass

    if config.QDRANT_LOCAL_PATH:
        _client = QdrantClient(path=config.QDRANT_LOCAL_PATH)
    else:
        _client = QdrantClient(
            host=config.QDRANT_HOST,
            port=config.QDRANT_PORT,
            grpc_port=config.QDRANT_GRPC_PORT,
            prefer_grpc=config.QDRANT_PREFER_GRPC,
            api_key=config.QDRANT_API_KEY,
            timeout=config.QDRANT_TIMEOUT_S,
        )
    _client_key = key
    return _client


def close_client() -> None:
    """Onbellekteki istemciyi kapatir (gomulu modda dosya kilidini birakir)."""
    global _client, _client_key
    if _client is not None:
        try:
            _client.close()
        except Exception:  # noqa: BLE001
            pass
        finally:
            _client, _client_key = None, None


# Yorumlayici kapanirken QdrantClient.__del__ modul tablosu bosaltilmis
# oldugu icin "ImportError: sys.meta_path is None" basiyordu - islevsel bir
# sorun degil ama korkutucu gorunuyor. atexit ile daha erken, duzgun kapatiyoruz.
atexit.register(close_client)


def _quantization_config():
    if config.QDRANT_QUANTIZATION != "scalar":
        return None
    # int8 skaler kuantizasyon: vektorleri 4x kucultur, always_ram=True ile
    # kuantize kopya RAM'de kalir (hizli tarama), orijinaller diskte kalip
    # rescoring'de kullanilir - 2048d x ~540M vektorde RAM'e sigmanin tek yolu.
    return qm.ScalarQuantization(
        scalar=qm.ScalarQuantizationConfig(type=qm.ScalarType.INT8, always_ram=True)
    )


def ensure_collection(client: QdrantClient, collection: str | None = None,
                       recreate: bool = False) -> str:
    """Koleksiyonu ve tüm payload indekslerini oluşturur (varsa dokunmaz)."""
    collection = collection or config.QDRANT_COLLECTION

    if recreate and client.collection_exists(collection):
        client.delete_collection(collection)

    if not client.collection_exists(collection):
        if config.QDRANT_LOCAL_PATH:
            # Gomulu mod: on_disk/HNSW/kuantizasyon sunucu ozellikleri,
            # saf Python implementasyonunda karsiligi yok.
            client.create_collection(
                collection_name=collection,
                vectors_config=qm.VectorParams(
                    size=config.EMBEDDING_DIM, distance=qm.Distance.COSINE),
            )
        else:
            client.create_collection(
                collection_name=collection,
                vectors_config=qm.VectorParams(
                    size=config.EMBEDDING_DIM,
                    distance=qm.Distance.COSINE,
                    on_disk=config.QDRANT_ON_DISK,
                ),
                hnsw_config=qm.HnswConfigDiff(
                    m=config.QDRANT_HNSW_M,
                    ef_construct=config.QDRANT_HNSW_EF_CONSTRUCT,
                ),
                quantization_config=_quantization_config(),
            )

    existing = client.get_collection(collection).payload_schema or {}
    for field_name, schema in PAYLOAD_INDEXES.items():
        if field_name not in existing:
            client.create_payload_index(
                collection_name=collection, field_name=field_name, field_schema=schema
            )
    # caption tam-metin arama (proje-ozeti.md §3.1 madde 5: hibrit vektor+tam metin)
    if "caption" not in existing:
        client.create_payload_index(
            collection_name=collection,
            field_name="caption",
            field_schema=qm.TextIndexParams(
                type=qm.TextIndexType.TEXT, tokenizer=qm.TokenizerType.WORD, lowercase=True
            ),
        )
    return collection


def upsert_clips(client: QdrantClient, collection: str,
                  embeddings: list[list[float]], payloads: list[ClipPayload],
                  chunk_size: int = 256) -> int:
    """Klipleri parça parça yazar.

    chunk_size: tek bir gRPC isteğine sığdırılan nokta sayısı. 2048d float32'de
    bir nokta ~8KB; 256'lık parça ~2MB istek demek. Daha büyük parçalar (10K)
    gerçek çalıştırmada bağlantı kopmasına yol açtı (WinError 10053).
    """
    if len(embeddings) != len(payloads):
        raise ValueError(f"embedding sayisi ({len(embeddings)}) payload sayisiyla "
                         f"({len(payloads)}) eslesmiyor")
    if not embeddings:
        return 0

    points = [
        qm.PointStruct(
            id=point_id(p.video_id, p.t_start),
            vector=emb,
            payload=p.to_dict(),
        )
        for emb, p in zip(embeddings, payloads)
    ]
    for start in range(0, len(points), chunk_size):
        client.upsert(
            collection_name=collection,
            points=points[start:start + chunk_size],
            wait=True,
        )
    return len(points)


def search(client: QdrantClient, collection: str, query_vector: list[float],
            query_filter: qm.Filter | None, top_k: int) -> list[qm.ScoredPoint]:
    """Filtreli vektör araması. Filtre HNSW gezinmesi sırasında uygulanır -
    ayrı bir prefilter/postfilter seçimi yok."""
    result = client.query_points(
        collection_name=collection,
        query=query_vector,
        query_filter=query_filter,
        limit=top_k,
        with_payload=True,
        search_params=qm.SearchParams(hnsw_ef=config.QDRANT_SEARCH_HNSW_EF),
    )
    return result.points


def delete_video(client: QdrantClient, collection: str, video_id: str) -> None:
    """Bir videonun tüm pencerelerini siler (yeniden ingest öncesi temizlik)."""
    client.delete(
        collection_name=collection,
        points_selector=qm.FilterSelector(
            filter=qm.Filter(must=[qm.FieldCondition(
                key="video_id", match=qm.MatchValue(value=video_id))])
        ),
        wait=True,
    )
