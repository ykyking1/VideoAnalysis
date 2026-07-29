"""Yapısal filtreleri Qdrant koşullarına çevirir ve gerektiğinde gevşetir.

NEDEN GEVŞETME VAR (ölçüme dayalı): Bu projede hard filtrenin doğru cevabı
gerçekten kaybettirdiğini iki bağımsız yöntemle ölçtük:
  1) Sentetik korpusta içerikle korelasyonlu bir filtre (%10 seçicilik)
     Recall@3'ü %28.6 -> %9.5'e düşürdü; eşleştirilmiş bootstrap %95 güven
     aralığı [-38.1, -4.8] puan, yani sıfırı dışlıyor - fark gerçek.
  2) Sadece gerçek veriyle (21 SeaDronesSee videosu + gerçek avg_altitude_m)
     "irtifa < 20m" filtresi 21 sorgunun 17'sinde doğru cevabı YAPISAL olarak
     dışladı - hangi arama algoritması kullanılırsa kullanılsın kurtarılamaz.
Detay: docs/worklog_2026-07-28.md.

Bu, filtrelemenin kötü bir fikir olduğu anlamına gelmiyor - filtrenin
YANLIŞ/DAR olduğu durumda sonucun kalıcı kaybedilmesi anlamına geliyor.
Çözüm: önce hard filtre (kesin ve öngörülebilir), sonuç azsa filtreyi
kademeli gevşet ve gevşetilmiş sonuçları AÇIKÇA işaretle. Kullanıcı hem
kesin eşleşmeleri hem "yakın" olanları görür, hangisinin ne olduğunu bilir.

GEVŞETME SIRASI: en çok çıkarıma dayanan (dolayısıyla en çok yanılan) alan
önce düşer. `is_night`/`is_sunset` LLM'in kelimeden çıkardığı + güneş açısı
eşiğine çevrilen alanlar - en kırılgan. `sensor_type` ise kullanıcının
açıkça yazdığı bir şey - en son düşer.

NULL TELEMETRİ: Telemetrisi olmayan klipte `over_sea` alanı null'dur ve
`over_sea=true` filtresini geçemez. Bu doğru davranış (bilinmiyor ≠ evet)
ama telemetrisiz arşiv bölümlerinin yapısal sorgularda görünmez olması
demek - gevşetme bunu da telafi eder.
"""
from dataclasses import replace

from qdrant_client.http import models as qm

from common import config
from query.llm_parser import StructuredFilters

# Once dusen en ustte: cikarima en cok dayanan alan.
RELAXATION_ORDER = (
    "is_sunset",
    "is_night",
    "over_sea",
    "min_speed_kmh",
    "max_speed_kmh",
    "min_agl_m",
    "max_agl_m",
    "min_vehicle_count",
    "sensor_type",
)


def build_filter(filters: StructuredFilters) -> qm.Filter | None:
    """StructuredFilters -> Qdrant Filter. Hiç aktif alan yoksa None döner
    (filtresiz arama)."""
    conditions: list[qm.FieldCondition] = []

    if filters.sensor_type is not None:
        conditions.append(qm.FieldCondition(
            key="sensor_type", match=qm.MatchValue(value=filters.sensor_type)))

    if filters.over_sea is not None:
        conditions.append(qm.FieldCondition(
            key="over_sea", match=qm.MatchValue(value=filters.over_sea)))

    if filters.min_speed_kmh is not None or filters.max_speed_kmh is not None:
        conditions.append(qm.FieldCondition(
            key="avg_speed_kmh",
            range=qm.Range(gte=filters.min_speed_kmh, lte=filters.max_speed_kmh)))

    if filters.min_agl_m is not None or filters.max_agl_m is not None:
        conditions.append(qm.FieldCondition(
            key="agl_m", range=qm.Range(gte=filters.min_agl_m, lte=filters.max_agl_m)))

    if filters.min_vehicle_count is not None:
        conditions.append(qm.FieldCondition(
            key="vehicle_count", range=qm.Range(gte=filters.min_vehicle_count)))

    # is_sunset / is_night -> sun_elevation esikleri. "gece" gibi bir kavramin
    # deterministik karsiligi; modele tahmin ettirilmiyor (proje-ozeti.md §3.1).
    if filters.is_sunset:
        lo, hi = config.SUNSET_ELEVATION_RANGE
        conditions.append(qm.FieldCondition(
            key="sun_elevation", range=qm.Range(gte=lo, lte=hi)))
    if filters.is_night:
        conditions.append(qm.FieldCondition(
            key="sun_elevation", range=qm.Range(lt=config.NIGHT_ELEVATION_MAX)))
    elif filters.is_night is False:
        conditions.append(qm.FieldCondition(
            key="sun_elevation", range=qm.Range(gte=config.NIGHT_ELEVATION_MAX)))

    return qm.Filter(must=conditions) if conditions else None


def relaxation_ladder(filters: StructuredFilters) -> list[tuple[StructuredFilters, list[str]]]:
    """Filtrenin kademeli gevşetilmiş sürümlerini döner.

    Her adım: (gevşetilmiş filtre, o ana kadar düşürülmüş alan adları).
    İlk adım her zaman orijinal (hiçbir şey düşmemiş) filtredir; son adım
    tamamen filtresizdir."""
    ladder: list[tuple[StructuredFilters, list[str]]] = [(filters, [])]

    current = filters
    dropped: list[str] = []
    for field_name in RELAXATION_ORDER:
        if getattr(current, field_name, None) is None:
            continue
        current = replace(current, **{field_name: None})
        dropped = dropped + [field_name]
        ladder.append((current, list(dropped)))

    return ladder


def describe(filters: StructuredFilters) -> str:
    active = filters.active_fields()
    if not active:
        return "filtre yok"
    return ", ".join(f"{k}={getattr(filters, k)}" for k in active)
