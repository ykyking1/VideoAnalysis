"""Mimariyi GPU olmadan uçtan uca dener.

NE TEST EDER: Depolama + arama katmanının tamamı - Qdrant koleksiyon kurulumu,
payload indeksleri, filtreli HNSW araması, kademeli filtre gevşetme, aralık
birleştirme. Yani bu projenin ASIL mimari kararlarını.

NE TEST ETMEZ: Embedding modeli, YOLO, caption VLM, sorgu ayrıştırma LLM'i.
Bunlar GPU gerektiriyor ve ayrı doğrulanmalı.

NASIL: Embedding modelini çalıştırmak yerine, daha önce gerçek donanımda
hesaplanmış gerçek vektörleri (data/qwen3vl_real_embeddings.npz - 21 gerçek
SeaDronesSee videosu, 2048d Qwen3-VL) kullanır. Vektörler gerçek, sorgular
gerçek; sadece "vektörü şimdi üret" adımı atlanıyor.

TELEMETRİ: SeaDronesSee'de uçuş telemetrisi yok. Filtre/gevşetme mantığını
test edebilmek için pencerelere sentetik ama gerçekçi telemetri atanır
(--synthetic-telemetry ile açık, varsayılan açık). Bu ÖLÇÜM DEĞİL, mekanizma
testidir - "filtre gevşetme tetikleniyor mu" sorusunu cevaplar, "gerçek
arşivde Recall ne olur" sorusunu DEĞİL.

Kullanım:
    python -m scripts.smoke_test
    python -m scripts.smoke_test --collection clips_smoke --keep
"""
import argparse
import pathlib
import sys

import numpy as np

from common.console import use_utf8_stdout
from common import config
from common.qdrant_store import ClipPayload, ensure_collection, get_client, upsert_clips
from query.filter_builder import build_filter, relaxation_ladder
from query.hybrid_search import Match, _to_match
from query.interval_merge import merge_matches
from query.llm_parser import StructuredFilters
from common.qdrant_store import search as qdrant_search

use_utf8_stdout()

EMBEDDINGS_PATH = pathlib.Path(__file__).parent.parent / "data" / "qwen3vl_real_embeddings.npz"
WINDOW_S = 8.0


def load_real_embeddings() -> dict:
    if not EMBEDDINGS_PATH.exists():
        raise SystemExit(
            f"{EMBEDDINGS_PATH} bulunamadi.\n"
            "Bu dosya 21 gercek SeaDronesSee videosunun Qwen3-VL embedding'lerini "
            "iceriyor ve GPU'lu bir ortamda uretilmisti. Yoksa bu duman testi "
            "calistirilamaz - onun yerine gercek ingest'i deneyin."
        )
    data = np.load(EMBEDDINGS_PATH, allow_pickle=True)
    return {
        "video": data["video"].astype(np.float32),
        "text": data["text"].astype(np.float32),
        "clip_ids": data["clip_ids"].tolist(),
        "queries": data["queries"].tolist(),
    }


def normalize(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return vectors / norms


def build_payloads(clip_ids: list[str], rng: np.random.Generator,
                    synthetic_telemetry: bool) -> list[ClipPayload]:
    """Her gercek videoyu bir pencere olarak kaydeder.

    Telemetri sentetik: SeaDronesSee'de MAVLink log yok. Deger araliklari
    gercekci tutuldu (irtifa 5-100m, hiz 0-40km/s) ama bunlar OLCUM DEGIL."""
    payloads = []
    for i, clip_id in enumerate(clip_ids):
        t_start = 0.0
        payload = ClipPayload(
            video_id=clip_id,
            t_start=t_start,
            t_end=t_start + WINDOW_S,
            sensor_type="rgb",
        )
        if synthetic_telemetry:
            payload.agl_m = float(rng.uniform(5.0, 100.0))
            payload.avg_speed_kmh = float(rng.uniform(0.0, 40.0))
            payload.sun_elevation = float(rng.uniform(-20.0, 60.0))
            payload.over_sea = bool(rng.random() < 0.7)
            payload.vehicle_count = int(rng.poisson(1.5))
        payloads.append(payload)
    return payloads


def check_infrastructure() -> object:
    print("[1/5] Altyapi kontrolu")
    try:
        client = get_client()
        collections = client.get_collections()
    except Exception as exc:  # noqa: BLE001
        raise SystemExit(
            f"  X Qdrant'a baglanilamadi ({config.QDRANT_HOST}:{config.QDRANT_PORT}): {exc}\n"
            f"  Cozum: docker compose up -d qdrant"
        )
    print(f"  OK Qdrant ayakta ({len(collections.collections)} koleksiyon)")
    return client


def load_corpus(client, collection: str, real: dict, synthetic_telemetry: bool) -> list[ClipPayload]:
    print(f"[2/5] Korpus yukleniyor -> '{collection}'")
    ensure_collection(client, collection, recreate=True)

    vectors = normalize(real["video"])
    if vectors.shape[1] != config.EMBEDDING_DIM:
        raise SystemExit(
            f"  X Boyut uyusmazligi: npz {vectors.shape[1]}d, EMBEDDING_DIM={config.EMBEDDING_DIM}\n"
            f"  Cozum: EMBEDDING_DIM={vectors.shape[1]} olarak ayarlayin."
        )

    rng = np.random.default_rng(42)
    payloads = build_payloads(real["clip_ids"], rng, synthetic_telemetry)
    written = upsert_clips(client, collection, vectors.tolist(), payloads)

    info = client.get_collection(collection)
    print(f"  OK {written} nokta yazildi ({vectors.shape[1]}d)")
    print(f"  OK payload indeksleri: {', '.join(sorted(info.payload_schema or {}))}")
    return payloads


def test_unfiltered_search(client, collection: str, real: dict) -> float:
    """Filtresiz vektor aramasi calisiyor mu + taban Recall."""
    print("[3/5] Filtresiz arama (taban)")
    texts = normalize(real["text"])
    hits1 = hits3 = 0

    for qi, clip_id in enumerate(real["clip_ids"]):
        points = qdrant_search(client, collection, texts[qi].tolist(), None, 3)
        ids = [(p.payload or {}).get("video_id") for p in points]
        if ids[:1] == [clip_id]:
            hits1 += 1
        if clip_id in ids:
            hits3 += 1

    n = len(real["clip_ids"])
    print(f"  OK Recall@1={hits1 / n * 100:.1f}%  Recall@3={hits3 / n * 100:.1f}%  (n={n})")
    print("     NOT: 21 videoluk havuzda olculdu - gercek arsiv Recall'u DEGIL.")
    return hits3 / n


def test_filtered_search(client, collection: str, real: dict, payloads: list[ClipPayload]) -> None:
    """Filtreli HNSW ve gevsetme merdiveni calisiyor mu."""
    print("[4/5] Filtreli arama + kademeli gevsetme")
    texts = normalize(real["text"])

    # Kasitli olarak COK DAR bir filtre: gevsetmenin tetiklenmesi bekleniyor.
    tight = StructuredFilters(min_agl_m=90.0, over_sea=True, min_vehicle_count=3)
    qdrant_filter = build_filter(tight)

    matching = client.count(collection_name=collection, count_filter=qdrant_filter).count
    print(f"  Dar filtre ({tight.active_fields()}) -> {matching}/{len(payloads)} nokta geciyor")

    ladder = relaxation_ladder(tight)
    print(f"  Gevsetme merdiveni {len(ladder)} adim: "
          f"{' -> '.join('(' + ','.join(d) + ')' if d else '(orijinal)' for _, d in ladder)}")

    # Merdiveni gercekten yuruyup ilk yeterli sonucu bulan adimi raporla
    min_results = config.SEARCH_MIN_RESULTS
    query_vector = texts[0].tolist()
    for step, (filters, dropped) in enumerate(ladder):
        points = qdrant_search(client, collection, query_vector, build_filter(filters), 10)
        status = "YETERLI" if len(points) >= min_results else "yetersiz"
        label = "orijinal" if not dropped else f"dusen: {', '.join(dropped)}"
        print(f"    adim {step}: {len(points):2d} sonuc  [{status}]  ({label})")
        if len(points) >= min_results:
            break
    else:
        raise SystemExit("  X Merdiven sonuna gelindi ama yeterli sonuc yok - beklenmedik")

    print(f"  OK Gevsetme calisiyor (esik={min_results})")


def test_interval_merge(client, collection: str, real: dict) -> None:
    """Arama sonuclari araliklara dogru birlesiyor mu."""
    print("[5/5] Aralik birlestirme")
    texts = normalize(real["text"])
    points = qdrant_search(client, collection, texts[0].tolist(), None, 5)
    matches = [_to_match(p, exact=True) for p in points]
    intervals = merge_matches(matches)

    print(f"  {len(matches)} eslesme -> {len(intervals)} aralik")
    top = intervals[0]
    print(f"  En iyi: {top.video_id}  {top.t_start:.0f}-{top.t_end:.0f}s  "
          f"skor={top.score:.3f}  pencere={top.n_windows}")

    # Ayni videoda bitisik iki pencere tek araliga birlesmeli
    synthetic = [
        Match("v_test", 0.0, 8.0, score=0.9),
        Match("v_test", 8.0, 16.0, score=0.4),
    ]
    merged = merge_matches(synthetic)
    assert len(merged) == 1 and merged[0].t_end == 16.0, "bitisik pencereler birlesmedi"
    assert merged[0].score == 0.9, "aralik skoru en iyi pencereyi almali"
    print("  OK Bitisik pencereler birlesiyor, skor en iyi pencereden aliniyor")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--collection", default="clips_smoke",
                     help="Test koleksiyonu (uretim koleksiyonuna dokunmamak icin ayri)")
    ap.add_argument("--keep", action="store_true", help="Test sonunda koleksiyonu silme")
    ap.add_argument("--no-synthetic-telemetry", action="store_true")
    args = ap.parse_args()

    print("=" * 70)
    print("MIMARI DUMAN TESTI - depolama + arama katmani (GPU gerektirmez)")
    print("=" * 70)

    real = load_real_embeddings()
    print(f"Gercek veri: {len(real['clip_ids'])} SeaDronesSee videosu, "
          f"{real['video'].shape[1]}d Qwen3-VL embedding\n")

    client = check_infrastructure()
    payloads = load_corpus(client, args.collection, real,
                            not args.no_synthetic_telemetry)
    test_unfiltered_search(client, args.collection, real)
    test_filtered_search(client, args.collection, real, payloads)
    test_interval_merge(client, args.collection, real)

    if not args.keep:
        client.delete_collection(args.collection)
        print(f"\nTest koleksiyonu '{args.collection}' silindi (--keep ile saklayin)")

    print("\n" + "=" * 70)
    print("SONUC: Depolama + arama katmani calisiyor.")
    print("Test EDILMEYENLER (GPU gerekir): embedding modeli, YOLO26,")
    print("caption VLM, sorgu ayristirma LLM'i.")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
