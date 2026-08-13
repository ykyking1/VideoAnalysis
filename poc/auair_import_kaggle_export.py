"""Kaggle'da (gerçek GPU'da) embed edilmiş AU-AIR Qdrant noktalarını -
JSON'a dışarı aktarılmış hâlde - yerel Qdrant'a (Docker) aktarır. GPU/model
GEREKTİRMEZ, sadece nokta kopyalama - embedding'i tekrar üretmek yerine
Kaggle'da zaten hesaplanmış vektörleri yeniden kullanır (bkz.
docs/worklog_2026-08-13.md: yerel GPU ~0.10x gerçek-zaman, 8 video ~21 saat
sürerdi; Kaggle T4 8 videoyu 814.5sn'de bitirdi).

DIŞA AKTARMA (Kaggle tarafı, ayrı bir hücrede çalıştırılır):
    import json
    from common.qdrant_store import get_client
    client = get_client()
    points, offset = client.scroll(collection_name='clips_auair_test', limit=200,
                                    with_payload=True, with_vectors=True)
    all_points = list(points)
    while offset is not None:
        more, offset = client.scroll(collection_name='clips_auair_test', limit=200,
                                      offset=offset, with_payload=True, with_vectors=True)
        all_points += more
    export = [{'id': p.id, 'vector': p.vector, 'payload': p.payload} for p in all_points]
    with open('/kaggle/working/clips_auair_test_export.json', 'w', encoding='utf-8') as f:
        json.dump(export, f)
    # Dosyayi Kaggle'in sol panelindeki dosya gezgininden (Output) indirin.

İÇE AKTARMA (bu makinede, bu script):
    python -m poc.auair_import_kaggle_export clips_auair_test_export.json
"""
import argparse
import json

from qdrant_client.http import models as qm

from common.qdrant_store import ensure_collection, get_client


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("export_path", help="Kaggle'dan indirilen JSON dosyasi")
    ap.add_argument("--collection", default="clips_auair_test")
    args = ap.parse_args()

    with open(args.export_path, encoding="utf-8") as f:
        data = json.load(f)
    print(f"{len(data)} nokta okundu: {args.export_path}")
    if not data:
        print("UYARI: dosya bos, aktarilacak bir sey yok.")
        return 1

    client = get_client()
    ensure_collection(client, collection=args.collection)

    points = [qm.PointStruct(id=d["id"], vector=d["vector"], payload=d["payload"]) for d in data]
    chunk = 256
    for start in range(0, len(points), chunk):
        client.upsert(collection_name=args.collection, points=points[start:start + chunk], wait=True)
        print(f"  {min(start + chunk, len(points))}/{len(points)} yazildi")

    info = client.get_collection(args.collection)
    print(f"\nTamam: '{args.collection}' koleksiyonunda simdi {info.points_count} nokta var.")
    print("Video onizlemesi icin ayrica: python -m poc.auair_local_proxies --all")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
