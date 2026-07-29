"""Depolama katmanını kurar: MinIO bucket'ları, Qdrant koleksiyonu, Postgres
durum tabloları.

Idempotent - var olanlara dokunmaz, tekrar tekrar çalıştırılabilir.

Kullanım:
    python -m scripts.init_storage
    python -m scripts.init_storage --recreate-collection   # DIKKAT: verileri siler
"""
import argparse
import sys

from common import config
from common.minio_client import ensure_buckets
from common.qdrant_store import ensure_collection, get_client


def init_qdrant(recreate: bool) -> None:
    client = get_client()
    collection = ensure_collection(client, recreate=recreate)
    info = client.get_collection(collection)
    print(f"[qdrant] koleksiyon '{collection}' hazir "
          f"(boyut={config.EMBEDDING_DIM}, nokta={info.points_count}, "
          f"kuantizasyon={config.QDRANT_QUANTIZATION}, on_disk={config.QDRANT_ON_DISK})")
    schema = info.payload_schema or {}
    print(f"[qdrant] payload indeksleri: {', '.join(sorted(schema)) or '(yok)'}")


def init_minio() -> None:
    ensure_buckets()
    print(f"[minio] bucket'lar hazir: {config.MINIO_BUCKET_RAW}, {config.MINIO_BUCKET_PROXY}")


def init_postgres() -> None:
    """Ingest durum takibi tabloları (proje-ozeti.md §2). Postgres yoksa
    uyarı verip geçer - arama hattı Postgres'e bağlı değil."""
    try:
        import psycopg
    except ImportError:
        print("[postgres] psycopg kurulu degil, atlandi")
        return

    ddl_path = "schema/postgres_state.sql"
    try:
        with open(ddl_path, encoding="utf-8") as f:
            ddl = f.read()
        with psycopg.connect(config.postgres_dsn(), autocommit=True) as conn:
            conn.execute(ddl)
        print(f"[postgres] durum tablolari hazir ({ddl_path})")
    except FileNotFoundError:
        print(f"[postgres] {ddl_path} bulunamadi, atlandi")
    except Exception as exc:  # noqa: BLE001
        print(f"[postgres] atlandi: {exc}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--recreate-collection", action="store_true",
                     help="Qdrant koleksiyonunu SILIP yeniden olusturur (tum vektorler kaybolur)")
    ap.add_argument("--skip-postgres", action="store_true")
    args = ap.parse_args()

    if args.recreate_collection:
        answer = input(f"'{config.QDRANT_COLLECTION}' koleksiyonundaki TUM veriler "
                        f"silinecek. Devam? [yes/N] ").strip().lower()
        if answer != "yes":
            print("iptal edildi")
            return 1

    init_minio()
    init_qdrant(args.recreate_collection)
    if not args.skip_postgres:
        init_postgres()
    return 0


if __name__ == "__main__":
    sys.exit(main())
