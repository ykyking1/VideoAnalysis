"""Depolama kıstası: ClickHouse `clips` tablosunun gerçek disk üzerindeki
boyutunu (sıkıştırma dahil) raporlar - proje-ozeti.md §6'daki teorik
hesaplardan farklı olarak gerçek ölçüm.

Kullanım: python scripts/storage_report.py
"""
import clickhouse_connect

from common import config


def report() -> None:
    client = clickhouse_connect.get_client(
        host=config.CLICKHOUSE_HOST, port=config.CLICKHOUSE_PORT,
        username=config.CLICKHOUSE_USER, password=config.CLICKHOUSE_PASSWORD,
        database=config.CLICKHOUSE_DB,
    )

    row = client.query(
        """
        SELECT
            sum(rows) AS total_rows,
            sum(bytes_on_disk) AS total_bytes,
            sum(data_uncompressed_bytes) AS uncompressed_bytes
        FROM system.parts
        WHERE database = {db:String} AND table = 'clips' AND active
        """,
        parameters={"db": config.CLICKHOUSE_DB},
    ).result_rows[0]

    total_rows, total_bytes, uncompressed_bytes = row
    total_rows = total_rows or 0

    print("--- Depolama kıstası (gerçek ClickHouse ölçümü) ---")
    print(f"Toplam klip sayısı: {total_rows}")
    if total_rows == 0:
        print("Henüz klip yok.")
        return

    print(f"Disk üzerinde (sıkıştırılmış): {total_bytes / 1024:.1f} KB "
          f"({total_bytes / total_rows:.0f} byte/klip)")
    print(f"Sıkıştırılmamış: {uncompressed_bytes / 1024:.1f} KB "
          f"({uncompressed_bytes / total_rows:.0f} byte/klip)")
    if total_bytes:
        print(f"Sıkıştırma oranı: {uncompressed_bytes / total_bytes:.2f}x")

    print("\nNOT: Bu, mevcut küçük test korpusunun ölçümüdür. Gerçek arşiv "
          "ölçeğine (proje-ozeti.md §8 - 1,5 PB varsayımı hâlâ doğrulanmadı) "
          "extrapolasyon yapılmadan bu sayı doğrudan kapasite planlamasında "
          "kullanılmamalı.")


if __name__ == "__main__":
    report()
