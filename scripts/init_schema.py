"""schema/clickhouse_clips.sql ve schema/clips_videoclip_xl.sql'i çalışan
ClickHouse'a uygular, sonra HNSW vektör indekslerini mevcut satırlar için
MATERIALIZE eder (yeni INSERT'ler otomatik index'lenir, bu sadece geriye
dönük doldurma içindir).
(schema/postgres_state.sql zaten docker-compose.yml içinde
docker-entrypoint-initdb.d ile otomatik uygulanıyor.)

Kullanım: python scripts/init_schema.py
"""
import pathlib

import clickhouse_connect

from common import config

SCHEMA_DIR = pathlib.Path(__file__).parent.parent / "schema"
SCHEMA_FILES = ["clickhouse_clips.sql", "clips_videoclip_xl.sql"]
INDEXES = [
    ("clips", "clips_embedding_idx"),
    ("clips_videoclip_xl", "clips_videoclip_xl_embedding_idx"),
]


def apply_schema_file(client, path: pathlib.Path) -> None:
    sql = path.read_text(encoding="utf-8")
    for statement in sql.split(";"):
        code_lines = [
            line for line in statement.splitlines()
            if line.strip() and not line.strip().startswith("--")
        ]
        if code_lines:
            client.command("\n".join(code_lines))


def main() -> None:
    client = clickhouse_connect.get_client(
        host=config.CLICKHOUSE_HOST, port=config.CLICKHOUSE_PORT,
        username=config.CLICKHOUSE_USER, password=config.CLICKHOUSE_PASSWORD,
        database=config.CLICKHOUSE_DB,
    )
    for filename in SCHEMA_FILES:
        apply_schema_file(client, SCHEMA_DIR / filename)
    print("ClickHouse şeması uygulandı.")

    for table, index_name in INDEXES:
        client.command(f"ALTER TABLE {table} MATERIALIZE INDEX {index_name}")
        print(f"MATERIALIZE INDEX {index_name} ON {table} tamamlandı.")


if __name__ == "__main__":
    main()
