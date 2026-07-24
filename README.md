# İHA Video Arşivinde Semantik Arama

Doğal dil sorgusu → video kimliği + zaman aralığı listesi döndüren hibrit (vektör + yapısal filtre) arama sistemi.

Mimari kararların ve gerekçelerinin tam dökümü için bkz. [proje-ozeti.md](proje-ozeti.md).
Claude/gelecekteki oturumlar için bağlam özeti: [CLAUDE.md](CLAUDE.md).

## Durum

İlk uçtan uca yerel test yapılıyor (bkz. "Yerel test ortamı" altında). Kapasite/GPU-saat
gibi sayısal varsayımlar hâlâ [proje-ozeti.md §8](proje-ozeti.md#8-doğrulanmamış-varsayımlar--kritik-i̇mplementasyondan-önce-kontrol-edi̇lmeli)'de
listelendiği gibi doğrulanmadı - bu test yalnızca pipeline'ın *mekanik olarak*
uçtan uca çalıştığını doğruluyor, production kapasitesini değil.

## Yerel test ortamı

Donanım: NVIDIA GT1030 4GB (Pascal, NVENC yok - proxy encode yazılımla yapılıyor,
decode NVDEC ile hızlandırılıyor). Bu VRAM'e Qwen14B+SGLang ve Qwen2.5-VL+vLLM
sığmadığı için, o iki bileşen yerel testte Ollama üzerinden küçük quantize
modellerle değiştirildi (bkz. [CLAUDE.md](CLAUDE.md) "Yerel test sapmaları").

Veri: SeaDronesSee (deniz/maritime İHA sahnesi). Bu veri setinde uçuş telemetrisi
(MAVLink) yok, bu yüzden `ingest/activities/telemetry_processing.py` sabit
8sn/4sn pencereleme yapıyor ve telemetriden türeyen alanları (hız, irtifa,
güneş açısı, deniz-üstü) NULL bırakıyor - bkz. dosyanın docstring'i.

### Kurulum ve çalıştırma

```
docker compose up -d
python scripts/init_schema.py                 # ClickHouse `clips` tablosu
docker compose exec ollama ollama pull qwen2.5:3b
docker compose exec ollama ollama pull moondream

python scripts/register_video.py <video_id> <yerel_video_dosyası>
python scripts/ingest_video.py <video_id>

python scripts/query_cli.py "gün batımında deniz üzerinde yüksek hızlarda uçan bir tekne"
```

MinIO konsolu: http://localhost:9001 · Temporal UI: http://localhost:8080

## Klasör yapısı

```
proje-ozeti.md      Tasarım dokümanı (kararlar, gerekçeler, açık sorular)
CLAUDE.md            Gelecekteki Claude oturumları için proje bağlamı
schema/               ClickHouse / PostgreSQL tablo tanımları
ingest/               Temporal workflow + aktiviteler (proxy, telemetri, embedding, YOLO, caption)
query/                Sorgu ayrıştırma, hibrit arama, aralık birleştirme, rerank
poc/                  Adım 0 doğrulama script'leri (envanter, hız ölçümü, golden set)
docs/                 Ek notlar / ölçüm sonuçları
```

## Sonraki adımlar

Bkz. [proje-ozeti.md §11](proje-ozeti.md#11-sonraki-adımlar-öneri-sırası).
