# CLAUDE.md

Bu depo, İHA video arşivi için doğal dil → video kimliği + zaman aralığı döndüren
hibrit (vektör + yapısal filtre) arama sistemini kurmak için kullanılıyor.

Tam tasarım dokümanı: [proje-ozeti.md](proje-ozeti.md) — herhangi bir implementasyon
kararı öncesinde okunmalı, özellikle §8 (doğrulanmamış varsayımlar).

## Kritik kural

`proje-ozeti.md` §8'de listelenen varsayımlar (1,5 PB → 300.000 saat dönüşümü, embedding
hızı ~40x gerçek-zaman, gecikme rakamları, model seçimi) **doğrulanmadı**. Bu sayılara
dayanan kapasite planlaması, sabit kod, veya "X TB tutar" gibi iddialar üretmeden önce
kullanıcıya bu varsayımların hâlâ doğrulanmamış olduğunu hatırlat.

## Mimari özet

- **Ingest**: MinIO bucket notification → Kafka → Temporal workflow. 5 aktivite: proxy
  üretimi (ffmpeg+NVDEC), telemetri parse (pymavlink+polars, 8sn/4sn kaydırmalı pencere),
  klip embedding (model TBD, bkz. §5), YOLO26 görsel alanlar, Qwen2.5-VL seçici caption.
  Tek satır ClickHouse `clips` tablosuna yazılır.
- **Query**: Qwen 14B + xgrammar (SGLang) ile yapısal filtre + semantik metne ayrıştırma
  → ClickHouse'da tek sorguda hibrit arama (skip index + HNSW vektör) → ardışık pencereleri
  ≤10sn boşluk toleransıyla aralık birleştirme → opsiyonel Qwen2.5-VL rerank.
- **Depolama**: Ham video soğuk katmana taşınabilir (sorgu hattı dokunmuyor). Proxy
  240-360p HEVC, önizleme kalitesi değil model kalitesi hedefleniyor.

## Model seçimi (§5) — KESİNLEŞMEDİ

X-CLIP (`xuguohai/X-CLIP`, MIT) mevcut lider aday — `microsoft/xclip` (Kinetics
sınıflandırma) ile karıştırılmamalı. VideoCLIP-XL yeni aday. InternVideo2/VideoPrism/
LanguageBind test edilmedi. Karar golden set sonucuna göre verilecek (§7, §11 madde 4).

## Yerel test sapmaları (GT1030 4GB + SeaDronesSee)

İlk uçtan uca yerel test için üretim tasarımından bilinçli sapmalar yapıldı - bunlar
**geçici, sadece mekanik doğrulama** amaçlı, model/altyapı kararı olarak sayılmamalı:

| Bileşen | Üretim hedefi (proje-ozeti.md) | Yerel test |
|---|---|---|
| Query ayrıştırma | Qwen 14B + xgrammar + SGLang | Ollama `qwen2.5:3b` + JSON schema |
| Caption/rerank VLM | Qwen2.5-VL + vLLM | Ollama `moondream` |
| Embedding modeli | xuguohai/X-CLIP (§5, kesinleşmedi) | `microsoft/xclip-base-patch32` (Kinetics fine-tune - §5'in "karıştırılmamalı" dediği model, sadece pip/HF'ten hazır yüklenebildiği için) |
| Görsel alanlar | YOLO26 IR fine-tune | `yolov8n` (COCO ön-eğitimli, "boat" sınıfı vehicle_count'a sayılıyor) |
| Sahne değişim skoru | ffmpeg scene filtresi | Basit ardışık kare farkı (frame diff) |
| Proxy encode | ffmpeg+NVDEC | NVDEC decode + **yazılım** HEVC encode (GT1030'da NVENC donanımı yok) |
| Telemetri | pymavlink+astral+shapely | SeaDronesSee'de telemetri yok → sabit pencereleme, türetilmiş alanlar NULL |
| Vektör indeksi | ClickHouse HNSW | Yok (küçük test korpusu, brute-force cosineDistance) |
| Orkestrasyon | Temporal workflow (`ingest/workflow.py`) | `scripts/ingest_video.py` aktiviteleri Temporal olmadan doğrudan çağırıyor |

Gerçek kapasite/kalite kararları (model seçimi §5, chunking §9, vektör indeks
parametreleri §6) bu yerel testten ÇIKARILMAMALI - onlar hâlâ golden set +
gerçek donanım POC'u gerektiriyor (§7, §11).

## Klasör yapısı

- `schema/` — ClickHouse `clips` tablosu ve PostgreSQL durum takibi DDL'leri
- `common/` — paylaşılan config (.env okuma) ve MinIO istemcisi
- `ingest/` — Temporal workflow + 5 aktivite (yerel test için implemente edildi,
  yukarıdaki tabloda listelenen sapmalarla)
- `query/` — sorgu ayrıştırma (Ollama), hibrit arama (ClickHouse), aralık
  birleştirme, rerank (henüz stub - opsiyonel olduğu için öncelik verilmedi)
- `scripts/` — yerel test çalıştırıcıları: şema kurulumu, video kaydı,
  manuel ingest, sorgu CLI'ı (bkz. README.md "Yerel test ortamı")
- `poc/` — Adım 0 doğrulama: gerçek envanter taraması, embedding hız ölçümü, golden set
- `docker-compose.yml` — MinIO, Kafka, ClickHouse, Postgres, Temporal, Ollama

## Teknoloji yığını

Bkz. proje-ozeti.md §10. Öne çıkanlar: ffmpeg+NVDEC, Kafka, Temporal, ClickHouse
(HNSW+skip index), YOLO26, Qwen2.5-VL+vLLM, Qwen 14B+xgrammar+SGLang, PostgreSQL, MinIO.
