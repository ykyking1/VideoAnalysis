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

## Klasör yapısı

- `schema/` — ClickHouse `clips` tablosu ve PostgreSQL durum takibi DDL'leri
- `ingest/` — Temporal workflow + 5 aktivite (şu an iskelet, gerçek model/parametre
  seçimleri Adım 0 sonrası netleşecek)
- `query/` — sorgu ayrıştırma, hibrit arama, aralık birleştirme (bu mantık bağımsız,
  implemente edildi), rerank
- `poc/` — Adım 0 doğrulama: gerçek envanter taraması, embedding hız ölçümü, golden set

## Teknoloji yığını

Bkz. proje-ozeti.md §10. Öne çıkanlar: ffmpeg+NVDEC, Kafka, Temporal, ClickHouse
(HNSW+skip index), YOLO26, Qwen2.5-VL+vLLM, Qwen 14B+xgrammar+SGLang, PostgreSQL, MinIO.
