# İHA Video Arşivinde Semantik Arama

Doğal dil sorgusundan **video kimliği + zaman aralığı** döndüren hibrit
(vektör + yapısal filtre) arama sistemi.

```
"gün batımında deniz üzerinde iki tekne"
   ↓
mission_042   0:14:08 - 0:15:04   (56s, 7 pencere, skor=0.812)
mission_017   1:02:24 - 1:02:56   (32s, 4 pencere, skor=0.774)
```

Tam tasarım dokümanı: [proje-ozeti.md](proje-ozeti.md).
Karar günlüğü ve ölçüm kayıtları: [docs/](docs/).

> **Yeni bir makinede sıfırdan kurup deneyecekseniz:**
> [docs/deneme-rehberi.md](docs/deneme-rehberi.md) — kurulumdan sorgu
> testlerine kadar adım adım, her adımda ne görmeniz gerektiği ve ters
> giderse ne yapacağınızla birlikte.

---

## ⚠️ Önce okuyun: kapasite gerçeği

Bu depo çalışan bir sistem, ama **kapasite planlaması yapılmadan üretime
alınmamalı**. Ölçtüğümüz somut veri:

| | Değer | Kaynak |
|---|---|---|
| Arşiv büyüklüğü | ~300.000 video × 3-5 saat ≈ **900K-1.5M saat** | Kullanıcı teyidi (2026-07-29) |
| Pencere sayısı (8sn/8sn) | **~405M-675M vektör** | Yukarıdakinden hesap |
| Ölçülen embedding hızı | **~0.7x gerçek-zaman** (T4, 1080p, batch'siz) | Colab ölçümü |
| proje-ozeti.md §8 varsayımı | 40x gerçek-zaman | **Doğrulanmadı** |

Aradaki ~60 katlık fark batch'leme, 240-360p proxy ve daha hızlı GPU ile
kapanabilir — ama **tek bir GPU bu arşivi ingest edemez**. Sistem yatay
ölçeklenecek şekilde yazıldı (N worker = N GPU); worker sayısını gerçek bir
throughput ölçümüne dayandırın:

```bash
python -m scripts.ingest_video test_video raw-videos/test.mp4 --local
# cikti sonunda gercek-zaman kati basilir -> proje-ozeti.md §8'i guncelleyin
```

proje-ozeti.md §8'deki diğer doğrulanmamış varsayımlar (gecikme rakamları,
doğruluk yüzdeleri, depolama tahminleri) hâlâ geçerli — bu README'deki
hiçbir sayı onların yerine geçmez.

---

## Mimari

### Ingest hattı

```
MinIO (yeni video)
   └─> bucket notification ─> Kafka ─> Temporal workflow
                                          │
        ┌─────────────────────────────────┴──────────────────────────────┐
        │ 1. proxy uretimi      ffmpeg + NVDEC/NVENC, 360p HEVC          │
        │ 2. telemetri          pymavlink -> 8sn/8sn pencere + turetilmis│
        │                       alanlar (hiz, irtifa, gunes acisi, deniz)│
        │ 3. embedding          Qwen3-VL-Embedding-2B -> pencere/vektor  │
        │ 4. gorsel alanlar     YOLO26 -> vehicle_count                  │
        │ 5. secici caption     Qwen2.5-VL, en hareketli ~%10 pencere    │
        │ 6. yazim              Qdrant (vektor + telemetri payload)      │
        └────────────────────────────────────────────────────────────────┘
```

### Sorgu hattı

```
"gun batiminda deniz uzerinde iki tekne"
   │
   ├─ LLM ayristirma (vLLM + xgrammar)
   │     yapisal: {over_sea: true, is_sunset: true, min_vehicle_count: 2}
   │     semantik: "iki tekne"
   │
   ├─ Hibrit arama (Qdrant): filtre HNSW gezinmesi ICINDE uygulanir
   │     └─ sonuc azsa: filtreyi kademeli gevset, sonuclari isaretle
   │
   ├─ Aralik birlestirme (<=10sn bosluk toleransi)
   │
   └─ [opsiyonel] VLM rerank
```

---

## Neden bu teknoloji seçimleri

Her biri ölçüme dayanıyor; detaylı kayıtlar [docs/worklog_2026-07-28.md](docs/worklog_2026-07-28.md).

### Vektör deposu: Qdrant (ClickHouse değil)

ClickHouse'ta `WHERE` + vektör `ORDER BY` birlikte kullanıldığında bir ikilem
doğuyor — `EXPLAIN indexes=1` ile doğruladık:

| Strateji | HNSW kullanımı | Sonuç |
|---|---|---|
| `prefilter` | **Granül budama 72/72 = hiç yok** | Tam ama brute-force |
| `postfilter` | Granül budama 29/72 | Hızlı ama LIMIT'ten az sonuç dönebilir |

Qdrant filtreyi HNSW graf gezinmesinin **içinde** uyguluyor — tek yolda hem
hızlı hem tam. Ölçtük: 100K korpusta varsayılan filtreli-HNSW, `exact=True`
brute-force ile 21 sorgunun **21'inde birebir aynı top-3**'ü verdi ve ~1.5x
daha hızlıydı (22ms vs 33ms). 100K ölçekte ClickHouse'a göre 4-16x hızlı.

**Dürüst çekince:** 1M ölçekte hem ClickHouse hem Qdrant bu test ortamında
(Docker Desktop/Windows, ~7.6GB paylaşımlı VM belleği) ciddi yavaşladı. Kök
neden kesin belirlenemedi — muhtemelen altyapı sınırı, ama kanıtlanmadı.
Gerçek donanımda yeniden ölçülmeli.

### Embedding modeli: Qwen3-VL-Embedding-2B

Üç aday gerçek SeaDronesSee verisinde karşılaştırıldı. **Belirleyici kriter
lisans oldu:**

| Model | Lisans | Durum |
|---|---|---|
| VideoCLIP-XL | CC-BY-NC-SA 4.0 | ✗ ticari/savunma kullanımına kapalı |
| EBind | CC-BY-NC-SA 4.0 | ✗ ticari/savunma kullanımına kapalı |
| **Qwen3-VL-Embedding-2B** | **Apache-2.0** | ✓ tek uygun aday |

Retrieval kalitesi açısından "en iyi" olduğu **iddia edilmiyor** —
proje-ozeti.md §5'in gerektirdiği golden set karşılaştırması
(InternVideo2/VideoPrism/LanguageBind dahil) hâlâ yapılmadı.

### Pencereleme: 8sn/8sn (örtüşmesiz)

%50 örtüşmeli 4sn kaydırmadan değiştirildi. Gerçek envanterde (~1.2M saat)
örtüşme ~1 milyar vektör demekti; kaydırmayı pencereye eşitlemek vektör
sayısını, GPU maliyetini ve indeks boyutunu **yarıya** indiriyor.

Recall'e etkisini N=6 gerçek klipte test ettik: örtüşmesiz şema **daha kötü
çıkmadı** (Recall@3 %50 vs %33). Ancak N=6 güvenilir bir sonuç değil —
tek bir klibin sırası bu farkı tamamen açıklıyor. Gerçek karar golden set
gerektiriyor (§7).

### Filtreleme: hard + otomatik gevşetme

Bu projede **hard filtrenin doğru cevabı gerçekten kaybettirdiğini** iki
bağımsız yöntemle ölçtük:

1. **Sentetik korpus:** içerikle korelasyonlu %10 seçicilikte filtre,
   Recall@3'ü %28.6 → %9.5'e düşürdü. Eşleştirilmiş bootstrap %95 GA
   `[-38.1, -4.8]` puan — **sıfırı dışlıyor, fark gerçek**.
2. **Sadece gerçek veri:** 21 SeaDronesSee videosu + gerçek `avg_altitude_m`
   ile "irtifa < 20m" filtresi, 21 sorgunun **17'sinde** doğru cevabı
   *yapısal olarak* dışladı — hangi arama algoritması kullanılırsa kullanılsın
   kurtarılamaz.

Ayrıca izole ettik: bu kayıp **HNSW'nin yaklaşıklığından gelmiyor**. Filtreyi
geçen adaylar arasında HNSW, brute-force ile birebir aynı sonucu verdi
(ClickHouse 0/10 ve 0/2 sapma, Qdrant 0/21 sapma). Sorun tamamen filtrenin
kendisinde.

**Çözüm:** önce hard filtre (kesin, öngörülebilir); sonuç `SEARCH_MIN_RESULTS`
altına düşerse filtre kademeli gevşetilir ve gevşetilmiş sonuçlar
`[yaklasik]` olarak işaretlenir. Gevşetme sırası, çıkarıma en çok dayanan
alandan (`is_sunset`, `is_night`) kullanıcının açıkça yazdığına (`sensor_type`)
doğru gider.

---

## Kurulum

### Gereksinimler

- Python 3.11+
- NVIDIA GPU (RTX 4060 sınıfı yeterli — ingest için; üretim ölçeği için filo)
- ffmpeg (NVDEC/NVENC destekli derleme önerilir)
- Docker + Docker Compose

### 1. Bağımlılıklar

```bash
pip install -r requirements.txt      # CUDA'li PyTorch dahil (GPU varsayilan)
python -m scripts.check_env          # <- ATLAMAYIN
```

`requirements.txt` içindeki `--extra-index-url .../cu126` satırı torch'u
PyTorch'un CUDA index'inden çeker. Farklı bir CUDA sürümü gerekiyorsa
(ör. RTX 50xx/Blackwell) o satırı `cu128` ile değiştirin.

| Dosya | Ne zaman |
|---|---|
| `requirements.txt` | **Varsayılan** — GPU'lu ingest/sorgu makinesi |
| `requirements-cpu.txt` | Sadece geliştirme/CI. Üretimde kullanmayın (ölçüldü: 0,045x gerçek-zaman) |
| `requirements-serving.txt` | vLLM sunucusu (sorgu ayrıştırma/caption/rerank). **Linux gerektirir** |

> **⚠️ Neden `check_env` şart:** Bu yığındaki en pahalı iki hata çökmüyor,
> **sessizce bozuluyor.** (1) torch'un CPU derlemesi kurulursa hiçbir hata
> almazsınız — sistem çalışır, ingest ~15-20 kat yavaş koşar. (2)
> `qwen-vl-utils < 0.0.14` Qwen3-VL çağrılarında hata fırlatmadan placeholder
> vektör döndürür ve Recall'ü şansa eşitler. İkincisi gerçekten başımıza geldi
> (bkz. [docs/worklog_2026-07-29.md](docs/worklog_2026-07-29.md)). `check_env`
> her ikisini de yakalar.

### vLLM (yapısal sorgu ayrıştırma) — ayrı kurulum

vLLM'in resmi platform listesinde **Windows yok** (NVIDIA CUDA / AMD ROCm /
Intel XPU / Apple Silicon — hepsi Linux tabanlı). Windows'ta `pip install vllm`
hazır paket bulamayıp kaynaktan derlemeye çalışır ve başarısız olur.

| Ortam | Yol |
|---|---|
| Linux / Colab / Kaggle | `pip install -r requirements-serving.txt` |
| Windows + Docker | `docker compose --profile gpu up -d vllm` |
| Windows + WSL2 | WSL içine kurun, host'tan `localhost:8000` |
| Ayrı sunucu | `VLLM_BASE_URL`'i ona yönlendirin |

**vLLM olmadan da arama çalışır** — sorgu ayrıştırıcı zarifçe semantik metne
düşer. Kaybedilen tek şey yapısal filtreleme ("gece", "3 tekne" gibi
ifadelerin filtreye dönüşmesi); embedding, gevşetme ve aralık birleştirme
etkilenmez.

### 2. Altyapı

```bash
cp .env.example .env        # parolalari degistirin
docker compose up -d        # MinIO, Qdrant, Postgres, Kafka, Temporal
python -m scripts.init_storage
```

Kontrol: MinIO konsolu <http://localhost:9001>, Qdrant <http://localhost:6333/dashboard>,
Temporal UI <http://localhost:8080>.

### 3. vLLM sunucusu (sorgu tarafı)

```bash
vllm serve Qwen/Qwen2.5-7B-Instruct-AWQ \
    --guided-decoding-backend xgrammar \
    --gpu-memory-utilization 0.85 --max-model-len 8192
```

> Tek GPU'da hem embedding worker'ı hem vLLM çalıştırmak VRAM'i böler.
> 4060 sınıfı tek kartta önce ingest'i bitirip sonra sorgu tarafını açmak
> daha pratik. vLLM kapalıyken arama yine çalışır — sorgu tamamen semantik
> metne düşer, yapısal filtreler devre dışı kalır.

---

## Kullanım

### Otomatik ingest (üretim yolu)

```bash
python -m scripts.setup_minio_notifications   # bir kez
python -m ingest.worker            # her GPU makinesinde bir tane
python -m ingest.kafka_consumer    # bir tane
```

Artık `raw-videos` bucket'ına düşen her video otomatik ingest edilir.

### Elle ingest

```bash
# Video (+ opsiyonel telemetri) yukle
python -m scripts.register_video mission_042 /veri/mission_042.mp4 \
    --telemetry /veri/mission_042.tlog

# Temporal uzerinden (dayanikli)
python -m scripts.ingest_video mission_042 mission_042/raw.mp4

# Temporal olmadan (debug + hiz olcumu)
python -m scripts.ingest_video mission_042 mission_042/raw.mp4 --local

# Klasordeki tum videolar
python -m scripts.register_video --dir /veri/ucuslar/
```

### Sorgu

```bash
python -m scripts.query_cli "gun batiminda deniz uzerinde iki tekne"
python -m scripts.query_cli "50 metre uzerinde hizli ucan arac" --top-k 50
python -m scripts.query_cli --interactive
```

Python API:

```python
from query import run_query

response = run_query("gece kiyi seridinde hareket eden tekne")
for interval in response.intervals:
    print(interval.video_id, interval.t_start, interval.t_end, interval.score)

if response.was_relaxed:
    print("Gevsetilen filtreler:", response.relaxed_fields)
```

### Değerlendirme

```bash
# Golden set (proje-ozeti.md §7 - ASIL yontem)
python -m scripts.eval_retrieval --golden poc/golden_set/queries.jsonl

# Golden set yokken zayif saglik sinyali
python -m scripts.eval_retrieval --self-retrieval
```

Golden set formatı (JSONL):

```json
{"query": "deniz uzerinde iki tekne", "video_id": "mission_042", "t_start": 848.0, "t_end": 904.0}
```

---

## Yapılandırma

Tümü ortam değişkeni — kodda sabit bağlantı bilgisi yok. Tam liste:
[.env.example](.env.example). Ölçekte en çok etkisi olanlar:

| Değişken | Varsayılan | Etki |
|---|---|---|
| `STRIDE_S` | `8.0` | `WINDOW_S`'e eşit = örtüşmesiz. `4.0` yaparsanız vektör sayısı ve GPU maliyeti **ikiye katlanır** |
| `EMBEDDING_DIM` | `2048` | MRL ile 64-2048 arası kısaltılabilir. 512d ≈ 4x daha az depolama (kalite etkisi ölçülmedi) |
| `EMBEDDING_BATCH_SIZE` | `8` | Throughput'un en büyük belirleyicisi — VRAM'inize göre yükseltin |
| `QDRANT_QUANTIZATION` | `scalar` | int8, 4x küçültür. ~540M vektörde RAM'e sığmanın pratik yolu |
| `QDRANT_ON_DISK` | `true` | Vektörler diskte, HNSW grafı RAM'de |
| `SEARCH_MIN_RESULTS` | `5` | Bu sayının altında sonuç kalırsa filtre gevşetilir |
| `MAX_CONCURRENT_ACTIVITIES` | `1` | Tek GPU'lu worker'da 1 bırakın (VRAM bölünmesi/OOM) |
| `EMBEDDING_DTYPE` | `auto` | Ampere+ (cc≥8.0) bf16, öncesi fp16. Turing'de bf16 ~10x yavaş (ölçüldü) |

---

## Proje yapısı

```
common/
  config.py          tum ortam degiskenleri
  qdrant_store.py    koleksiyon kurulumu, yazim, filtreli arama
  llm.py             vLLM istemcisi (sema-zorlamali JSON + gorulu sohbet)
  minio_client.py    nesne deposu

ingest/
  workflow.py        Temporal workflow (6 aktivite)
  worker.py          Temporal worker (yatay olcekler)
  kafka_consumer.py  MinIO notification -> workflow tetikleyici
  activities/
    proxy_generation.py     ffmpeg + NVDEC/NVENC
    telemetry_processing.py pymavlink + pencereleme + turetilmis alanlar
    clip_embedding.py       Qwen3-VL-Embedding-2B (+ MRL, batch, dtype)
    visual_fields.py        YOLO26
    selective_caption.py    Qwen2.5-VL, sahne degisimine gore secici
    write_clips.py          Qdrant yazimi (idempotent)

query/
  pipeline.py        uctan uca: ayristir -> ara -> birlestir -> rerank
  llm_parser.py      vLLM + xgrammar, "false yerine null" guvenlik agi
  filter_builder.py  Qdrant filtresi + kademeli gevsetme merdiveni
  hybrid_search.py   filtreli HNSW + gevsetme dongusu
  interval_merge.py  ardisik pencereleri araliga birlestirme
  rerank.py          opsiyonel VLM dogrulama

scripts/             init_storage, register_video, ingest_video,
                     query_cli, eval_retrieval, setup_minio_notifications
tests/               37 test (filtre gevsetme, aralik birlestirme,
                     pencereleme, ayristirici guvenlik agi)
```

---

## Testler

```bash
python -m pytest tests/ -q      # 37 test, harici servis gerektirmez
```

Testler kritik mantığı kapsıyor: filtre gevşetme merdiveni, aralık
birleştirme skorlama/işaretleme kuralları, pencereleme sınır durumları,
nokta kimliği idempotanslığı ve ayrıştırıcının "false yerine null" güvenlik
ağı. Model çıkarımı ve veritabanı çağrıları test edilmiyor (harici servis
gerektirirler).

---

## Bilinen sınırlamalar

- **Golden set yok.** proje-ozeti.md §7 200-500 etiketli sorgu istiyor;
  elimizde yok. Model seçimi (§5), pencereleme (§9) ve MRL boyutu kararları
  bu olmadan doğrulanamaz.
- **1M+ ölçekte davranış doğrulanmadı.** Test ortamında hem ClickHouse hem
  Qdrant yavaşladı; kök neden kesinleşmedi.
- **YOLO26 IR fine-tune yok.** Varsayılan COCO ağırlıkları RGB'de çalışır,
  termalde başarımı ölçülmedi.
- **Telemetri zaman hizalaması varsayım.** MAVLink log'unun ilk kaydı video
  t=0 kabul ediliyor; arşivinizde sabit kayma varsa `TELEMETRY_OFFSET_S`.
- **`over_sea` kıyı poligonu gerektirir.** `COASTLINE_GEOJSON` verilmezse
  alan `None` kalır ve o filtre çalışmaz.
- **Sorgu gecikmesi ölçülmedi.** §8'deki 300ms tahmini mimari akıl
  yürütmeydi, hiç doğrulanmadı.
