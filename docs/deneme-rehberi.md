# Deneme Rehberi — kurulumdan sonuçlara

Yeni bir makinede (4060 sınıfı GPU) sıfırdan kurup büyük bir veri setiyle
denemek için adım adım. Her adımda **ne görmeniz gerektiği** ve **ters
giderse ne yapacağınız** yazılı.

> **Altın kural:** Adım 5'i (tek videoyla kalibrasyon) atlamayın. Büyük veriyi
> yüklemeden önce gerçek hızı bilmek, saatler sonra "bu bitmeyecek" demekten
> iyidir.

---

## 0. Ön koşullar

```bash
nvidia-smi                    # surucu + GPU gorunmeli
docker --version
ffmpeg -version
python --version              # 3.11+
```

**Disk alanı — küçümsemeyin.** Ölçülen gerçek oranlar:

| Ne | Saat başına | 100 saat video için |
|---|---|---|
| Proxy (360p HEVC) | ~360 MB | ~36 GB |
| Vektörler (2048d, int8 kuantize) | ~15 MB | ~1,5 GB |
| Ham video (siz sağlıyorsunuz) | değişken | — |

Proxy baskın. Ham videoyu MinIO'ya yüklüyorsanız onun alanını da ekleyin.

---

## 1. Kurulum

```bash
git clone https://github.com/ykyking1/VideoAnalysis.git
cd VideoAnalysis
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

**RTX 50xx / Blackwell kartsa:** `requirements.txt`'in ilk satırındaki
`cu126`'yı `cu128` yapın, sonra kurun.

---

## 2. Ortamı doğrulayın — ATLAMAYIN

```bash
python -m scripts.check_env --skip-services
```

**Görmeniz gereken:**
```
[OK  ] torch 2.x.x+cu126 | GPU: NVIDIA GeForce RTX 4060 (8.0 GB, compute 8.9)
[OK  ] bfloat16 Tensor Core destekli (Ampere+)
[OK  ] transformers 4.5x
[OK  ] qwen-vl-utils 0.0.14
```

**`[HATA] torch ... CPU-only` görürseniz DURUN.** Sistem çalışır ama 15-20 kat
yavaş olur ve bunu saatler sonra fark edersiniz:
```bash
pip install torch torchvision --force-reinstall \
    --index-url https://download.pytorch.org/whl/cu126
```

`hevc_nvenc ... BU DONANIMDA calismiyor` uyarısı görürseniz sorun değil — kod
yazılım encode'a düşer, sadece proxy üretimi yavaşlar.

---

## 3. Servisleri başlatın

```bash
cp .env.example .env
# .env icindeki parolalari degistirin

docker compose up -d               # MinIO, Qdrant, Postgres, Kafka, Temporal
python -m scripts.init_storage
python -m scripts.check_env        # bu kez servislerle birlikte
```

**Görmeniz gereken:**
```
[qdrant] koleksiyon 'clips' hazir (boyut=2048, nokta=0, kuantizasyon=scalar)
[qdrant] payload indeksleri: agl_m, avg_speed_kmh, caption, over_sea, ...
```

Arayüzler: MinIO <http://localhost:9001>, Qdrant <http://localhost:6333/dashboard>,
Temporal <http://localhost:8080>.

---

## 4. Telemetri kararı — testin kapsamını belirler

| Durumunuz | Sonuç |
|---|---|
| **Gerçek `.tlog` var** | Tüm yapısal alanlar dolar. Asıl değerli test bu. |
| **Telemetri yok** | `agl_m`, `avg_speed_kmh`, `over_sea`, `sun_elevation` hep `None`. "gece", "deniz üzerinde", "50m üstünde" gibi filtreler **hiç eşleşme bulamaz**, her seferinde gevşetme tetiklenir. |

Telemetri yoksa yapısal testlerinizi **`vehicle_count` üzerinden** kurun —
o YOLO'dan geliyor, telemetriye bağlı değil.

Telemetri varsa `over_sea` için ayrıca kıyı poligonu gerekiyor:
```bash
# .env
COASTLINE_GEOJSON=/veri/land_polygons.geojson
```
Verilmezse `over_sea` `None` kalır (diğer alanlar çalışmaya devam eder).

---

## 5. KALİBRASYON — tek videoyla (en kritik adım)

```bash
python -m scripts.register_video kalib1 /veri/ornek_video.mp4
python -m scripts.ingest_video kalib1 kalib1/raw.mp4 --local
```

Çıktının sonu:
```
Toplam sure     : XXXs (N.NNx gercek-zaman)
Embedding suresi: XXXs (N.NNx gercek-zaman)
```

**Bu rakam denemenizin temeli.** proje-ozeti.md §8 embedding için 40x
gerçek-zaman varsayıyor ve bu **doğrulanmadı** — ölçtüğünüz sayıyı §8'e işleyin.

### Batch ayarı (throughput'un en büyük belirleyicisi)

`EMBEDDING_BATCH_SIZE`'ı yükseltip tekrar ölçün. Model bf16'da ~4,3 GB:

| VRAM | Başlangıç | Deneyin |
|---|---|---|
| 8 GB | 4 | 6, 8 |
| 12-16 GB | 12 | 16, 24 |

```bash
EMBEDDING_BATCH_SIZE=8 python -m scripts.ingest_video kalib1 kalib1/raw.mp4 --local
```

`CUDA out of memory` alırsanız bir kademe düşürün. En iyi değeri `.env`'e yazın.

### Bütçe hesabı

```
islenebilir video suresi = olculen_kat x ayirabildiginiz_saat
```

Örnek: 10x ölçtünüz, 4 saat ayırdınız → ~40 saat video.

### Hedef korpus büyüklüğü

Önemli olan video sayısı değil, **pencere sayısı** (8sn/8sn → saat başına 450):

| Video süresi | Pencere | Ne test eder |
|---|---|---|
| ~2 saat | ~900 | Sadece mekanik doğrulama — her sorgu her şeyi bulur |
| **20-50 saat** | **9K-22K** | **İyi denge** — HNSW anlam kazanır, filtre gerçekten ayırt eder |
| 200+ saat | 90K+ | Ölçek davranışının ilk gerçek sinyali |

---

## 6. Toplu yükleme

### 6a. Önce Temporal yolunu tek videoyla sınayın

```bash
# 1. terminal
python -m ingest.worker

# 2. terminal
python -m scripts.register_video temporal_test /veri/baska_video.mp4
python -m scripts.ingest_video temporal_test temporal_test/raw.mp4
```

Temporal UI'da (<http://localhost:8080>) workflow'un tamamlandığını görün.
Bu yol `--local`'dan farklı (retry, checkpoint, dağıtım) ve **daha önce
hiç çalıştırılmadı** — büyük yüklemeden önce burada doğrulayın.

### 6b. Otomatik tetikleme (isteğe bağlı)

```bash
python -m scripts.setup_minio_notifications
python -m ingest.kafka_consumer      # 3. terminal
```

Artık `raw-videos` bucket'ına düşen her video kendiliğinden ingest edilir.
**Bu zincir de hiç test edilmedi** — 6a çalıştıktan sonra deneyin.

### 6c. Veriyi yükleyin

```bash
python -m scripts.register_video --dir /veri/ucuslar/
```

Kafka tüketicisi çalışıyorsa ingest kendiliğinden başlar. Çalışmıyorsa her
video için `ingest_video` çağırın.

**Çok GPU'nuz varsa:** her GPU için bir worker çalıştırın —
`CUDA_VISIBLE_DEVICES=0 python -m ingest.worker`, `=1` ... Temporal işi
aralarında dağıtır.

### İzleme

```bash
# Kac nokta yazildi
python -c "
from common.qdrant_store import get_client
from common import config
i = get_client().get_collection(config.QDRANT_COLLECTION)
print(i.points_count, 'nokta |', i.status)
"

docker stats --no-stream          # bellek/CPU baskisi
```

---

## 7. vLLM — sorgu tarafı (yapısal ayrıştırma)

### Önce: hangi ortamdasınız?

vLLM **Windows'u desteklemiyor** — resmi platform listesi Linux tabanlı
(NVIDIA CUDA / ROCm / Intel XPU / Apple Silicon). Windows'ta `pip install vllm`
hazır paket bulamayıp kaynaktan derlemeye çalışır ve başarısız olur (denendi).

| Ortamınız | Ne yapmalı |
|---|---|
| **Linux sunucu** | `pip install uv && uv pip install -r requirements-serving.txt --torch-backend=auto` |
| **Colab / Kaggle** | Zaten Linux — doğrudan kurulur |
| **Windows + Docker** | `docker compose --profile gpu up -d vllm` (konteyner Linux, altta WSL2) |
| **Windows + WSL2** | WSL içine kurun, host'tan `localhost:8000` üzerinden erişilir |
| **Windows, çıplak** | Mümkün değil — yukarıdaki iki yoldan birini seçin |

### Kurulum

```bash
pip install uv
uv pip install -r requirements-serving.txt --torch-backend=auto

vllm serve Qwen/Qwen2.5-7B-Instruct-AWQ \
    --guided-decoding-backend xgrammar \
    --gpu-memory-utilization 0.85 --max-model-len 8192

python -m scripts.check_env                  # "vLLM erisilebilir" gormeli
```

> Düz `pip install vllm` en güncel paketi çekip CUDA 13 runtime bekleyerek
> `libcudart.so.13: cannot open shared object file` hatasıyla çökebilir —
> vLLM'in bilinen bir paketleme sorunu (GitHub #43435, "not planned"
> kapatıldı). `uv --torch-backend=auto` mevcut CUDA sürümünü tespit edip
> uyumlu paketi seçiyor. Otomatik tespit başarısız olursa elle belirtin:
> `--torch-backend=cu128` (`nvidia-smi` çıktısındaki CUDA sürümüyle eşleşen).

**Tek GPU'daysanız: ingest BİTTİKTEN SONRA başlatın.** İkisi aynı anda VRAM'i
böler, ikisi de yavaşlar veya OOM olur.

> vLLM kendi torch sürümünü çekebilir — kurduktan sonra `check_env`'i tekrar
> çalıştırıp torch'un hâlâ CUDA derlemesi olduğunu doğrulayın.

### vLLM olmadan ne kaybedersiniz

**Arama durmaz.** vLLM erişilemezse sorgu ayrıştırıcı zarifçe semantik metne
düşer (canlı doğrulandı). Kaybedilen tek şey **yapısal ayrıştırma**: "gece",
"deniz üzerinde", "3 tekne" gibi ifadeler filtreye dönüşmez, hepsi vektör
aramasına gider.

Yani vLLM'i kuramıyorsanız da embedding kalitesini, gevşetme mekanizmasını
(elle `StructuredFilters` kurarak) ve hız ölçümlerini test edebilirsiniz —
sadece §3.2 madde 1 doğrulanmamış kalır.

---

## 8. Sorgu testleri

```bash
python -m scripts.query_cli --interactive
```

Sırayla deneyin:

| # | Prompt | Sınadığı | Beklenen |
|---|---|---|---|
| 1 | `a boat moving fast on open water` | Saf semantik | `Yapisal filtre: filtre yok` |
| 2 | `en az 3 tekne görünen kayıtlar` | Yapısal (YOLO) | `min_vehicle_count=3`, gevşetme YOK |
| 3 | `kalabalık bir sahilde 5'ten fazla tekne` | Hibrit | Hem filtre hem semantik metin |
| 4 | `20 tekne olan görüntüler` | **Gevşetme** | 0 → filtre düşer → `[yaklasik]` işaretli |
| 5 | `gün batımında deniz üzerinde tekne` | Telemetri yolu | Telemetri varsa filtre; yoksa gevşetme |

### Zor-negatif çifti (§7'nin özellikle istediği)

```
gün batımında kıyıya yaklaşan tekne
gün doğumunda kıyıya yaklaşan tekne
```

Görsel olarak neredeyse aynı, anlamca zıt. **İkisi de aynı sonucu
döndürüyorsa** embedding modeli bu ayrımı yapamıyor demektir — ve rerank tam
bunun için var:

```bash
python -m scripts.query_cli "gün batımında kıyıya yaklaşan tekne" --rerank
```

Rerank'li/rerank'siz farkı, §3.2 madde 4'ün değerini ölçen tek gerçek test.

---

## 9. Neye bakmalı

Sonuç listesine değil, şu üçüne:

1. **`gerçek-zaman katı`** — §8'in 40x varsayımından ne kadar uzak? Bu rakam
   tüm kapasite planlamasının temeli.
2. **`[yaklasik]` işareti ve `Filtre gevsetildi:` satırı** — doğru alanlar mı
   düşüyor? Gereğinden erken mi tetikleniyor?
3. **`Yapisal filtre:` satırı** — vLLM sorguyu doğru ayrıştırdı mı? "gece"
   yazınca `is_night=true` mi çıkıyor, yoksa alakasız alan mı doluyor?
   **Bu hat hiç test edilmedi, ilk gerçek sınavı burada.**

### Gecikme kırılımı — "vLLM ne kadar yavaşlatıyor" sorusunun cevabı

Her sorgudan sonra basılan satır:

```
1 aralik (5649ms)
  gecikme: parse=17ms embed=5533ms qdrant=25ms merge=0ms
```

| Alan | Ne | Ölçekle büyür mü |
|---|---|---|
| `parse` | vLLM'in yapısal/semantik ayrımı — **sorgu başına tek çağrı** | Hayır, sabit |
| `embed` | Sorgu metninin vektöre çevrilmesi | Hayır, sabit |
| `qdrant` | Filtreli HNSW; gevşetme olursa her adım eklenir | **Evet** |
| `rerank` | Aday başına bir VLM çağrısı (varsayılan kapalı) | Aday sayısıyla |

**vLLM'siz koşup `parse` değerini not edin, sonra vLLM'i açıp tekrar bakın** —
aradaki fark yapısal ayrıştırmanın gerçek maliyeti. §8'in 300ms tahmini hiç
ölçülmedi; bu kırılım onu gerçek veriyle değiştirmek için var.

Yukarıdaki örnek bu depodan (GT1030, CPU torch, vLLM kapalı): `qdrant=25ms`
ile arama katmanı hızlı, `embed=5533ms` CPU olduğu için baskın. GPU'da embed
düşecek ve `parse` görünür hale gelecek.

---

## 10. Sorun giderme

| Belirti | Sebep / çözüm |
|---|---|
| `CUDA out of memory` | `EMBEDDING_BATCH_SIZE` düşürün. vLLM aynı anda çalışıyorsa kapatın. |
| Embedding çok yavaş | `check_env` — torch CPU derlemesi olabilir. |
| Tüm sorgular aynı sonucu veriyor | `qwen-vl-utils` eski olabilir → embedding placeholder döner. `check_env` yakalar. Elle kontrol: iki farklı metnin kosinüs benzerliği 1.0 çıkarsa bozuk. |
| Her sorguda gevşetme tetikleniyor | Telemetri yok (Adım 4). `vehicle_count` ile test edin. |
| `Cannot load nvEncodeAPI64.dll` | GPU'da NVENC yok; kod yazılıma düşer, sorun değil. |
| Qdrant 100K+ üzerinde yavaş | Bilinen açık konu — bkz. docs/worklog_2026-07-28.md. Ölçüp raporlayın. |
| Temporal workflow takılı | Temporal UI'da hata detayı. Worker çalışıyor mu? |
| MinIO'ya yükleme yavaş | Ham videoyu MinIO'ya koymak zorunda değilsiniz; yerel yol da verilebilir. |

---

## 11. Sonuçları kaydedin

Denemeden sonra bu üç şey güncellenmeli:

1. **proje-ozeti.md §8** — ölçülen gerçek-zaman katı ve gecikme rakamları.
   §8'in en kritik iki varsayımı bunlar ve hâlâ doğrulanmamış durumdalar.
2. **docs/** — yeni bir worklog: ne çalıştı, ne çalışmadı, hangi rakamlar.
3. **Zor-negatif sonucu** — model bu ayrımı yapabiliyor mu? §5'in (model
   seçimi) hâlâ açık olan kısmına ilk gerçek veri bu olur.
