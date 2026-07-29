# Sıfırdan Kurulum ve Deneme Rehberi

**Varsayım: makinede hiçbir şey kurulu değil.** NVIDIA sürücüsünden başlayıp
çalışan bir arama sistemine kadar her adım burada.

> **Aceleniz varsa bu rehbere ihtiyacınız olmayabilir.** Ön koşullar
> (`nvidia-smi` + `git ffmpeg python3-venv docker`) hazırsa dört komut yeter:
> ```bash
> ./scripts/setup.sh
> source .venv/bin/activate
> python -m scripts.ingest_all --dir ~/videolar/
> ./scripts/start_vllm.sh     # ayri terminal
> python -m scripts.query_cli --interactive
> ```
> Bu rehber o adımların **ne yaptığını**, neyin neden gerektiğini ve ters
> giderse ne yapacağınızı anlatıyor. Sorun çıkmazsa okumanıza gerek yok.

**Kapsam:**
- Tam pipeline (ingest + sorgu)
- **vLLM yapısal filtreleme dahil** — sorgu aşamasında sürekli açık
  (ingest sırasında kapalı, gerekçesi 0. bölümde)
- **Rerank opsiyonel** — 12. bölüm, ekstra VRAM istiyor
- Birkaç GB'lık veriyle deneme (7. bölümde ne anlama geldiği açıklanıyor)

---

## 0. Önce bilmeniz gereken üç şey

**1. vLLM Windows'ta çalışmaz.** Resmi platform listesinde Windows yok;
`pip install vllm` Windows'ta kaynaktan derlemeye çalışıp başarısız oluyor
(denendi). Bu rehber **Ubuntu Linux** varsayıyor. Windows'taysanız 1. bölümde
WSL2 kuruyoruz — WSL içi Ubuntu ile birebir aynı, sonraki adımlar değişmiyor.

**2. İki aşama var ve aynı anda çalışmaları GEREKMİYOR.**

| Aşama | Gereken modeller | vLLM? |
|---|---|---|
| **Ingest** — bir kez, uzun toplu iş | embedding + YOLO | ❌ gerekmiyor |
| **Sorgu servisi** — sürekli | embedding + ayrıştırıcı | ✅ gerekiyor |

vLLM sorgu ayrıştırması **her sorguda** çalışır, o yüzden sorgu servisi
ayaktayken modelin yüklü kalması gerekir (5 GB'lık ağırlığı sorgu başına
yüklemek onlarca saniye sürer, kabul edilemez). Ama **ingest sırasında
kapalı olmalı** — hem gereksiz hem zararlı: VRAM'i bölüp embedding batch'ini
küçültür, ingest'i yavaşlatır.

**Bu rehberde sıra: önce ingest (vLLM kapalı) → sonra vLLM → sorgu testleri.**

Gerçek ağırlık boyutları (HuggingFace'ten doğrulandı):

| Model | Ağırlık | Rolü |
|---|---|---|
| Qwen3-VL-Embedding-2B | 3.96 GB | embedding (her iki aşamada) |
| Qwen2.5-3B-Instruct-AWQ | 2.50 GB | ayrıştırıcı (küçük) |
| Qwen2.5-7B-Instruct-AWQ | 5.19 GB | ayrıştırıcı (büyük) |
| Qwen2.5-VL-7B-Instruct-AWQ | 6.45 GB | ayrıştırıcı **+ rerank** (tek model, iki iş) |

Sorgu aşamasında ikisi birlikte yüklü olur:

| VRAM | Ayrıştırıcı | Rerank |
|---|---|---|
| **8 GB** | 7B-AWQ sıkışık (3.96+5.19=9.2 GB → sığmaz); **3B-AWQ** kullanın (6.5 GB) | ❌ |
| **12 GB** | 7B-AWQ rahat (9.2 GB) | ⚠️ VL-7B ile denenebilir |
| **16 GB+** | 7B-AWQ ya da VL-7B | ✅ VL-7B tek modelde ikisi |

Bunlar sadece ağırlıklar — KV cache ve ara tensörler için ~%20 pay bırakın.

> **Üretimde bu sorun yok:** ingest worker'ları ve sorgu sunucusu ayrı
> makinelerde olur (proje-ozeti.md §3.1 — N worker = N GPU). Tek makinede
> deneme yaptığımız için sıralama gerekiyor.

**3. Birkaç GB veri az bir veridir.** 7. bölümde neden ve ne beklemeniz
gerektiği yazıyor. Kısacası: hız ölçümü ve mekanik doğrulama için yeterli,
arama kalitesi değerlendirmesi için değil.

---

## 1. İşletim sistemi

### Zaten Ubuntu/Debian iseniz
Bu bölümü atlayın, 2'ye geçin.

### Windows iseniz — WSL2

PowerShell'i **yönetici olarak** açın:

```powershell
wsl --install -d Ubuntu-24.04
```

Bilgisayarı yeniden başlatın. Ubuntu ilk açılışta kullanıcı adı/parola ister.
**Bundan sonraki TÜM komutlar Ubuntu terminalinde çalıştırılacak** (Başlat →
Ubuntu).

WSL2 GPU'yu görüyor mu — Ubuntu terminalinde:
```bash
nvidia-smi
```
GPU görünmüyorsa Windows tarafında güncel NVIDIA sürücüsü kurun
(<https://www.nvidia.com/Download/index.aspx>). WSL için ayrıca CUDA
kurmanıza **gerek yok** — Windows sürücüsü yeterli.

> WSL2'ye varsayılan olarak host RAM'inin yarısı verilir. 32 GB RAM'iniz
> varsa 16 GB'a düşmüş olursunuz. Artırmak için Windows'ta
> `C:\Users\<siz>\.wslconfig` oluşturun:
> ```ini
> [wsl2]
> memory=24GB
> ```
> sonra PowerShell'de `wsl --shutdown`.

---

## 2. NVIDIA sürücüsü doğrulaması

```bash
nvidia-smi
```

Şuna benzer bir çıktı görmelisiniz:
```
| NVIDIA GeForce RTX 4060 ...  |  8188MiB |
```

**Göremiyorsanız** (saf Ubuntu'da):
```bash
sudo ubuntu-drivers autoinstall
sudo reboot
```

CUDA Toolkit'i **ayrıca kurmanıza gerek yok** — PyTorch kendi CUDA
kütüphanelerini paketliyor. Sadece sürücü lazım.

---

## 3. Sistem paketleri

```bash
sudo apt update
sudo apt install -y git ffmpeg python3-venv python3-pip curl
```

Python sürümünü kontrol edin (3.11+ gerekiyor):
```bash
python3 --version
```

3.11'den eskiyse (ör. Ubuntu 22.04 → 3.10):
```bash
sudo add-apt-repository -y ppa:deadsnakes/ppa
sudo apt update
sudo apt install -y python3.11 python3.11-venv
```
ve aşağıda `python3` yerine `python3.11` kullanın.

### Docker (altyapı servisleri için)

```bash
curl -fsSL https://get.docker.com | sudo sh
sudo usermod -aG docker $USER
newgrp docker          # ya da oturumu kapatip acin
docker run --rm hello-world
```

> **NVIDIA Container Toolkit'e gerek yok.** Bu rehberde Docker sadece
> CPU servisleri (Qdrant, MinIO, Postgres, Kafka, Temporal) için; GPU işleri
> (embedding, YOLO, vLLM) doğrudan host'ta çalışacak. Bu, kurulumu ciddi
> ölçüde basitleştiriyor.

---

## 4. Depo ve Python bağımlılıkları

```bash
git clone https://github.com/ykyking1/VideoAnalysis.git
cd VideoAnalysis

python3 -m venv .venv
source .venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt
```

`requirements.txt` CUDA'lı PyTorch'u PyTorch'un kendi index'inden çeker
(`--extra-index-url .../cu126`). RTX 50xx/Blackwell kartınız varsa dosyanın
ilk satırındaki `cu126`'yı `cu128` yapın.

### vLLM (yapısal filtreleme)

Kurulumu şimdi yapıyoruz ama **sunucuyu 10. bölümde başlatacağız** —
ingest sırasında açık olması VRAM'i boşuna bölerdi.

```bash
pip install -r requirements-serving.txt
```

---

## 5. Ortam doğrulaması — ATLAMAYIN

```bash
python -m scripts.check_env --skip-services
```

**Görmeniz gerekenler:**
```
[OK  ] torch 2.x.x+cu126 | GPU: NVIDIA GeForce RTX 4060 (8.0 GB, compute 8.9)
[OK  ] bfloat16 Tensor Core destekli (Ampere+)
[OK  ] transformers 4.5x
[OK  ] qwen-vl-utils 0.0.14
[OK  ] ffmpeg mevcut (NVDEC/cuda: evet, hevc_nvenc derlemede: evet)
```

**`[HATA] torch ... CPU-only` görürseniz DURUN.** Sistem çalışır ama ~15-20
kat yavaş olur ve bunu ancak saatler sonra fark edersiniz:
```bash
pip install torch torchvision --force-reinstall \
    --index-url https://download.pytorch.org/whl/cu126
```

> vLLM kendi torch sürümünü çekmiş olabilir — bu yüzden `check_env`'i
> vLLM kurulumundan **sonra** çalıştırıyoruz.

---

## 6. Altyapı servisleri

```bash
cp .env.example .env
nano .env          # POSTGRES_PASSWORD ve MINIO_ROOT_PASSWORD'u degistirin

docker compose up -d
docker compose ps  # hepsi Up olmali
```

Ayağa kalkanlar: MinIO (nesne deposu), Qdrant (vektör), Postgres (durum),
Kafka + Temporal (orkestrasyon).

```bash
python -m scripts.init_storage
```

Beklenen:
```
[depo] minio (localhost:9000) - bucket'lar hazir: raw-videos, proxy-videos
[qdrant] koleksiyon 'clips' hazir (boyut=2048, nokta=0, kuantizasyon=scalar, on_disk=True)
[qdrant] payload indeksleri: agl_m, avg_speed_kmh, caption, over_sea, ...
```

Arayüzler: MinIO <http://localhost:9001>, Qdrant <http://localhost:6333/dashboard>,
Temporal <http://localhost:8080>.

---

## 7. Veri — "birkaç GB" ne demek

### Önce dürüst bir uyarı

**GB yanlış birim.** Önemli olan **süre**, çünkü sistem videoyu 8 saniyelik
pencerelere bölüyor ve arama bu pencereler arasında yapılıyor.

Aynı 5 GB, bitrate'e göre çok farklı süreler demek:

| Kaynak | Bitrate | 5 GB ≈ | Pencere sayısı |
|---|---|---|---|
| DJI 1080p (yüksek kalite) | ~60 Mbps | ~11 dk | ~85 |
| Tipik 1080p H.264 | ~20 Mbps | ~33 dk | ~250 |
| Sıkıştırılmış / 720p | ~5 Mbps | ~2.2 saat | ~1000 |

**Birkaç GB ile ne yapabilirsiniz:**

| Amaç | Yeterli mi |
|---|---|
| Pipeline uçtan uca çalışıyor mu | ✅ Evet |
| Embedding hızı ölçümü (§8'in en kritik açığı) | ✅ Evet |
| vLLM yapısal ayrıştırma doğru mu | ✅ Evet |
| Filtre gevşetme mekanizması | ⚠️ Kısmen (aşağıya bakın) |
| Arama kalitesi / Recall | ❌ **Hayır** — binlerce pencere gerekir |

> **Küçük korpusta gevşetme her zaman tetiklenir** — bu hata değil. Eşik
> `SEARCH_MIN_RESULTS=5`; korpusta yeterli pencere yoksa hiçbir filtre 5
> sonuç bulamaz. Az veriyle test ederken `.env`'e `SEARCH_MIN_RESULTS=1`
> yazın.

### Veriyi yerleştirme

```bash
mkdir -p ~/videolar
# Videolarinizi buraya kopyalayin.
```

Windows'tan WSL'e kopyalıyorsanız — **Windows diskinden doğrudan okumayın**,
yavaş ve bazen kısmi okuma yapar:
```bash
cp /mnt/c/Users/<siz>/Videolar/*.mp4 ~/videolar/
```

Süreyi ve sağlamlığı kontrol edin:
```bash
for f in ~/videolar/*; do
  printf "%-40s %s\n" "$(basename "$f")" \
    "$(ffprobe -v error -show_entries format=duration -of csv=p=0 "$f" 2>&1)"
done
```
Süre yerine hata görüyorsanız o dosya bozuk (yükleme yarım kalmış olabilir).
Sistem bunu zaten kayıt anında yakalıyor ama önden bilmek iyidir.

### Disk alanı

Ölçülen oranlar (bu depodan):

| Ne | Saat başına |
|---|---|
| Proxy (360p HEVC) | ~360 MB |
| Vektörler (2048d, int8) | ~15 MB |

Ham videolar MinIO'ya da kopyalanır — yani ham videonun **iki kopyası** olur.
5 GB veri için ~11 GB boş alan bulundurun.

### Telemetri (varsa)

`.tlog` dosyalarınız varsa `--telemetry` ile verin; `agl_m`, `avg_speed_kmh`,
`over_sea`, `sun_elevation` alanları dolar ve "gece", "deniz üzerinde",
"50 metre üstünde" gibi filtreler **gerçekten** test edilebilir.

Telemetri yoksa bu alanlar `None` kalır ve o filtreler hiç eşleşme bulamaz
(gevşetme devreye girer). O durumda yapısal testlerinizi **`vehicle_count`**
üzerinden kurun — o YOLO'dan geliyor, telemetriye bağlı değil.

`over_sea` için ayrıca kıyı poligonu gerekiyor:
```bash
echo "COASTLINE_GEOJSON=/yol/land_polygons.geojson" >> .env
```

---

## 8. KALİBRASYON — tek videoyla (en kritik adım)

**Toplu yüklemeden önce mutlaka bu.**

> **vLLM'i henüz başlatmayın.** Bu ve sonraki bölüm (ingest) vLLM
> gerektirmiyor; kapalı olması tüm VRAM'i embedding'e bırakır, daha büyük
> batch kullanabilirsiniz ve ingest daha hızlı biter. vLLM 10. bölümde.

```bash
VIDEO=$(ls ~/videolar/* | head -1)
python -m scripts.register_video kalib1 "$VIDEO"
python -m scripts.ingest_video kalib1 kalib1/raw.mp4 --local
```

Çıktının sonu:
```
Embedding suresi: XXXs (N.NNx gercek-zaman)
```

**Bu sayı denemenizin en değerli çıktısı.** proje-ozeti.md §8 burada 40x
gerçek-zaman varsayıyor ve bu varsayım **doğrulanmadı** — ölçtüğünüzü §8'e
işleyin.

### Batch ayarı (throughput'un en büyük belirleyicisi)

```bash
EMBEDDING_BATCH_SIZE=16 python -m scripts.ingest_video kalib1 kalib1/raw.mp4 --local
```

8 GB'de 4 → 8, 16 GB'de 12 → 16 → 24 deneyin. `CUDA out of memory` alırsanız
bir kademe düşün. En iyi değeri `.env`'e yazın:
```bash
echo "EMBEDDING_BATCH_SIZE=16" >> .env
```

> vLLM aynı GPU'da çalışıyor ve OOM alıyorsanız: ya vLLM'i geçici durdurun
> ya `--gpu-memory-utilization`'ı düşürün ya da batch'i küçültün.

---

## 9. Toplu ingest

### Önce Temporal yolunu tek videoyla sınayın

```bash
# Terminal 2 (venv aktif)
python -m ingest.worker
```
```bash
# Terminal 1
python -m scripts.register_video test2 ~/videolar/<baska_video>.mp4
python -m scripts.ingest_video test2 test2/raw.mp4
```

Temporal UI'da (<http://localhost:8080>) workflow'un tamamlandığını görün.
Bu yol `--local`'dan farklı (retry, checkpoint, dağıtım).

### Tüm veriyi yükleyin — tek komut

```bash
python -m scripts.ingest_all --dry-run --dir ~/videolar/   # once plani gorun
python -m scripts.ingest_all --dir ~/videolar/
```

`ingest_all` kaydetme ve ingest'i birlikte yapar, modeli **bir kez** yükler
(her video için ayrı süreç video başına ~1 dakikayı modeli yeniden yüklemeye
harcıyordu). Ayrıca:

- **Zaten ingest edilmiş videoları atlar** — yarıda kesilirse aynı komutla
  kaldığı yerden devam eder (`--force` ile yeniden işler)
- **Bozuk dosyaları atlar ve sonda raporlar** — tek bozuk dosya tüm
  yüklemeyi düşürmez
- Sonda **genel gerçek-zaman katını** basar

`--limit N` ile ilk N videoyla sınırlayabilirsiniz.

### Otomatik tetikleme (opsiyonel)

```bash
python -m scripts.setup_minio_notifications
python -m ingest.kafka_consumer     # Terminal 3
```
Artık `raw-videos` bucket'ına düşen her video kendiliğinden ingest edilir.

### İzleme

```bash
python -c "
from common.qdrant_store import get_client
from common import config
i = get_client().get_collection(config.QDRANT_COLLECTION)
print(i.points_count, 'pencere |', round(i.points_count*8/3600, 2), 'saat')
"
```

---

## 10. vLLM sunucusu (yapısal filtreleme)

> **Bu adım ingest BİTTİKTEN SONRA.** vLLM ingest sırasında gerekmez ve
> açık olursa VRAM'i bölüp embedding batch'ini küçültür — ingest'i
> yavaşlatır. Sıra: ingest (vLLM kapalı) → vLLM başlat → sorgu testleri.

vLLM'in tek işi kullanıcı sorgusunu **yapısal filtre + semantik metne**
ayırmak:

```
"gece deniz uzerinde 3 tekne"
   ↓ vLLM + xgrammar (sema zorlamali)
{is_night: true, over_sea: true, min_vehicle_count: 3} + semantik: "tekne"
   ↓ query/filter_builder.py
Qdrant filtresi
```

Model **sorgu metni yazmıyor**, sadece şemadaki alanları dolduruyor; sorguyu
kod kuruyor. Bu sayede geçersiz sorgu ya da enjeksiyon üretilemiyor.

Ayrıştırma **her sorguda** çalışır, o yüzden sorgu servisi ayaktayken model
yüklü kalır — 5 GB ağırlığı sorgu başına yüklemek onlarca saniye sürerdi.

**VRAM'inize göre model seçin** (0. bölümdeki tablo):

```bash
# 8 GB VRAM
export PARSE_MODEL=Qwen/Qwen2.5-3B-Instruct-AWQ
export VLLM_GPU_FRAC=0.30

# 12-16 GB VRAM
export PARSE_MODEL=Qwen/Qwen2.5-7B-Instruct-AWQ
export VLLM_GPU_FRAC=0.45
```

Ayrı bir terminalde (venv aktifken) başlatın:

```bash
source .venv/bin/activate
vllm serve $PARSE_MODEL \
    --guided-decoding-backend xgrammar \
    --gpu-memory-utilization $VLLM_GPU_FRAC \
    --max-model-len 8192
```

> **`--gpu-memory-utilization` kritik.** vLLM varsayılan olarak GPU'nun
> %90'ını kendine ayırır ve embedding modeline yer bırakmaz. Yukarıdaki
> değerler embedding modeline (3.96 GB) pay bırakacak şekilde seçildi.

İlk çalıştırmada model indirilir (2.5-5 GB), birkaç dakika sürer.
`Application startup complete` görünce hazırdır.

**İlk terminalde** doğrulayın:
```bash
python -m scripts.check_env
```
`[OK  ] vLLM erisilebilir` görmelisiniz.

`.env`'e kalıcı yazın ki her seferinde export etmeyesiniz:
```bash
echo "PARSE_MODEL=$PARSE_MODEL" >> .env
```

---

## 11. Sorgu testleri

```bash
python -m scripts.query_cli --interactive
```

### Yapısal ayrıştırmanın çalıştığını doğrulayın

vLLM'in asıl işi bu. Her sorguda **`Yapisal filtre:` satırına bakın**:

| Sorgu | Beklenen `Yapisal filtre:` |
|---|---|
| `en az 3 tekne gorunen kayitlar` | `min_vehicle_count=3` |
| `gece deniz uzerinde tekne` | `is_night=True, over_sea=True` |
| `50 metreden yuksekte ucan arac` | `min_agl_m=50` |
| `a boat on the water` | **`filtre yok`** ← boş yere filtre üretmemeli |

**Son satır özellikle önemli.** Sorguda geçmeyen bir kavram için filtre
üretilirse doğru sonuçlar sessizce elenir — bu projede bu riskin gerçek
olduğu ölçüldü (bkz. proje-ozeti.md §8 eki).

### Gevşetmenin çalıştığını doğrulayın

```
20 tekne olan goruntuler
```
Beklenen: `! Filtre gevsetildi: min_vehicle_count dusuruldu` ve sonuçlar
`[yaklasik]` işaretli.

### Gecikme kırılımı

Her sorgudan sonra:
```
  gecikme: parse=812ms embed=45ms qdrant=18ms merge=0ms
```
`parse` = vLLM'in maliyeti. proje-ozeti.md §8'in 300ms tahmini hiç
ölçülmedi — bu sayıyı not edin.

### Zor-negatif çifti (en değerli kalite testi)

```
gun batiminda kiyiya yaklasan tekne
gun dogumunda kiyiya yaklasan tekne
```
Görsel olarak neredeyse aynı, anlamca zıt. **İkisi de aynı sonucu
döndürüyorsa** embedding modeli bu ayrımı yapamıyor demektir — §5'in
(model seçimi) açık kalan kısmına ilk gerçek veri bu olur.

---

## 12. Rerank (opsiyonel)

Rerank, VLM'in top adayları tek tek doğrulaması. Zor-negatif probleminin
çözümü olması bekleniyor ama **gecikmeyi ciddi artırıyor** (aday başına bir
VLM çağrısı) ve varsayılan olarak kapalı.

**VRAM gereksinimi:** üçüncü bir model demek. En verimli yol, ayrıştırıcı ve
rerank için **aynı VLM'i** kullanmak — Qwen2.5-VL hem metin hem görüntü
işleyebiliyor, yani tek sunucu iki işi de yapar:

```bash
# vLLM'i durdurun, VL modeliyle yeniden baslatin (16 GB+ onerilir)
vllm serve Qwen/Qwen2.5-VL-7B-Instruct-AWQ \
    --guided-decoding-backend xgrammar \
    --gpu-memory-utilization 0.55 --max-model-len 8192
```

```bash
# .env
PARSE_MODEL=Qwen/Qwen2.5-VL-7B-Instruct-AWQ
VLM_MODEL=Qwen/Qwen2.5-VL-7B-Instruct-AWQ
RERANK_ENABLED=false        # varsayilan kapali kalsin, --rerank ile ac
RERANK_CANDIDATES=10
```

Tek sorguda açıp farkı ölçün:
```bash
python -m scripts.query_cli "gun batiminda kiyiya yaklasan tekne" --rerank
```

Karşılaştırın: rerank'li ve rerank'siz aynı sorgu — sıralama düzeldi mi,
gecikme ne kadar arttı? Bu, §3.2 madde 4'ün değerini ölçen tek gerçek test.

**8 GB VRAM'de rerank'i atlayın** — üç model sığmaz.

---

## 13. Sonuçları kaydedin

Denemeden sonra bunları not edin; proje-ozeti.md §8 bunları bekliyor:

1. **Embedding gerçek-zaman katı** (batch değeriyle) — §8'in 40x varsayımı
   ne kadar uzak?
2. **`parse=...ms`** — vLLM ayrıştırmanın gerçek gecikmesi.
3. **Yapısal ayrıştırma doğruluğu** — LLM alanları doğru dolduruyor mu,
   boş yere filtre üretiyor mu?
4. **Zor-negatif sonucu** — sunset/sunrise ayrımı yapılabiliyor mu?
5. **Rerank denendiyse** — kaliteyi artırdı mı, gecikme bedeli ne?

---

## 14. Sorun giderme

| Belirti | Sebep / çözüm |
|---|---|
| `nvidia-smi` çalışmıyor (WSL) | Windows tarafında NVIDIA sürücüsünü güncelleyin; WSL'e CUDA kurmayın |
| `[HATA] torch CPU-only` | `pip install torch torchvision --force-reinstall --index-url .../cu126` |
| `CUDA out of memory` (ingest) | `EMBEDDING_BATCH_SIZE` düşürün; vLLM'in `--gpu-memory-utilization`'ını azaltın |
| vLLM başlarken OOM | `--gpu-memory-utilization` düşürün ya da daha küçük model (3B-AWQ) |
| Tüm sorgular aynı sonucu veriyor | `qwen-vl-utils` eski olabilir → `check_env`. Elle: iki farklı metnin kosinüsü 1.0 çıkarsa embedding bozuk |
| Her sorguda gevşetme | Korpus küçük (7. bölüm) ya da telemetri yok → `SEARCH_MIN_RESULTS=1` |
| `moov atom not found` | Video dosyası eksik/bozuk — kopyalamayı tekrarlayın |
| `Yapisal filtre: filtre yok` (hep) | vLLM erişilemiyor → `check_env`, sunucu terminaline bakın |
| Docker izin hatası | `sudo usermod -aG docker $USER` sonra `newgrp docker` |
| Qdrant 100K+ üzerinde yavaş | Bilinen açık konu — bkz. docs/worklog_2026-07-28.md |

**İlk komut her zaman:** `python -m scripts.check_env`
