# İHA Video Arşivinde Semantik Arama — Proje Özeti

> Bu doküman, önceki tasarım/danışmanlık konuşmasının çıktısıdır. Amaç: doğal dil sorgusu → video kimliği + zaman aralığı listesi döndüren bir hibrit arama sistemi kurmak. Aşağıdaki bilgiler mimari kararları, gerekçelerini ve **henüz doğrulanmamış varsayımları** ayrı ayrı işaretleyerek özetler. Implementasyona başlamadan önce "Doğrulanmamış Varsayımlar" bölümü mutlaka okunmalı.

## 1. Proje Hedefi

Kullanıcı doğal dilde bir sorgu yazıyor (örn. *"günbatımında deniz üzerinde yüksek hızlarda uçan bir TB2 videosu"*), sistem buna karşılık ilgili videoların zaman aralıklarını metin olarak döndürüyor:

```
video1 0:02:10 – 0:03:12
video3 1:34:28 – 1:47:02
```

**Önizleme/oynatma gereksinimi yok** — çıktı sadece video kimliği + zaman aralığı. Bu kısıt, mimarideki proxy/thumbnail kararlarını doğrudan etkiliyor (bkz. §3).

## 2. Mevcut Veri Altyapısı

| Bileşen | Rol |
|---|---|
| MinIO | Ham video (~1,5 PB varsayılan — **doğrulanmadı**, bkz. §8), model proxy'leri |
| PostgreSQL | Video/görev kimlik kayıtları, ingest durum takibi (state machine) |
| ~~ClickHouse~~ → **Qdrant** | Klip düzeyinde arama — vektör + telemetri + metadata aynı noktada (payload) |

> **GÜNCELLEME (2026-07-29):** Arama katmanı ClickHouse'tan Qdrant'a taşındı.
> Gerekçe ölçüme dayanıyor: ClickHouse'ta `WHERE` + vektör `ORDER BY` birlikte
> kullanıldığında `prefilter` stratejisi HNSW indeksini tamamen devre dışı
> bırakıyor (`EXPLAIN indexes=1` ile doğrulandı: granül budama 72/72 = hiç yok),
> `postfilter` ise LIMIT'ten az sonuç dönebiliyor. Qdrant filtreyi HNSW graf
> gezinmesinin içinde uyguluyor — 100K korpusta `exact=True` brute-force ile
> 21/21 birebir aynı top-3 verdi ve ~1.5x daha hızlıydı. Detay: docs/worklog_2026-07-28.md.
>
> **Çekince:** 1M ölçekte hem ClickHouse hem Qdrant test ortamında (Docker
> Desktop/Windows, ~7,6GB paylaşımlı VM belleği) ciddi yavaşladı; kök neden
> kesinleşmedi. Gerçek donanımda yeniden ölçülmeli.

## 3. Sistem Mimarisi

### 3.1 Ingest Hattı (video girişi)

Tetik: MinIO bucket notification → Kafka → Temporal workflow (checkpoint + heartbeat ile dayanıklı, çöken GPU işi kaldığı pencereden devam eder).

1. **Model proxy üretimi** — ffmpeg + NVDEC, 240-360p HEVC. Önizleme için değil, model tüketimi (embedding/detektör/rerank) ve gelecekteki backfill'lerin ucuz decode'u için. Kalıcı tutulması opsiyonel (bkz. §4, JIT alternatifi).
2. **Telemetri işleme** — pymavlink + polars ile `.tlog`/MAVLink log ayrıştırma. Video 60 sn pencere / 60 sn kaydırma ile bölünür (**chunking yöntemi: sabit-uzunluk, örtüşmesiz** — 2026-07-29'da %50 örtüşmeli 4sn kaydırmadan değiştirildi, gerekçe: gerçek envanterde (§8) vektör sayısını yarıya indirmek; bkz. §9 için iyileştirme önerisi). Pencere boyutunun kendisi (8sn→60sn) **2026-08-01'de, SINIRLI kanıtla** değiştirildi: birleştirilmiş 21 SeaDroneSee klibinden oluşan tek bir 914.8sn'lik videoda N=10 sorguyla ölçüldü, Recall@10 %20→%70 ve MRR 0.083→0.408 — yön güçlü ama video yapay birleştirme ve kısa/ani olayların 60sn'de kaybolup kaybolmadığı hiç test edilmedi (bkz. docs/worklog_2026-08-01.md, §9'daki çok-ölçekli pencereleme fikri bu riski adresleyebilir ama uygulanmadı). Her pencere için türetilmiş alanlar otomatik hesaplanır:
   - `avg_speed_kmh`, `agl_m` (irtifa)
   - `sun_elevation` (astral/pysolar — "gece/günbatımı" kavramlarının deterministik karşılığı)
   - `over_sea` (Shapely + GeoPandas point-in-polygon, OSM kıyı poligonları)
   - `sensor_type`, ham telemetri özeti (lat/lon/heading/gimbal — öngörülmeyen gelecekteki filtreler için sigorta)
3. **Klip embedding** — video-metin modeli, pencere başına tek vektör. **Model seçimi henüz kesinleşmedi**, bkz. §5.
4. **Terfi etmiş görsel alanlar** — YOLO26 (IR fine-tune), sorgu loglarından sık talep gören ve deterministik çözülebilen görsel kavramları (örn. `vehicle_count`) kolonlaştırır. Katalog kullanım verisine göre organik büyür.
5. **Seçici caption** — Qwen2.5-VL (vLLM üzerinde), ffmpeg sahne değişim skoruna göre seçilen ~%10'luk "olay penceresi"ne kısa açıklama üretir; hibrit (vektör+tam metin) aramada kullanılır.
6. **Yazım** — ~~ClickHouse `clips` satırı~~ → tek Qdrant noktası: vektör + payload `(video_id, t_start, t_end, sensor_type, avg_speed_kmh, sun_elevation, over_sea, agl_m, vehicle_count, caption, ham telemetri...)`. Filtrelenebilir tüm alanlarda payload index, `caption`'da tam-metin index. Nokta kimliği `(video_id, t_start)`'tan deterministik türetilir → yeniden ingest idempotent.

### 3.2 Sorgu Hattı

1. **LLM ayrıştırma** — Qwen 14B + xgrammar (şema zorlamalı yapısal çıktı), SGLang üzerinde (tekrarlayan sistem promptu — alan kataloğu/eşleme tablosu — RadixAttention'dan fayda görür). Sorgu, yapısal filtrelere (telemetriden gelen, deterministik) ve semantik artığa (`semantic_text`, embedding modeline giden) ayrılır. Katalogda karşılığı olmayan kavramlar hataya değil, tamamen semantik aramaya düşer.
2. **Hibrit arama** — Qdrant tek sorguda: filtre HNSW graf gezinmesinin İÇİNDE uygulanır (payload index sayesinde), yani prefilter/postfilter ikilemi yok. **Sonuç `SEARCH_MIN_RESULTS` altına düşerse filtre kademeli gevşetilir** ve gevşetilmiş sonuçlar `exact_filter_match=False` ile işaretlenir — ölçtük ki dar/yanlış bir hard filtre doğru cevabı yapısal olarak dışlıyor (bkz. §8 notu ve docs/worklog_2026-07-28.md).
3. **Aralık birleştirme** — ardışık eşleşen pencereler ≤10 sn boşluk toleransıyla sürekli aralıklara birleştirilir.
4. **Opsiyonel rerank** — Qwen2.5-VL, top 10-20 adayı doğrular (embedding'in görsel benzerlik yanılgılarını ayıklar).

## 4. Depolama Stratejisi

- Orijinal video: **tamamen soğuk/arşiv katmana taşınabilir** — sorgu hattı hiçbir adımda videoya dokunmuyor.
- Proxy: 240-360p HEVC, ~25-35 TB (150 TB'lik ilk tahminden küçültülmüş; önizleme gereksinimi olmadığı için "insan kalitesi" değil "model kalitesi" yeterli).
- Thumbnail: kaldırıldı (önizleme yok, tüketicisi kalmadı).
- Arama katmanı (ClickHouse): embedding modeline bağlı, bkz. §5-6.

## 5. Model Seçimi — LİSANSLA DARALDI, KALİTE KARARI HÂLÂ AÇIK

> **GÜNCELLEME (2026-07-29):** Üç aday gerçek SeaDronesSee verisinde test edildi ve
> **belirleyici kriter lisans oldu**:
>
> | Model | Lisans | Durum |
> |---|---|---|
> | VideoCLIP-XL | CC-BY-NC-SA 4.0 | ✗ NonCommercial — ticari/savunma kullanımına kapalı |
> | EBind (`encord-team/ebind-audio-vision`) | CC-BY-NC-SA 4.0 | ✗ aynı sebeple elendi |
> | **Qwen/Qwen3-VL-Embedding-2B** | **Apache-2.0** | ✓ ticari kullanıma açık tek aday, 2048d, MRL (64-2048) destekli |
>
> Implementasyon Qwen3-VL-Embedding-2B ile yapıldı. Bu **kalite kararı değil,
> lisans zorunluluğudur** — "en iyi retrieval modeli" olduğu iddia edilmiyor.
> Aşağıdaki aday listesi ve golden set gereksinimi (§7) hâlâ geçerli;
> InternVideo2/VideoPrism/LanguageBind test edilmedi. Detay: docs/worklog_2026-07-28.md.

**Lisans elemesi öncesi durum:** X-CLIP (Ma vd. 2022, MIT lisans, `github.com/xuguohai/X-CLIP`) önde giden adaydı.

**Geçmiş:** İlk taslakta InternVideo2 6B önerilmişti (video-native mimari argümanıyla). Bağımsız bir sıfır-atış (zero-shot) kıyaslama (Mixpeek, açık kod+veri, BEIR/MTEB metodolojisi) InternVideo2'yi test edilen 6 modelin en kötüsü olarak ölçtü (NDCG@10=0,302). Bu, model seçiminin **doğrulanmadan rapora yazılmaması** gerektiğinin somut kanıtı oldu — X-CLIP bu ölçümde en iyi self-hosted seçenek olarak (NDCG@10=0,470) öne çıktı.

**Aday listesi (Adım 0'da test edilecek):**

| Model | Durum | Not |
|---|---|---|
| **X-CLIP** | Mevcut lider adayı | 512 boyut, MIT lisans, `microsoft/xclip` (Kinetics sınıflandırma) ile **karıştırılmamalı** — doğru repo `xuguohai/X-CLIP` (AOSM/retrieval varyantı) |
| **VideoCLIP-XL** | Yeni aday | `alibaba-pai/VideoCLIP-XL` (+ `-v2`), ViT-L/14 (X-CLIP'ten büyük, muhtemelen 768d — depolama kazancını geri alabilir), uzun açıklama/halüsinasyon-farkındalıklı sıralamada (HDR görevi) güçlü — bizim "günbatımı/gündoğumu" gibi ince ayrım ihtiyacımızla örtüşüyor |
| InternVideo2 (Stage-1 kontrol noktası) | Test edilmemiş | Mixpeek'in test ettiği Stage-2 kontrol noktası retrieval için optimize değildi; retrieval-özel Stage-1 farklı sonuç verebilir |
| VideoPrism, LanguageBind | Test edilmemiş | Google DeepMind / açık kaynak, kendi benchmark'larında güçlü iddialar var, bağımsız doğrulanmadı |

**Bilinen model-özgü kısıtlar:**
- X-CLIP, CLIP metin kodlayıcısını miras alıyor → **77 token sert sınır, ~20 token etkin sınır** (Long-CLIP'in ölçümü). Sorgu tarafında risk düşük (LLM zaten `semantic_text`'i kısa tutuyor) ama **fine-tune planı varsa** (domain-özgü uzun caption'larla) ciddi risk. Long-CLIP/TULIP tipi düzeltmeler mevcut, gerekirse taban model değiştirilebilir.

## 6. Sıkıştırma (X-CLIP, 512d, ~270M vektör varsayımıyla — GÜNCEL DEĞİL, bkz. aşağı)

**Vektör sayısı güncellendi (2026-07-29):** ~270M rakamı eski (ve yanlış çıkan)
300.000 saat varsayımına dayanıyordu. Teyit edilen envanter (~300.000 video ×
3-5sa, bkz. §8) + güncel 60sn pencere/60sn kaydırma (örtüşmesiz, bkz. §9;
8sn'den 60sn'ye 2026-08-01'de SINIRLI kanıtla değiştirildi, bkz. §3.1) ile
vektör sayısı ≈ **54M-90M (orta: ~72M)** — aşağıdaki tablo hâlâ eski 270M
tabanıyla, oranlar değişmez ama mutlak GB'lar ~2x ölçeklenmeli. Ayrıca tablo
X-CLIP/512d varsayımıyla yazıldı; model seçimi hâlâ kesinleşmedi (§5) — bu
oturumda lisans nedeniyle Qwen3-VL-Embedding-2B (2048d, MRL ile 64-2048d'ye
küçültülebilir) öne çıktı, o karar netleşince bu tablo yeniden hesaplanmalı.

| Yöntem | Toplam (270M taban, X-CLIP 512d) | Beklenen kayıp |
|---|---|---|
| fp16 (referans) | 276 GB | — |
| int8 | 138 GB | ~%1-2 |
| int8 + 256d (PCA/MRL) | 69 GB | ~%3-5 |
| binary + fp16 rescoring | RAM 17 GB + disk 276 GB | ~%1-3 |

**Not:** HNSW indeks boyutu vektör *sayısına* bağlıdır, vektör *boyutuna* değil — 768d→512d geçişiyle orantılı küçültme yanlış olabilir, gerçek ClickHouse implementasyonuyla ölçülmeli.

## 7. Doğrulama Planı (Adım 0)

**Golden set (altın set):** 200-500 etiketli sorgu-sonuç çifti, üçlü sorgu tasarımıyla (tam eşleşme / kısmi eşleşme / zor-negatif — "günbatımı" vs "gündoğumu" gibi görsel benzer ama anlamca yanlış). Metrikler: NDCG@10, MRR, Recall@K (`ranx` veya `pytrec_eval` ile hesaplanabilir).

**İHA verisine erişim yoksa** kullanılabilecek kamuya açık kaynaklar:

| Amaç | Kaynak |
|---|---|
| Video+metin retrieval (genel İHA) | DVTMD + CapERA (2.864 video, 14.320 caption, drone-özel) |
| Deniz/maritime sahne | SeaDronesSee |
| Genel İHA çeşitliliği | VisDrone, UAVDT |
| Hareket/takip sorguları | UAV123 |
| Gerçek İHA telemetrisi (kod doğrulama için) | PX4 Flight Review (`review.px4.io`, 120K+ gerçek log) |
| Genel GPS/hız/irtifa (telemetri kod mantığı testi) | OpenSky Network (ADS-B, ücretsiz) |
| Video+telemetri eşleştirilmiş (uçtan uca pipeline testi) | Mid-Air (sentetik, ULiège), AU-AIR |

**Ayrım önemli:** kamuya açık veri "embedding modeli genel olarak iyi mi" sorusunu cevaplar; "bizim gerçek arşivimizde doğruluk ne" sorusunu cevaplamaz — o, ya gerçek veri erişimi ya da "harness ayrımı" (siz test aracını hazırlarsınız, veriye erişimi olan biri çalıştırıp sadece sayısal sonucu geri gönderir) gerektirir.

## 8. Doğrulanmamış Varsayımlar — KRİTİK, İMPLEMENTASYONDAN ÖNCE KONTROL EDİLMELİ

| Varsayım | Nerede kullanılıyor | Durum |
|---|---|---|
| ~~1,5 PB ≈ 300.000 saat video~~ → **~300.000 video × 3-5sa ≈ 900.000-1.500.000 saat** | Tüm depolama/klip sayısı hesaplarının kökü | Kullanıcı tarafından teyit edildi (2026-07-29) — eski 300.000 saat rakamı **3-5 kat düşükmüş**. 1,5 PB'lik depolama tahmini bu yeni saat rakamıyla yeniden kontrol edilmeli (bitrate/çözünürlük varsayımı tutarlı mı). §6'daki ~270M vektör varsayımı da bu yüzden güncellendi (bkz. §6) |
| **Embedding ~40x gerçek-zaman hızı → ~7.500 GPU-saat** | İngest bütçesi, backfill süresi | Ciddi risk: InternVideo2 gerçekte 0,4x (yavaş) ölçüldü; X-CLIP'in 192ms/video rakamı küçük test videolarından, production throughput'u değil. Toplam saat 3-5x büyüdüğü için bu GPU-saat tahmini de aynı oranda büyümüş olabilir — henüz yeniden hesaplanmadı |
| Gecikme rakamları (300ms, rerank 3-15dk/8-10sn) | §3.2 performans beklentisi | Hiç ölçülmedi, mimari akıl yürütmeyle tahmin edildi |
| ~35.000 filtre-sonrası aday kümesi | Örnek senaryo | Gerçek telemetri dağılımınıza bağlı, büyük ihtimalle çok değişken |
| Doğruluk yüzdeleri (%55-75 vb.) | §7 beklenti | p⁴ yaklaştırması, NDCG ile "yakalama olasılığı" kavramsal olarak farklı şeyler — yön güvenilir, mutlak sayı değil |
| İngest aşama süreleri (proxy 5-6dk vb.) | Senaryo anlatımı | Uydurma, gösterim amaçlı |

**Öncelik:** İlk ikisi (kaynak dönüşümü + GPU bütçesi) düzeltilmeden geri kalan sayısal tablo büyük ölçüde anlamsız — implementasyona başlamadan önce gerçek envanterle ve küçük bir POC ölçümüyle bunlar sabitlenmeli.

### §8 eki — ölçülmüş bulgular (2026-07-29)

Aşağıdakiler varsayım değil, bu depoda yapılan ölçümlerdir (docs/worklog_2026-07-28.md):

- **Embedding hızı ~0,7x gerçek-zaman** (Qwen3-VL-Embedding-2B, T4, 1080p, batch'siz).
  §8'in 40x varsayımıyla arada ~60 kat fark var. Batch'leme + 240-360p proxy + daha
  hızlı GPU ile kapanabilir ama **tek GPU bu arşivi ingest edemez** — sistem yatay
  ölçeklenecek şekilde yazıldı (N worker = N GPU).
  *Ek taban ölçüm (2026-07-29):* CPU-only torch, bf16, batch=2, 360p proxy →
  **0,045x gerçek-zaman**. Bu bir ALT SINIR (hedef donanım değil), ama pipeline'ın
  uçtan uca çalıştığını gösteriyor. Gerçek kapasite kararı için 4060/A-serisi
  sınıfı GPU'da, batch ayarlanmış halde tekrar ölçülmeli.
- **Hard filtre gerçek Recall kaybı yaratıyor.** Sentetik korpusta içerikle
  korelasyonlu %10 seçicilik Recall@3'ü %28,6 → %9,5'e düşürdü (eşleştirilmiş
  bootstrap %95 GA [-38,1, -4,8] — sıfırı dışlıyor). Sadece gerçek veriyle
  (21 video + gerçek `avg_altitude_m`) "irtifa < 20m" filtresi 21 sorgunun
  17'sinde doğru cevabı *yapısal olarak* dışladı. → §3.2 madde 2'deki gevşetme
  mekanizmasının gerekçesi budur.
- **Kayıp HNSW'nin yaklaşıklığından DEĞİL.** Filtreyi geçen adaylar arasında
  HNSW, tam (brute-force) sıralamayla birebir aynı sonucu verdi: ClickHouse
  0/10 ve 0/2 sapma, Qdrant 0/21 sapma. Yani brute-force'a düşmenin faydası yok.
- **Örneklem uyarısı:** Bu ölçümlerin çoğu N=21 (bazıları N=6) üzerinde yapıldı.
  Yön güvenilir, mutlak sayılar değil — §7'nin istediği 200-500 sorguluk golden
  set hâlâ gerekli.

## 9. Bilinen İyileştirme Fırsatları (henüz uygulanmadı, test edilmeli)

- **Sahne-sınırına yaslanmış hibrit chunking**: mevcut sabit 60sn/60sn pencerelemeyi, zaten hesaplanan ffmpeg sahne değişim skoruna göre sınırları esnetip (tavan + snap-to-scene-boundary) iyileştirmek — düşük risk, düşük maliyet. Gerçek envanter ölçeğinde (§8: ~72M vektör, 60sn/60sn ile) bu artık sadece kalite değil, maliyet açısından da öncelikli olabilir. Saf shot-based chunking önerilmiyor (monoton İHA sahnelerinde — örn. 20 dk açık deniz — aşırı uzun tek segment riski var).
- **Çok-ölçekli (hiyerarşik) pencereleme — 2026-08-01'den beri ÖNCELİĞİ ARTTI**: kısa/ani olaylar için ince (8sn) ile uzun/sürekli aktiviteler için kaba (şu anki varsayılan, 60sn) katmanları birlikte tutmak. Taban pencere 8sn'den 60sn'ye çekildiğinden beri (SINIRLI kanıtla, bkz. §3.1) kısa/ani olayların artık HİÇ yakalanamama riski somutlaştı — bu artık "iyileştirme" değil, kapatılması gereken bir kör nokta olabilir. Önce sorgu loglarında süre-uyumsuzluğu gerçekten sorun mu diye ölçülmeli.
- **Late chunking**: metin-RAG dünyasından (Jina AI) gelen, komşu pencere bağlamını koruma fikri — video tarafında tooling henüz olgun değil, izlenmeli ama kısa vadede aksiyon alınmamalı.

## 10. Teknoloji Yığını Özeti

| Katman | Seçim | Alternatif değerlendirmesi |
|---|---|---|
| Video decode | ffmpeg + NVDEC | — |
| Olay kuyruğu | Kafka | RabbitMQ/Redis Streams yeterli olabilir, Kafka ekosistem uyumu için tercih edildi |
| Orkestrasyon | Temporal (backfill: Airflow) | Airflow adım-içi checkpoint sunmuyor |
| Video-metin embedding | **Qwen3-VL-Embedding-2B** (Apache-2.0, 2048d, MRL) | X-CLIP/VideoCLIP-XL/EBind lisans veya test durumu nedeniyle elendi — bkz. §5 |
| Embedding servisleme | transformers (batch'li) — Triton/TensorRT ileride | — |
| Nesne tespiti | YOLO26 | Ocak 2026'da yayınlandı, `ultralytics` ile kullanılabiliyor |
| Caption/rerank VLM | Qwen2.5-VL + vLLM | — |
| Sorgu ayrıştırma LLM | vLLM + xgrammar (§3.2 Qwen 14B öngörüyor; tek 4060 sınıfı GPU'da 7B-AWQ pratik sınır) | TGI kullanılmamalı (bakım modunda); SGLang yerine vLLM tercih edildi (guided decoding + OpenAI API tek serviste) |
| Vektör+filtre arama | **Qdrant** (filtreli HNSW) | ClickHouse ölçüldü ve elendi — prefilter HNSW'yi devre dışı bırakıyor, bkz. §2 notu |
| Metadata/durum | PostgreSQL | — |
| Nesne depolama | MinIO | — |
| Etiketleme/değerlendirme | FiftyOne (+ CVAT, YOLO etiketleme için) | — |
| İzleme | Prometheus + Grafana | — |

## 11. Sonraki Adımlar (öneri sırası)

**Tamamlananlar (2026-07-29):**
- ~~Gerçek envanter rakamları~~ → teyit edildi (~300.000 video × 3-5sa), §8 güncellendi
- ~~İngest pipeline iskelet kodu~~ → tam implementasyon yazıldı (Temporal workflow + 6 aktivite + Kafka tetikleyici + worker)
- ~~Model lisans/uygunluk elemesi~~ → Qwen3-VL-Embedding-2B (§5); kalite kararı hâlâ açık

**Sıradakiler:**
1. **Gerçek donanımda GPU-saat POC'u** — `scripts/ingest_video.py --local` gerçek-zaman katını basıyor. §8'deki en kritik riski (40x varsayımı vs ölçülen ~0,7x) kapatır. Batch boyutu, proxy çözünürlüğü ve GPU modeliyle birlikte ölçülmeli.
2. **Golden set kurulumu** (§7, 200-500 sorgu) — kamuya açık verilerle başlayıp mümkünse gerçek arşivle genişletin. Bunsuz §5 (model), §9 (chunking) ve MRL boyutu kararları doğrulanamaz. `scripts/eval_retrieval.py --golden` hazır bekliyor.
3. **1M+ ölçekte gerçek donanım testi** — hem Qdrant hem ClickHouse test ortamında 1M'de yavaşladı, kök neden kesinleşmedi. Sanallaştırılmamış donanımda tekrarlanmalı.
4. **Model seçimini golden set sonucuna göre kesinleştirin** — Qwen3-VL şu an lisans nedeniyle tek uygun aday; InternVideo2 Stage-1 / VideoPrism / LanguageBind lisansları da kontrol edilip test edilmeli.
5. **MRL boyutunu ölçün** — 2048d varsayılan; 512d ~4x depolama tasarrufu sağlar ama kalite etkisi ölçülmedi (`EMBEDDING_DIM` ile denenebilir).
6. **Chunking iyileştirmesi (§9 madde 1)** — ölçekte artık sadece kalite değil maliyet konusu.
