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
| ClickHouse | Klip düzeyinde arama/analitik tablosu — vektör + telemetri + metadata aynı satırda |

## 3. Sistem Mimarisi

### 3.1 Ingest Hattı (video girişi)

Tetik: MinIO bucket notification → Kafka → Temporal workflow (checkpoint + heartbeat ile dayanıklı, çöken GPU işi kaldığı pencereden devam eder).

1. **Model proxy üretimi** — ffmpeg + NVDEC, 240-360p HEVC. Önizleme için değil, model tüketimi (embedding/detektör/rerank) ve gelecekteki backfill'lerin ucuz decode'u için. Kalıcı tutulması opsiyonel (bkz. §4, JIT alternatifi).
2. **Telemetri işleme** — pymavlink + polars ile `.tlog`/MAVLink log ayrıştırma. Video 8 sn pencere / 4 sn kaydırma ile bölünür (**chunking yöntemi: sabit-uzunluk, örtüşmeli** — bkz. §9 için iyileştirme önerisi). Her pencere için türetilmiş alanlar otomatik hesaplanır:
   - `avg_speed_kmh`, `agl_m` (irtifa)
   - `sun_elevation` (astral/pysolar — "gece/günbatımı" kavramlarının deterministik karşılığı)
   - `over_sea` (Shapely + GeoPandas point-in-polygon, OSM kıyı poligonları)
   - `sensor_type`, ham telemetri özeti (lat/lon/heading/gimbal — öngörülmeyen gelecekteki filtreler için sigorta)
3. **Klip embedding** — video-metin modeli, pencere başına tek vektör. **Model seçimi henüz kesinleşmedi**, bkz. §5.
4. **Terfi etmiş görsel alanlar** — YOLO26 (IR fine-tune), sorgu loglarından sık talep gören ve deterministik çözülebilen görsel kavramları (örn. `vehicle_count`) kolonlaştırır. Katalog kullanım verisine göre organik büyür.
5. **Seçici caption** — Qwen2.5-VL (vLLM üzerinde), ffmpeg sahne değişim skoruna göre seçilen ~%10'luk "olay penceresi"ne kısa açıklama üretir; hibrit (vektör+tam metin) aramada kullanılır.
6. **Yazım** — tek satır ClickHouse `clips` tablosuna: `(video_id, t_start, t_end, embedding, sensor_type, avg_speed_kmh, sun_elevation, over_sea, agl_m, vehicle_count, caption, ham telemetri...)`. Vektör kolonunda HNSW (`vector_similarity`) indeksi.

### 3.2 Sorgu Hattı

1. **LLM ayrıştırma** — Qwen 14B + xgrammar (şema zorlamalı yapısal çıktı), SGLang üzerinde (tekrarlayan sistem promptu — alan kataloğu/eşleme tablosu — RadixAttention'dan fayda görür). Sorgu, yapısal filtrelere (telemetriden gelen, deterministik) ve semantik artığa (`semantic_text`, embedding modeline giden) ayrılır. Katalogda karşılığı olmayan kavramlar hataya değil, tamamen semantik aramaya düşer.
2. **Hibrit arama** — ClickHouse tek sorguda: skip index'lerle filtre + küçültülmüş kümede vektör karşılaştırması. Filtre ve vektör aynı satırda olduğu için ayrı vektör DB'lerdeki pre-filter/post-filter ikilemi yok.
3. **Aralık birleştirme** — ardışık eşleşen pencereler ≤10 sn boşluk toleransıyla sürekli aralıklara birleştirilir.
4. **Opsiyonel rerank** — Qwen2.5-VL, top 10-20 adayı doğrular (embedding'in görsel benzerlik yanılgılarını ayıklar).

## 4. Depolama Stratejisi

- Orijinal video: **tamamen soğuk/arşiv katmana taşınabilir** — sorgu hattı hiçbir adımda videoya dokunmuyor.
- Proxy: 240-360p HEVC, ~25-35 TB (150 TB'lik ilk tahminden küçültülmüş; önizleme gereksinimi olmadığı için "insan kalitesi" değil "model kalitesi" yeterli).
- Thumbnail: kaldırıldı (önizleme yok, tüketicisi kalmadı).
- Arama katmanı (ClickHouse): embedding modeline bağlı, bkz. §5-6.

## 5. Model Seçimi — HENÜZ KESİNLEŞMEDİ

**Mevcut durum:** X-CLIP (Ma vd. 2022, MIT lisans, `github.com/xuguohai/X-CLIP`) önde giden aday, ama **nihai karar değil**.

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

## 6. Sıkıştırma (X-CLIP, 512d, ~270M vektör varsayımıyla — bkz. §8)

| Yöntem | Toplam | Beklenen kayıp |
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
| **1,5 PB ≈ 300.000 saat video** | Tüm depolama/klip sayısı hesaplarının kökü | Hiç doğrulanmadı — gerçek dosya sayısı/süre/bitrate ile değiştirilmeli |
| **Embedding ~40x gerçek-zaman hızı → ~7.500 GPU-saat** | İngest bütçesi, backfill süresi | Ciddi risk: InternVideo2 gerçekte 0,4x (yavaş) ölçüldü; X-CLIP'in 192ms/video rakamı küçük test videolarından, production throughput'u değil |
| Gecikme rakamları (300ms, rerank 3-15dk/8-10sn) | §3.2 performans beklentisi | Hiç ölçülmedi, mimari akıl yürütmeyle tahmin edildi |
| ~35.000 filtre-sonrası aday kümesi | Örnek senaryo | Gerçek telemetri dağılımınıza bağlı, büyük ihtimalle çok değişken |
| Doğruluk yüzdeleri (%55-75 vb.) | §7 beklenti | p⁴ yaklaştırması, NDCG ile "yakalama olasılığı" kavramsal olarak farklı şeyler — yön güvenilir, mutlak sayı değil |
| İngest aşama süreleri (proxy 5-6dk vb.) | Senaryo anlatımı | Uydurma, gösterim amaçlı |

**Öncelik:** İlk ikisi (kaynak dönüşümü + GPU bütçesi) düzeltilmeden geri kalan sayısal tablo büyük ölçüde anlamsız — implementasyona başlamadan önce gerçek envanterle ve küçük bir POC ölçümüyle bunlar sabitlenmeli.

## 9. Bilinen İyileştirme Fırsatları (henüz uygulanmadı, test edilmeli)

- **Sahne-sınırına yaslanmış hibrit chunking**: mevcut sabit 8sn/4sn pencerelemeyi, zaten hesaplanan ffmpeg sahne değişim skoruna göre sınırları esnetip (tavan + snap-to-scene-boundary) iyileştirmek — düşük risk, düşük maliyet. Saf shot-based chunking önerilmiyor (monoton İHA sahnelerinde — örn. 20 dk açık deniz — aşırı uzun tek segment riski var).
- **Çok-ölçekli (hiyerarşik) pencereleme**: kısa/ani olaylar (8sn) ile uzun/sürekli aktiviteler (örn. 60sn "takip") için ayrı katmanlar — önce sorgu loglarında süre-uyumsuzluğu gerçekten sorun mu diye ölçülmeli.
- **Late chunking**: metin-RAG dünyasından (Jina AI) gelen, komşu pencere bağlamını koruma fikri — video tarafında tooling henüz olgun değil, izlenmeli ama kısa vadede aksiyon alınmamalı.

## 10. Teknoloji Yığını Özeti

| Katman | Seçim | Alternatif değerlendirmesi |
|---|---|---|
| Video decode | ffmpeg + NVDEC | — |
| Olay kuyruğu | Kafka | RabbitMQ/Redis Streams yeterli olabilir, Kafka ekosistem uyumu için tercih edildi |
| Orkestrasyon | Temporal (backfill: Airflow) | Airflow adım-içi checkpoint sunmuyor |
| Video-metin embedding | **X-CLIP (kesinleşmedi)** | bkz. §5 |
| Embedding servisleme | NVIDIA Triton + TensorRT | — |
| Nesne tespiti | YOLO26 | YOLOv8 bayatladı, güncellendi |
| Caption/rerank VLM | Qwen2.5-VL + vLLM | — |
| Sorgu ayrıştırma LLM | Qwen 14B + xgrammar + SGLang | TGI kullanılmamalı (bakım modunda) |
| Vektör+filtre arama | ClickHouse (HNSW + skip index) | Qdrant, "complex filtering" için 2026'da güçlü alternatif — ölçek büyürse değerlendirilebilir |
| Metadata/durum | PostgreSQL | — |
| Nesne depolama | MinIO | — |
| Etiketleme/değerlendirme | FiftyOne (+ CVAT, YOLO etiketleme için) | — |
| İzleme | Prometheus + Grafana | — |

## 11. Sonraki Adımlar (öneri sırası)

1. **Gerçek envanter rakamlarını toplayın** (dosya sayısı, ortalama süre, bitrate) — §8'deki kök varsayımı düzeltir.
2. **Küçük ölçekli GPU-saat POC'u** — X-CLIP'i (ve varsa diğer adayları) kendi donanımınızda, gerçek 8sn pencereler üzerinde, Triton batch'lemesiyle ölçün. §8'deki en kritik riski kapatır.
3. **Golden set kurulumu** (§7) — kamuya açık verilerle başlayıp, mümkünse gerçek arşivle (veya harness-ayrımı yöntemiyle) genişletin.
4. **Model seçimini golden set sonucuna göre kesinleştirin** — X-CLIP mi, VideoCLIP-XL mi, başka bir aday mı.
5. **Chunking iyileştirmesini (§9, madde 1) düşük riskli bir sonraki iterasyon olarak planlayın.**
6. İngest pipeline'ının iskelet kodunu yazın (Temporal workflow + 5 aktivite).
