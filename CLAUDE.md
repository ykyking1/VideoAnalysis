# CLAUDE.md

Bu depo, İHA video arşivi için doğal dil → video kimliği + zaman aralığı döndüren
hibrit (vektör + yapısal filtre) arama sistemidir.

Tam tasarım dokümanı: [proje-ozeti.md](proje-ozeti.md) — herhangi bir implementasyon
kararı öncesinde okunmalı, özellikle §8 (doğrulanmamış varsayımlar).
Ölçüm ve karar kayıtları: [docs/](docs/).

## Kritik kural

`proje-ozeti.md` §8'de listelenen varsayımlar **doğrulanmadı**. Bu sayılara dayanan
kapasite planlaması, sabit kod veya "X TB tutar" gibi iddialar üretmeden önce
kullanıcıya varsayımların hâlâ doğrulanmamış olduğunu hatırlat.

Şu an §8'de teyit edilen TEK şey arşiv büyüklüğü (~300.000 video × 3-5sa ≈
900K-1.5M saat, kullanıcı teyidi 2026-07-29). Doğrulanmamış olarak KALANLAR:
embedding hızı (§8 40x varsayıyor; ölçtüğümüz ~0.7x — T4/1080p/batch'siz),
gecikme rakamları, doğruluk yüzdeleri, depolama tahminleri, model seçimi (§5).

## Mimari

- **Ingest**: MinIO bucket notification → Kafka → Temporal workflow. 6 aktivite:
  proxy üretimi (ffmpeg+NVDEC/NVENC), telemetri parse (pymavlink+polars,
  **60sn/60sn örtüşmesiz** pencere), klip embedding (Qwen3-VL-Embedding-2B),
  YOLO26 görsel alanlar, Qwen2.5-VL seçici caption, Qdrant yazımı.
- **Query**: vLLM + xgrammar ile yapısal filtre + semantik metne ayrıştırma →
  Qdrant'ta filtreli HNSW araması → **sonuç azsa kademeli filtre gevşetme** →
  ardışık pencereleri ≤10sn boşluk toleransıyla aralık birleştirme →
  opsiyonel VLM rerank.
- **Depolama**: Ham video soğuk katmana taşınabilir (sorgu hattı dokunmuyor).
  Proxy 240-360p HEVC, önizleme kalitesi değil model kalitesi hedefleniyor.

## Ölçüme dayalı kararlar (docs/worklog_2026-07-28.md)

Bunlar tahmin değil, bu depoda yapılan ölçümlerin sonucu — değiştirmeden önce
worklog'u oku:

- **Qdrant, ClickHouse yerine.** ClickHouse'ta `prefilter` HNSW'yi tamamen devre
  dışı bırakıyor (`EXPLAIN indexes=1`: granül budama 72/72), `postfilter` ise
  eksik sonuç dönebiliyor. Qdrant filtreyi HNSW gezinmesi içinde uyguluyor;
  100K korpusta `exact=True` ile 21/21 birebir aynı top-3, ~1.5x daha hızlı.
- **Qwen3-VL-Embedding-2B.** Karar LİSANS temelli: VideoCLIP-XL ve EBind
  CC-BY-NC-SA (ticari/savunma kullanımına kapalı), Qwen3-VL Apache-2.0.
  Retrieval kalitesinde "en iyi" olduğu İDDİA EDİLMİYOR (§5 hâlâ açık).
- **Hard filtre + otomatik gevşetme.** Hard filtrenin doğru cevabı gerçekten
  kaybettirdiği iki bağımsız yöntemle ölçüldü (sentetik: Recall@3 %28.6→%9.5,
  bootstrap %95 GA [-38.1,-4.8] sıfırı dışlıyor; gerçek veri: irtifa<20m
  filtresi 21 sorgunun 17'sinde doğru cevabı yapısal olarak dışladı).
  Kayıp HNSW'nin yaklaşıklığından DEĞİL — filtreyi geçen adaylar arasında
  HNSW brute-force ile birebir aynı (0/10, 0/2, 0/21 sapma).
- **Örtüşmesiz pencereleme** (STRIDE_S=WINDOW_S). %50 örtüşme gerçek
  envanterde ~1 milyar vektör demekti; kaydırmayı pencereye eşitlemek
  bunu yarıya indiriyor. Recall etkisi N=6'da test edildi, kötüleşme
  görülmedi — ama N=6 güvenilir değil.
- **Pencere boyutu 8sn'den 60sn'ye çekildi (2026-08-01) — SINIRLI KANIT.**
  Birleştirilmiş (21 SeaDroneSee klibi uç uca, 914.8sn) tek bir uzun
  videoda N=10 sorguyla ölçüldü: Recall@10 %20→%70, MRR 0.083→0.408.
  Yön güçlü ama: video yapay birleştirme (gerçek kesintisiz çekim değil),
  kısa/ani olayların 60sn'de kaybolup kaybolmadığı HİÇ test edilmedi,
  yapısal alanların (agl_m, avg_speed_kmh) bulanması test edilmedi.
  §9'daki "çok-ölçekli (hiyerarşik) pencereleme" fikri (kısa olaylar için
  8sn + uzun aktiviteler için 60sn, ayrı katmanlar) bu sorunu daha temiz
  çözebilir ama henüz uygulanmadı. Detay: docs/worklog_2026-08-01.md.

## Model seçimi (§5) — HÂLÂ AÇIK

Qwen3-VL-Embedding-2B lisans nedeniyle seçildi, kalite nedeniyle değil.
InternVideo2/VideoPrism/LanguageBind test edilmedi. Golden set (§7: 200-500
etiketli sorgu) yok — model, pencereleme (§9) ve MRL boyutu kararları bu
olmadan doğrulanamaz.

## Klasör yapısı

- `common/` — config (tüm env), Qdrant erişimi, vLLM istemcisi, MinIO
- `ingest/` — Temporal workflow + worker + Kafka consumer + 6 aktivite
- `query/` — ayrıştırma, filtre kurma/gevşetme, hibrit arama, aralık
  birleştirme, rerank, uçtan uca pipeline
- `scripts/` — init_storage, register_video, ingest_video, query_cli,
  query_ui (Gradio - arama + manuel filtre alanları + kırpılmış proxy
  klip önizleme; ingest/pipeline izleme yok),
  eval_retrieval, setup_minio_notifications
- `schema/` — PostgreSQL durum takibi DDL (Qdrant şeması koddan kuruluyor)
- `poc/` — Adım 0 doğrulama: envanter taraması, golden set rehberi
- `tests/` — 37 test, harici servis gerektirmez
- `docs/` — karar günlüğü ve ölçüm kayıtları

## Kod konvansiyonları

- Yorumlar ve docstring'ler Türkçe; kod/değişken adları İngilizce.
- Her modülün docstring'i proje-ozeti.md'nin ilgili bölümüne referans verir.
- Doğrulanmamış bir varsayıma dayanan her yer bunu açıkça yazar
  ("ÖLÇÜLMEDİ", "doğrulanmadı"). Bu uyarıları silme.
- Bağlantı bilgisi/model adı kodda sabit değil, `common/config.py` üzerinden env.
- Ağır bağımlılıklar (ultralytics vb.) tembel import edilir — sorgu-only
  dağıtım YOLO kurulumu gerektirmemeli.

## Teknoloji yığını

ffmpeg+NVDEC/NVENC, Kafka, Temporal, **Qdrant** (filtreli HNSW), YOLO26,
Qwen3-VL-Embedding-2B, Qwen2.5-VL+vLLM, xgrammar, PostgreSQL (yalnızca ingest
durum takibi), MinIO.
