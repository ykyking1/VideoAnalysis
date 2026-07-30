# Golden Set (proje-ozeti.md §7, §11 madde 3)

200-500 etiketli sorgu-sonuç çifti, üçlü sorgu tasarımıyla:

- **Tam eşleşme** — sorgu ile birebir örtüşen klip
- **Kısmi eşleşme** — sorgunun bir kısmını karşılayan klip
- **Zor-negatif** — görsel benzer ama anlamca yanlış (örn. "günbatımı" vs "gündoğumu")

## Metrikler

NDCG@10, MRR, Recall@K — `ranx` veya `pytrec_eval` ile hesaplanabilir.

## Veri kaynakları (İHA verisine erişim yoksa)

| Amaç | Kaynak |
|---|---|
| Video+metin retrieval (genel İHA) | DVTMD + CapERA |
| Deniz/maritime sahne | SeaDronesSee |
| Genel İHA çeşitliliği | VisDrone, UAVDT |
| Hareket/takip sorguları | UAV123 |
| Gerçek İHA telemetrisi | PX4 Flight Review (review.px4.io) |
| Genel GPS/hız/irtifa | OpenSky Network (ADS-B) |
| Video+telemetri eşleştirilmiş | Mid-Air (sentetik, ULiège), AU-AIR |

**Not:** Kamuya açık veri "embedding modeli genel olarak iyi mi" sorusunu cevaplar;
"bizim gerçek arşivimizde doğruluk ne" sorusunu cevaplamaz.

## Format

`scripts/eval_retrieval.py --golden <dosya>` bu formatı okur. **JSONL** —
her satır tek bir JSON nesnesi:

```json
{"query": "gün batımında deniz üzerinde iki tekne", "video_id": "mission_042", "t_start": 848.0, "t_end": 904.0}
{"query": "gece kıyı şeridinde hareket eden araç", "video_id": "mission_017", "t_start": 3744.0, "t_end": 3776.0}
```

Bir sorgunun sonucu beklenen aralıkla **en az 1 saniye örtüşüyorsa** isabet
sayılır (`MIN_OVERLAP_S`). Rapor: Recall@1/@5/@10 + MRR + kaç sorguda filtre
gevşetildiği.

Zor-negatif sorgular (`"günbatımı"` vs `"gündoğumu"`) ayrı bir dosyada
tutulup ayrıca çalıştırılabilir — ölçmek istediğiniz şey "yanlış olanı
getirmiyor mu" olduğu için doğru cevabın olmadığı sorgular aynı Recall
tablosuna karıştırılmamalı.

## Filtreli vs filtresiz karşılaştırma

```
python -m scripts.eval_retrieval --golden poc/golden_set/queries.jsonl --compare-filters
```

Aynı golden set'i iki kez çalıştırır: bir kez normal hat (vLLM ayrıştırma +
yapısal filtre + gevşetme), bir kez yapısal filtre tamamen kapalı (ham sorgu
metni doğrudan embedding'e gider). İkisinin Recall@k/MRR'ını yan yana basar.
Bu, docs/worklog_2026-07-28.md'deki hard-filtre ölçümünün (sentetik +
N=21 gerçek veri) devamıdır — burada gerçek pipeline ve daha büyük bir
korpusla tekrarlanır.

## queries.jsonl kaynağı (2026-07-30, Kaggle + SeaDroneSee)

Bu dosyadaki 12 sorgu, `data/seadronessee_train.zip` içindeki `manifest.json`'dan
türetildi (21 klip, her biri için `max_concurrent_per_category`, `avg_altitude_m`,
`avg_speed_ms`). **Etiketler kendi pipeline'ımızın ürettiği değil, veri setinin
kendi meta verisidir** — döngüsellik yok (bkz. `evaluate_self_retrieval`'in
"ZAYIF PROXY" uyarısı, buradaki yöntem ondan farklı ve daha güçlü).

Kapsam/sınırlar:
- **Tekne sayısı sorguları (`min_vehicle_count`)** en güvenilir kısım: bu alan
  gerçekten YOLO26 ile dolduruluyor (`ingest/activities/visual_fields.py`,
  `VEHICLE_LIKE_CLASSES` içinde `boat`) ve manifest'in `boat` kategorisiyle
  aynı kavramı ölçüyor.
- **İrtifa sorguları (son 2 satır) SADECE semantik test** - `min_agl_m` alanı
  Qdrant'ta bu klipler için HİÇ dolu değil (SeaDroneSee video dosyalarında
  gömülü MAVLink telemetrisi yok, `manifest.json`'daki `avg_altitude_m` ayrı
  bir kaynaktan hesaplanmış, ingest hattımıza hiç ulaşmıyor). Bu sorularda
  filtre her zaman gevşeyecek - bu bir hata değil, "bu test verisinde gerçek
  telemetri yok" gerçeğinin yansıması.
- İki sorguda ("5 tekne", "4 tekne") birden fazla klip aynı derecede doğru
  cevap ama tek video_id etiketleyebildik (format kısıtı) - Recall@5/@10'u
  bozmaz (etiketlenen klip hâlâ doğru), sadece Recall@1/MRR'ı hafifçe
  muhafazakar tarafa çeker.

## Örneklem büyüklüğü uyarısı

Bu projede N=21'lik bir ölçekte **tek bir klibin sıra değiştirmesinin Recall'ü
~5 puan oynattığını** ölçtük. §7'nin 200-500 sorgu önerisi bu yüzden — altındaki
örneklemlerde çıkan farkları "model A, B'den iyi" diye yorumlamayın.
`eval_retrieval.py` N<200 olduğunda bu uyarıyı otomatik basar.
