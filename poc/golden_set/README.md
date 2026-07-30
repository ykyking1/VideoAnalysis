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

## Örneklem büyüklüğü uyarısı

Bu projede N=21'lik bir ölçekte **tek bir klibin sıra değiştirmesinin Recall'ü
~5 puan oynattığını** ölçtük. §7'nin 200-500 sorgu önerisi bu yüzden — altındaki
örneklemlerde çıkan farkları "model A, B'den iyi" diye yorumlamayın.
`eval_retrieval.py` N<200 olduğunda bu uyarıyı otomatik basar.
