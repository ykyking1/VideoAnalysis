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

## Format (öneri, henüz kesinleşmedi)

```json
{
  "query": "gün batımında deniz üzerinde yüksek hızlarda uçan bir TB2 videosu",
  "query_type": "exact | partial | hard_negative",
  "relevant": [
    {"video_id": "video1", "t_start": 130, "t_end": 192}
  ]
}
```
