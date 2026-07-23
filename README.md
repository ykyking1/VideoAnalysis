# İHA Video Arşivinde Semantik Arama

Doğal dil sorgusu → video kimliği + zaman aralığı listesi döndüren hibrit (vektör + yapısal filtre) arama sistemi.

Mimari kararların ve gerekçelerinin tam dökümü için bkz. [proje-ozeti.md](proje-ozeti.md).
Claude/gelecekteki oturumlar için bağlam özeti: [CLAUDE.md](CLAUDE.md).

## Durum

Henüz implementasyon aşamasında değil. [proje-ozeti.md §8](proje-ozeti.md#8-doğrulanmamış-varsayımlar--kritik-i̇mplementasyondan-önce-kontrol-edi̇lmeli)'de
listelenen kök varsayımlar (depolama hacmi, embedding hızı, model seçimi) doğrulanmadan
sayısal planlama ve tam implementasyon anlamsız. Sıradaki adım Adım 0 (`poc/`) doğrulamalarıdır.

## Klasör yapısı

```
proje-ozeti.md      Tasarım dokümanı (kararlar, gerekçeler, açık sorular)
CLAUDE.md            Gelecekteki Claude oturumları için proje bağlamı
schema/               ClickHouse / PostgreSQL tablo tanımları
ingest/               Temporal workflow + aktiviteler (proxy, telemetri, embedding, YOLO, caption)
query/                Sorgu ayrıştırma, hibrit arama, aralık birleştirme, rerank
poc/                  Adım 0 doğrulama script'leri (envanter, hız ölçümü, golden set)
docs/                 Ek notlar / ölçüm sonuçları
```

## Sonraki adımlar

Bkz. [proje-ozeti.md §11](proje-ozeti.md#11-sonraki-adımlar-öneri-sırası).
