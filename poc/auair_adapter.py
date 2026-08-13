"""AU-AIR veri setinin telemetrisini bizim pipeline'ın beklediği düz kayıt
formatına çevirir - proje-ozeti.md §8'deki "aday dataset deneme" fikrinin
bir parçası, poc/ altında (Adım 0 doğrulama - bu KALICI bir ingest yolu
DEĞİL, "AU-AIR bizim kullanım amacımıza uyar mı" sorusunu cevaplamak için).

NEDEN MAVLINK TAKLİDİ GEREKMİYOR: `ingest/activities/telemetry_processing.py`
`enrich_windows(windows, records, sensor_type)` fonksiyonu MAVLink'e bağlı
değil - `parse_mavlink()`'in ürettiği düz `list[dict]` kayıt formatını
(t, lat, lon, agl_m, speed_kmh, heading_deg, gimbal_pitch_deg) bekliyor.
Bu adaptör AU-AIR'in kendi JSON şemasından doğrudan bu formata çeviriyor,
`enrich_windows()` sonrasında AYNEN çağrılabiliyor.

KAYNAK: AU-AIR 2019 (Bozcan & Kayacan) - CC BY-NC-SA / CC BY-NC lisanslı,
bkz. sohbet kaydı - ticari/üretim kullanımı için AYRICA değerlendirilmeli,
bu script sadece iç doğrulama amaçlı.

DOĞRULANMAMIŞ VARSAYIMLAR (resmi dokümantasyonda birim açıkça yazmıyor,
gerçek örnek veriden DEĞER ARALIĞI mantıksallığıyla çıkarıldı - bkz.
_infer_units_sanity_check()):
- `altitude` alanı MİLİMETRE varsayıldı (4770.9-30237.8 ham değer ->
  4.8-30.2m "alçak irtifa trafik gözetimi" ile fiziksel olarak tutarlı;
  metre olsaydı 4.8-30.2KM olurdu - bir trafik-izleme dronu için anlamsız).
- `linear_x/y/z` METRE/SANİYE varsayıldı (ham vektör büyüklüğü 0.014-2.80
  -> 0.05-10.1 km/h - hovering/yavaş seyir yapan bir gözetim dronu için
  tutarlı).
- `angle_psi` (yaw, radyan) heading_deg'e çevrildi; AU-AIR'de ayrı bir
  gimbal açısı YOK - `angle_theta` (pitch, radyan) gimbal_pitch_deg
  YERİNE kullanıldı ama bu drone GÖVDE açısı, gerçek kamera/gimbal açısı
  DEĞİL (Parrot Bebop 2'nin sabit/yazılım-stabilize kamerası olması bunu
  kısmen makul kılıyor ama DOĞRULANMADI).
"""
import datetime as dt
import json
import math
import sys


def load_auair_records(annotations_path: str) -> list[dict]:
    """AU-AIR annotations.json'ı okuyup ingest/activities/telemetry_processing.py
    ::enrich_windows()'un beklediği düz kayıt formatına çevirir.

    NOT: AU-AIR tek bir video değil, 8 AYRI ham videodan çıkarılmış kareler
    içeriyor (dosya adı öneki hangi kaynak videoya ait olduğunu gösteriyor,
    ör. "frame_20190829091111_..."). Bu fonksiyon HEPSİNİ tek bir kayıt
    listesi olarak döner - gerçek ingest'te video başına AYRI ayrı
    çağrılmalı (bkz. main() örneği)."""
    with open(annotations_path, encoding="utf-8") as f:
        data = json.load(f)

    records = []
    for ann in data["annotations"]:
        t = ann["time"]
        when = dt.datetime(
            t["year"], t["month"], t["day"], t["hour"], t["min"], t["sec"],
            tzinfo=dt.timezone.utc,
        )
        unix_ts = when.timestamp() + t.get("ms", 0.0) / 1000.0

        speed_ms = math.sqrt(ann["linear_x"] ** 2 + ann["linear_y"] ** 2 + ann["linear_z"] ** 2)

        records.append({
            "t": unix_ts,  # gercek t=0 hizalamasi asagida video-basina yapiliyor
            "unix_ts": unix_ts,
            "lat": ann["latitude"],
            "lon": ann["longtitude"],  # AU-AIR'in KENDI yazim hatasi (longitude degil)
            "agl_m": ann["altitude"] / 1000.0,  # mm varsayimi - bkz. modul docstring'i
            "speed_kmh": speed_ms * 3.6,
            "heading_deg": math.degrees(ann["angle_psi"]) % 360.0,
            "gimbal_pitch_deg": math.degrees(ann["angle_theta"]),  # govde pitch'i, DOGRULANMADI
            "image_name": ann["image_name"],
        })
    return records


def split_by_source_video(records: list[dict]) -> dict[str, list[dict]]:
    """Tek bir kayit listesini, `image_name` onekine gore 8 ayri "video"ya
    boler ve her birinde t=0'i o videonun ilk karesine hizalar - tam olarak
    `parse_mavlink()`'in tek bir gercek log dosyasi icin yaptigi gibi."""
    groups: dict[str, list[dict]] = {}
    for r in records:
        prefix = r["image_name"].split("_x")[0]
        groups.setdefault(prefix, []).append(r)

    for prefix, group in groups.items():
        group.sort(key=lambda r: r["unix_ts"])
        t0 = group[0]["unix_ts"]
        for r in group:
            r["t"] = r["unix_ts"] - t0
    return groups


def _sanity_report(records: list[dict]) -> str:
    agl = [r["agl_m"] for r in records]
    speed = [r["speed_kmh"] for r in records]
    return (
        f"{len(records)} kayit | agl_m: {min(agl):.1f}-{max(agl):.1f}m | "
        f"speed_kmh: {min(speed):.2f}-{max(speed):.2f} km/h"
    )


def main() -> int:
    path = sys.argv[1] if len(sys.argv) > 1 else "data/auair_sample/annotations.json"
    records = load_auair_records(path)
    print(f"Toplam: {_sanity_report(records)}")

    groups = split_by_source_video(records)
    print(f"\n{len(groups)} kaynak video bulundu:")
    for prefix, group in sorted(groups.items()):
        print(f"  {prefix}: {_sanity_report(group)}  "
              f"(t araligi: {group[0]['t']:.0f}-{group[-1]['t']:.0f}s)")

    # enrich_windows() ile UCTAN UCA dogrulama - bkz. modul docstring'i.
    sys.path.insert(0, ".")
    from ingest.activities.telemetry_processing import enrich_windows
    from ingest.activities.types import TelemetryWindow

    first_prefix = sorted(groups)[0]
    group = groups[first_prefix]
    t_max = group[-1]["t"]
    windows = [TelemetryWindow(t_start=0.0, t_end=max(t_max, 1.0))]
    enriched = enrich_windows(windows, group, sensor_type="rgb")
    w = enriched[0]
    print(f"\nenrich_windows() ucTAN UCA test ({first_prefix}):")
    print(f"  agl_m={w.agl_m}  avg_speed_kmh={w.avg_speed_kmh}  "
          f"lat={w.lat}  lon={w.lon}  sun_elevation={w.sun_elevation}  over_sea={w.over_sea}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
