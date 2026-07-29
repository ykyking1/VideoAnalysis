"""Ingest aktiviteleri arasında paylaşılan veri tipleri.

Temporal aktivite argümanları JSON'a serileştirildiği için tüm alanlar
dataclass + ilkel tip - özel serileştirici gerekmiyor.
"""
from dataclasses import dataclass, field


@dataclass
class TelemetryWindow:
    """Bir zaman penceresi ve o pencereye ait türetilmiş telemetri alanları
    (proje-ozeti.md §3.1 madde 2). Telemetri yoksa türetilmiş alanlar None
    kalır - filtre katmanı None alanları atlar."""
    t_start: float
    t_end: float
    avg_speed_kmh: float | None = None
    agl_m: float | None = None
    sun_elevation: float | None = None
    over_sea: bool | None = None
    sensor_type: str = "unknown"
    lat: float | None = None
    lon: float | None = None
    heading_deg: float | None = None
    gimbal_pitch_deg: float | None = None

    @property
    def key(self) -> str:
        """Aktiviteler arası pencere eşleştirme anahtarı (caption sözlüğü gibi
        pencere-indeksli çıktılarda kullanılır)."""
        return f"{self.t_start:.3f}:{self.t_end:.3f}"


@dataclass
class VisualFields:
    """YOLO26'dan türetilen, kolonlaştırılmış görsel alanlar
    (proje-ozeti.md §3.1 madde 4)."""
    vehicle_count: int = 0


@dataclass
class IngestResult:
    video_id: str
    windows_written: int
    captions_generated: int
    proxy_key: str
    warnings: list[str] = field(default_factory=list)
