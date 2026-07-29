"""Ingest aktiviteleri (proje-ozeti.md §3.1).

TEMBEL YÜKLEME (PEP 562): Aktiviteler modül seviyesinde eager import EDİLMEZ.
Sebep: sorgu tarafı (query/) yalnızca `clip_embedding.embed_text`e ihtiyaç
duyuyor; eager import `ultralytics`/YOLO gibi yalnızca ingest'e ait ağır
bağımlılıkları da zorunlu kılardı. Böylece sorgu-only bir dağıtıma YOLO
kurmak gerekmiyor.

Worker tarafı `ALL_ACTIVITIES`i istediğinde hepsi bir kerede yüklenir.
"""
_ACTIVITY_MODULES = {
    "embed_clips": "ingest.activities.clip_embedding",
    "extract_visual_fields": "ingest.activities.visual_fields",
    "generate_captions": "ingest.activities.selective_caption",
    "generate_proxy": "ingest.activities.proxy_generation",
    "process_telemetry": "ingest.activities.telemetry_processing",
    "write_clips": "ingest.activities.write_clips",
}

__all__ = ["ALL_ACTIVITIES", *_ACTIVITY_MODULES]


def __getattr__(name: str):
    import importlib

    if name == "ALL_ACTIVITIES":
        return [
            getattr(importlib.import_module(module), attr)
            for attr, module in _ACTIVITY_MODULES.items()
        ]
    if name in _ACTIVITY_MODULES:
        return getattr(importlib.import_module(_ACTIVITY_MODULES[name]), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
