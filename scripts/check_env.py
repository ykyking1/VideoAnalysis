"""Önuçuş kontrolü: ortamın beklendiği gibi kurulup kurulmadığını doğrular.

NEDEN VAR: Bu yığındaki en tehlikeli hatalar çökmüyor, SESSİZCE bozuluyor:

- `pip install -r requirements.txt` PyPI'dan CPU torch çeker. Hiçbir hata
  alınmaz, sistem çalışır - sadece ~15-20 kat yavaş. 300.000 videoluk bir
  arşivde bu fark aylar demek.
- `qwen-vl-utils < 0.0.14` `image_patch_size` kwarg'ını bilmiyor; Qwen3-VL
  çağrısı hata fırlatmadan placeholder vektör döndürüyor. Recall şansa
  eşitleniyor ve bu yalnızca dikkatli bir ölçümle fark edilebiliyor
  (gerçekten başımıza geldi - bkz. docs/worklog_2026-07-29.md).

Bu yüzden kurulumdan sonra ve worker başlatmadan önce çalıştırın.

Kullanım:
    python -m scripts.check_env
    python -m scripts.check_env --strict   # uyari varsa da 1 doner (CI icin)
"""
import argparse
import shutil
import subprocess
import sys

from common.console import use_utf8_stdout

use_utf8_stdout()

OK, WARN, FAIL = "OK  ", "UYARI", "HATA "
_results: list[tuple[str, str]] = []


def report(level: str, message: str) -> None:
    _results.append((level, message))
    print(f"[{level}] {message}")


def check_torch() -> None:
    try:
        import torch
    except ImportError:
        report(FAIL, "torch kurulu degil")
        return

    build = torch.__version__
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
        cc = torch.cuda.get_device_capability()
        report(OK, f"torch {build} | GPU: {name} ({vram:.1f} GB, compute {cc[0]}.{cc[1]})")

        if cc[0] >= 8:
            report(OK, "bfloat16 Tensor Core destekli (Ampere+)")
        else:
            report(WARN, f"compute {cc[0]}.{cc[1]} < 8.0 - bf16 emulasyona duser, "
                         f"kod otomatik fp16'ya geciyor")
        if vram < 6:
            report(WARN, f"{vram:.1f} GB VRAM - Qwen3-VL-Embedding-2B (~4.3 GB fp16) "
                         f"sigmayabilir; EMBEDDING_BATCH_SIZE dusurun")
    else:
        detail = "CPU-only derleme" if "+cpu" in build else "CUDA gorunmuyor"
        report(FAIL, f"torch {build} - GPU KULLANILMIYOR ({detail}).")
        print("        Bu bir cokme degil: sistem calisir ama ~15-20 kat yavas.")
        print("        Cozum: pip install torch torchvision --force-reinstall \\")
        print("                 --index-url https://download.pytorch.org/whl/cu126")


def check_system_ram() -> None:
    """Model yuklemesi VRAM'den once SISTEM RAM'ini zorluyor.

    from_pretrained agirliklari once CPU'da olusturuyor; low_cpu_mem_usage
    ile tepe kullanim ~model boyutu (4 GB), onsuz ~2 kati. Colab'in ~12.7 GB
    RAM'inde bu sinira gercekten carpildi (yukleme %47'de cokme)."""
    total_gb = None
    try:  # Linux/WSL
        with open("/proc/meminfo", encoding="utf-8") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    total_gb = int(line.split()[1]) / 1024**2
                    break
    except OSError:
        try:
            import ctypes

            class _MS(ctypes.Structure):
                _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                            ("ullTotalPhys", ctypes.c_ulonglong),
                            ("ullAvailPhys", ctypes.c_ulonglong),
                            ("ullTotalPageFile", ctypes.c_ulonglong),
                            ("ullAvailPageFile", ctypes.c_ulonglong),
                            ("ullTotalVirtual", ctypes.c_ulonglong),
                            ("ullAvailVirtual", ctypes.c_ulonglong),
                            ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]
            st = _MS(); st.dwLength = ctypes.sizeof(_MS)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(st))
            total_gb = st.ullTotalPhys / 1024**3
        except Exception:  # noqa: BLE001
            pass

    if total_gb is None:
        return
    if total_gb < 8:
        report(FAIL, f"Sistem RAM {total_gb:.1f} GB - embedding modeli (4 GB) "
                     f"yuklenirken cokebilir")
    elif total_gb < 13:
        report(WARN, f"Sistem RAM {total_gb:.1f} GB - sinirda. Model yuklemesi "
                     f"cokerse EMBEDDING_BATCH_SIZE degil, RAM sorunudur "
                     f"(yukleme asamasinda cokuyor).")
    else:
        report(OK, f"Sistem RAM {total_gb:.1f} GB")


def check_qwen_vl_utils() -> None:
    """0.0.14 oncesi surumler Qwen3-VL'i SESSIZCE bozuyor."""
    try:
        from importlib.metadata import version
        v = version("qwen-vl-utils")
    except Exception:  # noqa: BLE001
        report(FAIL, "qwen-vl-utils kurulu degil")
        return

    parts = tuple(int(x) for x in v.split(".")[:3] if x.isdigit())
    if parts >= (0, 0, 14):
        report(OK, f"qwen-vl-utils {v}")
    else:
        report(FAIL, f"qwen-vl-utils {v} - 0.0.14'ten eski. Qwen3-VL cagrilari "
                     f"HATA FIRLATMADAN placeholder vektor dondurur.")
        print("        Cozum: pip install -U 'qwen-vl-utils>=0.0.14'")


def check_transformers() -> None:
    try:
        import transformers
    except ImportError:
        report(FAIL, "transformers kurulu degil")
        return
    v = transformers.__version__
    major, minor = (int(x) for x in v.split(".")[:2])
    if (major, minor) >= (4, 57):
        report(OK, f"transformers {v}")
    else:
        report(FAIL, f"transformers {v} - Qwen3-VL >=4.57 gerektiriyor")


def check_ffmpeg() -> None:
    if shutil.which("ffmpeg") is None:
        report(FAIL, "ffmpeg PATH'te yok - proxy uretimi calismaz")
        return
    encoders = subprocess.run(["ffmpeg", "-hide_banner", "-encoders"],
                               capture_output=True, text=True).stdout
    hwaccels = subprocess.run(["ffmpeg", "-hide_banner", "-hwaccels"],
                               capture_output=True, text=True).stdout

    has_cuda = "cuda" in hwaccels
    has_nvenc = "hevc_nvenc" in encoders
    report(OK if has_cuda else WARN,
           f"ffmpeg mevcut (NVDEC/cuda: {'evet' if has_cuda else 'HAYIR'}, "
           f"hevc_nvenc derlemede: {'evet' if has_nvenc else 'hayir'})")
    if has_nvenc:
        # Encoder listede olsa bile donanim olmayabilir (or. GP108/GT1030).
        # Kod bunu yakalayip yazilim encode'a duser, ama onceden bilmek iyi.
        probe = subprocess.run(
            ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-f", "lavfi",
             "-i", "testsrc=duration=1:size=320x180:rate=10",
             "-c:v", "hevc_nvenc", "-f", "null", "-"],
            capture_output=True, text=True)
        if "nvEncodeAPI" in probe.stderr or probe.returncode != 0:
            report(WARN, "hevc_nvenc listede ama BU DONANIMDA calismiyor - "
                         "kod yazilim encode'a dusecek (dogru davranis, sadece yavas)")


def check_services() -> None:
    from common import config

    try:
        from common.qdrant_store import get_client
        collections = get_client().get_collections().collections
        report(OK, f"Qdrant erisilebilir ({len(collections)} koleksiyon)")
    except Exception as exc:  # noqa: BLE001
        report(FAIL, f"Qdrant erisilemiyor ({config.QDRANT_HOST}:{config.QDRANT_PORT}): "
                     f"{str(exc)[:120]}")

    from common.minio_client import backend_name, get_client as storage_client
    try:
        buckets = storage_client().list_buckets()
        report(OK, f"Nesne deposu erisilebilir: {backend_name()} "
                   f"({len(buckets)} bucket)")
    except Exception as exc:  # noqa: BLE001
        report(FAIL, f"Nesne deposu erisilemiyor: {backend_name()} - {str(exc)[:120]}")
        if not config.LOCAL_STORAGE_PATH:
            print("        Docker calistiramiyorsaniz MinIO yerine dosya sistemi "
                  "kullanabilirsiniz:")
            print("        export LOCAL_STORAGE_PATH=/veri/storage")

    from common.llm import health_check
    if health_check():
        report(OK, f"vLLM erisilebilir ({config.VLLM_BASE_URL})")
    else:
        report(WARN, f"vLLM erisilemiyor ({config.VLLM_BASE_URL}) - sorgu ayristirma, "
                     f"caption ve rerank devre disi kalir (arama yine calisir)")


def check_config() -> None:
    from common import config

    if config.STRIDE_S < config.WINDOW_S:
        overlap = 1 - config.STRIDE_S / config.WINDOW_S
        report(WARN, f"STRIDE_S={config.STRIDE_S} < WINDOW_S={config.WINDOW_S} "
                     f"(%{overlap * 100:.0f} ortusme) - vektor sayisi ve GPU maliyeti "
                     f"{config.WINDOW_S / config.STRIDE_S:.1f}x artar")
    else:
        report(OK, f"Pencereleme {config.WINDOW_S}sn/{config.STRIDE_S}sn (ortusmesiz)")

    report(OK, f"EMBEDDING_DIM={config.EMBEDDING_DIM}, "
               f"batch={config.EMBEDDING_BATCH_SIZE}, dtype={config.EMBEDDING_DTYPE}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--strict", action="store_true",
                     help="Uyarilari da hata say (CI icin)")
    ap.add_argument("--skip-services", action="store_true")
    args = ap.parse_args()

    print("=" * 68)
    print("ONUCUS KONTROLU")
    print("=" * 68)

    print("\n-- Model calistirma --")
    check_torch()
    check_system_ram()
    check_transformers()
    check_qwen_vl_utils()

    print("\n-- Video isleme --")
    check_ffmpeg()

    print("\n-- Yapilandirma --")
    check_config()

    if not args.skip_services:
        print("\n-- Servisler --")
        check_services()

    failures = [m for lvl, m in _results if lvl == FAIL]
    warnings = [m for lvl, m in _results if lvl == WARN]

    print("\n" + "=" * 68)
    if failures:
        print(f"{len(failures)} HATA, {len(warnings)} uyari - ingest baslatmayin.")
        return 1
    if warnings and args.strict:
        print(f"{len(warnings)} uyari (--strict).")
        return 1
    print(f"Hata yok, {len(warnings)} uyari. Ortam hazir.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
