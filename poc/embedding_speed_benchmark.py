"""Adım 0, öncelik 2: embedding hızı GPU-saat POC'u (proje-ozeti.md §8, §11
madde 2). "~40x gerçek-zaman hızı -> ~7.500 GPU-saat" varsayımı ciddi risk
taşıyor: InternVideo2 gerçekte 0,4x (yavaş) ölçüldü, X-CLIP'in 192ms/video
rakamı küçük test videolarından, production throughput'u değil.

X-CLIP'i (ve varsa diğer adayları, bkz. §5) kendi donanımda, gerçek 8sn
pencereler üzerinde, Triton batch'lemesiyle ölçer.

Kullanım: python poc/embedding_speed_benchmark.py --model x-clip --clips-dir <dir>
"""
import argparse
import time
from dataclasses import dataclass


@dataclass
class BenchmarkResult:
    model_name: str
    num_clips: int
    total_wall_s: float
    clips_per_second: float
    realtime_multiplier: float  # clips_per_second * WINDOW_S / batch_size karşılığı


def load_model(model_name: str):
    """xuguohai/X-CLIP, alibaba-pai/VideoCLIP-XL vb. modeli yükler.
    Model kataloğu §5'teki aday listesiyle eşleşmeli."""
    raise NotImplementedError


def run_benchmark(model_name: str, clips_dir: str, batch_size: int = 8) -> BenchmarkResult:
    """clips_dir altındaki gerçek 8sn pencere videolarını modelden geçirip
    verim (clip/s) ve gerçek-zamana oranını ölçer."""
    raise NotImplementedError


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="bkz. proje-ozeti.md §5 aday listesi")
    parser.add_argument("--clips-dir", required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()

    start = time.monotonic()
    result = run_benchmark(args.model, args.clips_dir, args.batch_size)
    print(result)


if __name__ == "__main__":
    main()
