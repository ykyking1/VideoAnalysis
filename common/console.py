"""Konsol çıktısı yardımcıları.

Windows konsolu varsayılan olarak cp1254/cp437 kullanıyor ve Türkçe karakter
ile kutu çizim karakterlerinde `UnicodeEncodeError` fırlatıyor. Bunu
`sys.stdout.reconfigure(encoding="utf-8")` ile çözüyorduk - ama o çağrı
SADECE gerçek bir dosya akışında var. Jupyter/Colab'da `sys.stdout` bir
`ipykernel.iostream.OutStream` nesnesi ve `reconfigure` metodu yok:

    AttributeError: 'OutStream' object has no attribute 'reconfigure'

Bu yüzden script'ler notebook içinden import edildiğinde import anında
patlıyordu. `use_utf8_stdout()` her iki ortamda da güvenli.
"""
import sys


def use_utf8_stdout() -> None:
    """Mümkünse stdout'u UTF-8'e çevirir; desteklenmiyorsa sessizce geçer.

    Jupyter/Colab akışları zaten UTF-8 olduğu için orada bir şey yapmaya
    gerek yok - sadece çağrının patlamaması yeterli."""
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is None:
            continue
        try:
            reconfigure(encoding="utf-8")
        except (ValueError, OSError):
            # Kapali ya da yeniden yapilandirilamayan akis - cikti yine calisir
            pass
