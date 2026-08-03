# İnternetsiz kurulum — Windows hedef

Bu belge, hedef (internetsiz) makine **Windows** olduğunda izlenecek yolu
anlatır. Linux hedef için [docs/internetsiz-kurulum.md](internetsiz-kurulum.md)
kullanın — bu belge onun üzerine kurulu, **önce onu okuyun**.

## Neden ayrı bir yol var

`scripts/prepare_offline_bundle.sh`/`install_system_offline.sh`/`setup.sh`
tamamen Linux'a özgü (bash + apt/dpkg + systemd + Docker'ın statik Linux
ikilisi). Bunlar Windows'ta hiç çalışmaz. Ayrıca:

- **vLLM'in resmi Windows desteği yok** (`requirements-serving.txt`'te not
  düşülmüştü) - yapısal ayrıştırma/caption/rerank için vLLM Windows'ta
  **pip ile kurulmaz**, Docker container'ında çalıştırılır.
- **Python wheel'leri platforma özgüdür** - Ubuntu'da `pip download` ile
  inen wheel'ler (özellikle `torch`) Windows'a kurulamaz, ayrıca indirilmesi
  gerekir.
- **Docker imajları ve model ağırlıkları OS'tan bağımsızdır** - Linux
  hazırlığında (`prepare_offline_bundle.sh`) zaten indirilmiş
  `docker_images/` ve `models/` klasörleri **aynen** kullanılabilir, tekrar
  indirmeye gerek yok.

## Akış

```
[İnternetli Linux makine]              [İnternetli Windows makine]
prepare_offline_bundle.sh              prepare_offline_bundle_windows.ps1
  -> offline_bundle/                     -> offline_bundle_windows/
     docker_images/, models/                wheels_windows/, installers/
                    \                       /
                     \                     /
                      birleştir (aynı klasöre kopyala)
                                |
                                v
                    [Hedef, internetsiz Windows makine]
                    scripts\setup_windows.ps1 -Bundle <birleşik-klasör>
```

## 1. Hazırlık — Linux tarafı (zaten yaptıysanız atlayın)

```bash
./scripts/prepare_offline_bundle.sh
```

`offline_bundle/docker_images/` ve `offline_bundle/models/` kısımlarına
ihtiyacınız var. `wheels/` ve `system_packages/` (Linux'a özgü) Windows
hedefinde kullanılmayacak - dokunmanıza gerek yok, zararsızlar.

## 2. Hazırlık — Windows tarafı (internetli bir Windows makinede)

```powershell
powershell -ExecutionPolicy Bypass -File scripts\prepare_offline_bundle_windows.ps1
```

`offline_bundle_windows\` klasörüne indirir:

| Klasör | İçerik | Yaklaşık boyut |
|---|---|---|
| `wheels_windows\` | requirements.txt'in win_amd64 wheel'leri (torch dahil, CUDA index'inden) | ~3-5 GB |
| `installers\` | ffmpeg (statik, NVENC/NVDEC dahil), Git for Windows, Docker Desktop, Python 3.11 kurulum dosyaları | ~1-1.5 GB |

**`requirements-serving.txt` (vLLM) bilerek indirilmiyor** - Windows wheel'i
yok. Script bunu bir uyarı olarak yazdırır, hata değildir.

Bu script'i hangi Windows Python sürümüyle çalıştırırsanız, wheel'ler o
sürümün ABI'sine göre iner - hedef makinede de **aynı sürümü** (varsayılan
3.11) kullanın.

## 3. Birleştirme

`offline_bundle_windows\wheels_windows\` ve `offline_bundle_windows\installers\`
klasörlerini, Linux hazırlığının çıktısı olan `offline_bundle\` içine
kopyalayın (`docker_images\`, `models\` ile aynı seviyeye). Sonuçta tek bir
klasörde şunlar olmalı:

```
offline_bundle\
  docker_images\      <- Linux hazırlığından
  models\              <- Linux hazırlığından
  wheels_windows\      <- Windows hazırlığından
  installers\          <- Windows hazırlığından
  wheels\, system_packages\   <- Linux'a özgü, Windows'ta kullanılmıyor (kalabilir)
```

Bu birleşik klasörü (USB/ağ ile) hedef Windows makinesine taşıyın.

## 4. Kurulum — hedef Windows makinede

```powershell
git clone https://github.com/ykyking1/VideoAnalysis.git
cd VideoAnalysis
powershell -ExecutionPolicy Bypass -File scripts\setup_windows.ps1 -Bundle C:\yol\offline_bundle
```

(`git clone` da internet ister - repo'yu USB ile taşımanız gerekebilir,
hedefte gerçekten hiç internet yoksa.)

Python/git/ffmpeg/Docker'dan biri eksikse script durur ve
`-InstallMissing` ile tekrar çalıştırmanızı önerir:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\setup_windows.ps1 -Bundle C:\yol\offline_bundle -InstallMissing
```

Bu, eksik olanları `installers\` içindeki dosyalardan **sessiz modda**
kurmayı dener. **Docker Desktop'ın WSL2 arka ucu, "Windows Subsystem for
Linux" ve "Virtual Machine Platform" Windows özellikleri kapalıysa bir
YENİDEN BAŞLATMA isteyebilir** - script bunu algılayamaz, yeniden
başlattıktan sonra aynı komutu tekrar çalıştırmanız gerekir.

Script tamamlandığında: venv oluşturur, `requirements.txt`'i yerel
wheel'lerden kurar (**requirements-serving.txt/vLLM'i KURMAZ**), `.env`'e
yerel model yollarını yazar, Docker imajlarını yükler,
`docker compose -f docker-compose.yml -f docker-compose.offline.yml up -d`
ile altyapıyı ayağa kaldırır, `init_storage`/`check_env` çalıştırır.

## vLLM (yapısal ayrıştırma/caption/rerank) — Docker'da, ayrı adım

```powershell
docker compose --profile gpu -f docker-compose.yml -f docker-compose.offline.yml up -d vllm
```

`docker-compose.offline.yml`, `vllm` servisine `.env`'deki
`OFFLINE_BUNDLE_DIR\models\parse`'i container içine `/models/parse` olarak
bağlar (`volumes:`) ve `--served-model-name` ile **istemcinin (Python
kodunun) gördüğü model adını orijinal HF-ID olarak sabit tutar** - yani
`.env`'deki `PARSE_MODEL` **değiştirilmemeli**, HF-ID (varsayılan
`Qwen/Qwen2.5-7B-Instruct-AWQ`) olarak kalmalı. `setup_windows.ps1` zaten
bunu değiştirmiyor.

vLLM olmadan da arama çalışır - sadece yapısal filtreler devre dışı kalır,
sorgu tamamen semantiğe düşer.

**`models\vl` (caption/rerank VL modeli) bu script tarafından otomatik
bağlanmıyor** - `--with-vlm` ile indirdiyseniz, `docker-compose.offline.yml`'deki
`vllm` servisini örnek alıp ikinci bir servis/profil elle eklemeniz gerekir.

## Sınırlamalar (bu belgenin test etmediği)

- **Bu yol hiç gerçek bir internetsiz Windows makinesinde uçtan uca
  çalıştırılmadı** - kod incelemesi + `docker compose config` ile YAML
  merge'ünün doğru sonuç verdiği doğrulandı (gerçek `docker compose config`
  çıktısıyla), ama tam zincir (özellikle `-InstallMissing` yolu, Docker
  Desktop'ın WSL2 ilk-kurulum reboot akışı) test edilmedi.
- `pip download`'un `requirements.txt`'teki tüm paketler için win_amd64
  wheel bulacağı varsayılıyor (torch/torchvision CUDA index'inde,
  confluent-kafka/psycopg PyPI'de resmi Windows wheel yayınlıyor) - gerçek
  bir çalıştırmada doğrulanmadı.
- ffmpeg için BtbN'in topluluk statik derlemesi kullanılıyor (resmi
  ffmpeg.org ikilisi değil) - NVENC/NVDEC içerdiği bilinen bir özellik ama
  bu projede test edilmedi.
- Docker Desktop lisansı: büyük şirketlerde (250+ çalışan/10M+ USD gelir)
  ücretli abonelik gerektirebilir - bu script/belge lisans durumunu
  kontrol etmiyor, kurulumdan önce kendiniz doğrulayın.
