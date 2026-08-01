# İnternetsiz (air-gapped) kurulum

Hedef makinenin internet erişimi yoksa, gereken her şeyi **başka, internet olan
bir makinede** önceden hazırlayıp taşımanız gerekir. Bu belge iki aşamayı
anlatır: hazırlık (internet olan makinede) ve kurulum (hedef makinede).

## Kapsam ve varsayımlar

- Hazırlık makinesi hedef makineyle **aynı mimaride** olmalı (x86_64 Linux,
  mümkünse aynı Python sürümü) - pip wheel'leri ve Docker imajları platforma
  özgüdür. Farklıysa `pip download`'a `--platform`/`--python-version`
  bayrakları eklemeniz gerekir (`scripts/prepare_offline_bundle.sh`'i elle
  düzenleyin).
- **Docker, ffmpeg, git, python3 kendisi bu belgenin kapsamı dışında.**
  `scripts/setup.sh` bunları sudo ile kurmuyor (hiçbir zaman kurmadı, online
  modda da). İnternetsiz bir makinede bunları ayrıca (ör. `apt-offline` ile
  hazırlanmış bir yerel apt deposu, ya da üretici tarafından sağlanan bir
  kurulum imajı) sağlamanız gerekir - bu, işletim sistemi seviyesinde bir
  konu, proje kodunun çözebileceği bir şey değil.
- NVIDIA sürücüsü de aynı şekilde önceden kurulu olmalı.

## 1. Hazırlık (internet olan makinede)

```bash
git clone https://github.com/ykyking1/VideoAnalysis.git && cd VideoAnalysis
./scripts/prepare_offline_bundle.sh              # sadece ayristirma modeli
./scripts/prepare_offline_bundle.sh --with-vlm   # + caption/rerank icin VL modeli (~6.5 GB fazla)
```

Bu, `./offline_bundle/` klasörüne şunları indirir:

| Klasör | İçerik | Yaklaşık boyut |
|---|---|---|
| `wheels/` | Tüm Python paketleri (requirements.txt + requirements-serving.txt) | ~3-5 GB |
| `docker_images/` | MinIO, Qdrant, Postgres, Kafka, Temporal, Temporal UI, vLLM imajları (`docker save`) | ~8-10 GB |
| `models/embedding/` | Qwen3-VL-Embedding-2B | ~4 GB |
| `models/yolo/yolo26s.pt` | YOLO26 small ağırlığı | ~20 MB |
| `models/parse/` | Qwen2.5-7B-Instruct-AWQ (vLLM ayrıştırma modeli) | ~5.2 GB |
| `models/vl/` (opsiyonel) | Qwen2.5-VL-7B-Instruct-AWQ (caption/rerank) | ~6.5 GB |

Toplam **~20-25 GB** (VL modeli olmadan). `offline_bundle/`'ı USB disk ya da
yerel ağ üzerinden hedef makineye taşıyın.

## 2. Kurulum (hedef, internetsiz makinede)

```bash
git clone https://github.com/ykyking1/VideoAnalysis.git && cd VideoAnalysis
# git clone da internet ister - repo'yu da USB ile (ör. `git bundle` veya
# dosya kopyası olarak) taşımanız gerekebilir, hedefte gerçekten hiç
# internet yoksa.

./scripts/setup.sh --offline /yol/offline_bundle
```

Bu tek komut şunları yapar (hepsi yerelden, hiç internete çıkmadan):

1. Python paketlerini `pip install --no-index --find-links=<bundle>/wheels` ile kurar.
2. Docker imajlarını `docker load` ile yükler, `docker compose up`'ı
   `docker-compose.offline.yml` (`pull_policy: never`) ile çalıştırır - Docker
   internete çıkmayı hiç DENEMEZ bile.
3. `.env`'e `EMBEDDING_MODEL_DIR`, `YOLO_MODEL`, `PARSE_MODEL` (ve varsa
   `VLM_MODEL`) için **yerel** model klasör yollarını yazar.
4. `scripts/init_storage.py` ve `scripts/check_env.py` çalıştırır.

### Kritik: `.env` artık gerçekten okunuyor

Daha önce `python-dotenv` bağımlılıklarda vardı ama hiç çağrılmıyordu -
`.env` sadece Docker Compose'un kendi değişken değiştirmesi için etkiliydi,
Python tarafı (`common/config.py`) hiç okumuyordu. Bu, internetsiz kurulumun
temelini çürütecek bir şeydi (yerel model yollarını `.env`'e yazsak da
Python görmeyecekti) - `common/config.py`'ye `load_dotenv()` eklenerek
düzeltildi. Artık `python -m scripts.ingest_all` gibi komutlar `.env`'deki
yerel model yollarını gerçekten kullanıyor, HuggingFace/GitHub'a hiç
gitmiyor.

### vLLM'i başlatma

`scripts/start_vllm.sh` `PARSE_MODEL` **shell ortam değişkenini** okuyor
(`.env` dosyasını değil - bu script Python değil, ayrı bir süreç). Kurulum
sırasında `.env`'e yazılan yerel yolu vLLM'e de vermek için:

```bash
export PARSE_MODEL="$(grep ^PARSE_MODEL= .env | cut -d= -f2)"
./scripts/start_vllm.sh
```

## 3. Doğrulama

```bash
source .venv/bin/activate
python -m scripts.check_env
```

`check_env` model dosyalarının gerçekten yerel yollarda bulunup
bulunmadığını ve internete hiç çıkılmadığını kontrol eder.

## Sınırlamalar (bu belgenin test etmediği)

- Bu kurulum yolu **hiç gerçek bir internetsiz makinede uçtan uca
  çalıştırılmadı** - `prepare_offline_bundle.sh` ve `setup.sh --offline`
  kod olarak yazıldı ve mantığı doğrulandı (dotenv yükleme, pull_policy,
  model yolu yönlendirmesi ayrı ayrı test edildi) ama tam zincir gerçek bir
  air-gapped ortamda denenmedi.
- Docker imajları `:latest` etiketiyle kaydediliyor - hazırlık ve hedef
  makine arasında saatler/günler geçerse ve hazırlık makinesinde tekrar
  `prepare_offline_bundle.sh` çalıştırılırsa farklı bir `:latest` içeriği
  kaydedilebilir. Tam reprodüktibilite için imajları belirli sürüm
  etiketlerine/digest'lere sabitlemek daha güvenli olur - şu an yapılmadı.
- ffmpeg'in NVENC/NVDEC desteğiyle geldiğinden emin olun (bazı Ubuntu
  paketleri donanım kodlayıcıyı içermeyen "minimal" derlemeler olabilir) -
  bu proje kodu bunu kontrol etmiyor, sadece yazılım koduna geri çekiliyor.
