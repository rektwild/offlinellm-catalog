# OfflineLLM Catalog

Bu repo, uygulamanın uzaktan okuduğu `catalog.json` dosyasını üretir ve yayınlar.

## Dosya Yapısı

- `catalog.json`: Uygulamanın doğrudan okuduğu çıktı dosyası.
- `curated_models.yaml`: İnsan tarafından yönetilen seed model listesi.
- `schema/catalog.schema.json`: `ModelCatalog` ile birebir JSON şeması.
- `scripts/build_catalog.py`: Hugging Face verisini alıp katalogu üretir.
- `scripts/validate_catalog.py`: Schema + zorunlu alan + URL + duplicate id doğrulaması yapar.
- `.github/workflows/publish-catalog.yml`: Günlük otomasyon.

## Çalışma Prensibi

1. `curated_models.yaml` içindeki her model için `hfRepo` ve `fileName`/`fileRegex` okunur.
2. Hugging Face API'den `siblings` listesi çekilir.
3. Seçilen dosyadan `downloadURL` üretilir: `https://huggingface.co/{hfRepo}/resolve/main/{file}`
4. `sizeBytes` önce HF metadata'dan, yoksa HTTP HEAD/Range üzerinden bulunur.
5. Çıktı `models` listesi `id` ile sıralanarak yazılır.
6. Önceki `models` içeriği değiştiyse `version` bir artırılır, değişmediyse korunur.
7. `updatedAt` her üretimde UTC `YYYY-MM-DD` olarak güncellenir.

## Seed Alanları (`curated_models.yaml`)

Her model için:

- `id`, `name`, `family`, `group`, `tags`, `minRamGb`, `devices`, `description`, `contextLength`
- `hfRepo`
- `fileName` veya `fileRegex`
- `quantization` (opsiyonel değil, app şemasında zorunlu)
- `sha256` (opsiyonel)

## Lokal Çalıştırma

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python scripts/build_catalog.py
python scripts/validate_catalog.py
```

## GitHub Action

Workflow her gün `06:00 Europe/Istanbul` için `03:00 UTC` cron ile çalışır.

- `build_catalog.py` çalışır.
- `validate_catalog.py` çalışır.
- `catalog.json` değiştiyse commit/push atılır.

`HF_TOKEN` secret tanımlıysa kullanılır, tanımlı değilse anonim istekle devam edilir.
