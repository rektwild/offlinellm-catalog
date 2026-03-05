# OfflineLLM Catalog

Bu repo, uygulamanın uzaktan okuduğu `catalog.json` dosyasını üretir ve yayınlar.

## Dosya Yapısı

- `catalog.json`: Uygulamanın doğrudan okuduğu çıktı dosyası.
- `curated_models.yaml`: Otomatik keşifle üretilen curated model listesi.
- `model_discovery_policy.yaml`: Hugging Face keşif filtreleri ve segmentasyon kuralları.
- `schema/catalog.schema.json`: `ModelCatalog` ile birebir JSON şeması.
- `scripts/refresh_curated_models.py`: HF'den adayları toplayıp `curated_models.yaml` üretir.
- `scripts/build_catalog.py`: Hugging Face verisini alıp katalogu üretir.
- `scripts/validate_catalog.py`: Schema + zorunlu alan + URL + duplicate id + semantik kural doğrulaması yapar.
- `.github/workflows/publish-catalog.yml`: Günlük otomasyon.

## Çalışma Prensibi

1. `refresh_curated_models.py`, politika dosyasındaki filtrelerle Hugging Face adaylarını tarar.
2. Repo/dosya filtreleri uygulanır; quantization önceliğiyle (`Q4_K_M > Q4_K_S > Q4_0`) tek GGUF seçilir.
3. Dedupe + popülerlik skoru ile model seçimi yapılır; boyuta göre bucket kotası (`small/medium/large`) uygulanır.
4. Script `curated_models.yaml` dosyasını otomatik yazar.
5. `build_catalog.py`, curated girdiden `catalog.json` üretir.
6. Önceki `models` içeriği değiştiyse `version` bir artırılır, değişmediyse korunur.
7. `updatedAt` her üretimde UTC `YYYY-MM-DD` olarak güncellenir.

## Curated Alanları (`curated_models.yaml`)

Her model için:

- `id`, `name`, `family`, `group`, `tags`, `minRamGb`, `devices`, `description`, `contextLength`
- `hfRepo`
- `fileName` veya `fileRegex`
- `quantization` (opsiyonel değil, app şemasında zorunlu)
- `sha256` (opsiyonel)

## Lokal Çalıştırma

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python scripts/refresh_curated_models.py
python scripts/build_catalog.py
python scripts/validate_catalog.py
```

## GitHub Action

Workflow her gün `06:00 Europe/Istanbul` için `03:00 UTC` cron ile çalışır.

- `build_catalog.py` çalışır.
- `validate_catalog.py` çalışır.
- `curated_models.yaml` ve `catalog.json` değiştiyse commit/push atılır.

`HF_TOKEN` secret tanımlıysa kullanılır, tanımlı değilse anonim istekle devam edilir.
