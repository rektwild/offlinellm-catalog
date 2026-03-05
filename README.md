# OfflineLLM Catalog

Bu repo, uygulamanın uzaktan okuduğu katalog JSON dosyalarını üretir ve yayınlar:

- `catalog.json` (LLM)
- `embedding_catalog.json` (Embedding)
- `ocr_catalog.json` (OCR)

Tüm kataloglar otomatik keşif + otomatik fetch ile üretilir. Manuel model girişi hedef akışta yoktur.

## Dosya Yapısı

- `catalog.json`: LLM model kataloğu.
- `embedding_catalog.json`: Embedding model kataloğu.
- `ocr_catalog.json`: OCR model kataloğu.
- `curated_models.yaml`: LLM için discovery çıktısı (generated ara artefakt).
- `embedding_curated_models.yaml`: Embedding için discovery çıktısı (generated ara artefakt).
- `ocr_curated_models.yaml`: OCR için discovery çıktısı (generated ara artefakt).
- `model_discovery_policy.yaml`: LLM discovery politikası.
- `embedding_discovery_policy.yaml`: Embedding discovery politikası.
- `ocr_discovery_policy.yaml`: OCR discovery politikası.
- `schema/catalog.schema.json`: LLM katalog şeması.
- `schema/embedding_catalog.schema.json`: Embedding katalog şeması.
- `schema/ocr_catalog.schema.json`: OCR katalog şeması.
- `scripts/refresh_curated_models.py`: LLM adaylarını keşfedip `curated_models.yaml` üretir.
- `scripts/refresh_embedding_curated_models.py`: Embedding adaylarını keşfedip `embedding_curated_models.yaml` üretir.
- `scripts/refresh_ocr_curated_models.py`: OCR adaylarını keşfedip `ocr_curated_models.yaml` üretir.
- `scripts/build_catalog.py`: Curated girdiden katalog üretir (`catalog.json` veya `embedding_catalog.json`).
- `scripts/build_ocr_catalog.py`: OCR curated girdiden `ocr_catalog.json` üretir.
- `scripts/validate_catalog.py`: LLM katalog doğrulaması.
- `scripts/validate_embedding_catalog.py`: Embedding katalog doğrulaması.
- `scripts/validate_ocr_catalog.py`: OCR katalog doğrulaması.
- `.github/workflows/publish-catalog.yml`: Günlük otomasyon.

## Çalışma Prensibi

1. LLM discovery: `refresh_curated_models.py` Hugging Face adaylarını politika ile tarar.
2. Embedding discovery: `refresh_embedding_curated_models.py` embedding GGUF adaylarını tarar.
3. OCR discovery: `refresh_ocr_curated_models.py` `main GGUF + mmproj GGUF` çiftlerini tarar.
4. Her discovery scripti, kataloga girecek URL'ler için strict canlılık kontrolü uygular (anonymous `Range GET`).
5. `build_catalog.py` LLM ve Embedding kataloglarını üretir.
6. `build_ocr_catalog.py` OCR kataloğunu üretir.
7. `version` alanı, `models` değişmişse artar; değişmemişse korunur.
8. `updatedAt` her üretimde UTC `YYYY-MM-DD` olarak güncellenir.

## Fetch Hardening

- URL üretimi sadece HF API/tree'den gelen exact path ile yapılır.
- Path case normalize edilmez.
- 401/403/404 gibi erişim hatası veren candidate'ler discovery aşamasında elenir.
- OCR için hem ana model hem `mmproj` URL'si canlı olmak zorundadır.

## Lokal Çalıştırma

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Discover (generated curated artifacts)
python scripts/refresh_curated_models.py
python scripts/refresh_embedding_curated_models.py
python scripts/refresh_ocr_curated_models.py

# Build catalogs
python scripts/build_catalog.py
python scripts/build_catalog.py --curated embedding_curated_models.yaml --output embedding_catalog.json
python scripts/build_ocr_catalog.py

# Validate catalogs
python scripts/validate_catalog.py
python scripts/validate_embedding_catalog.py
python scripts/validate_ocr_catalog.py
```

## GitHub Action

Workflow her gün `06:00 Europe/Istanbul` için `03:00 UTC` cron ile çalışır:

1. LLM curated refresh
2. Embedding curated refresh
3. OCR curated refresh
4. LLM catalog build
5. Embedding catalog build
6. OCR catalog build
7. LLM validate
8. Embedding validate
9. OCR validate
10. `curated_models.yaml`, `catalog.json`, `embedding_curated_models.yaml`, `embedding_catalog.json`, `ocr_curated_models.yaml`, `ocr_catalog.json` değiştiyse commit/push

`HF_TOKEN` secret tanımlıysa API çağrılarında kullanılır; strict URL canlılık kontrolü anonymous probe ile yapılır.
