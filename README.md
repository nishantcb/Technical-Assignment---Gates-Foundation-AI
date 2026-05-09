# TranslateAI — Enterprise AI Translation Platform

> Built as part of the **Gates Foundation AI Fellowship – India 2026** technical assignment (Option B: Critique & Rebuild).

A full-stack, self-hosted document translation platform with 40+ automated evaluation metrics, real-time progress via WebSockets, and multi-format report export. The backend is a single FastAPI application (`main2.py`) that serves both the REST API and an embedded HTML/JS frontend.

---

## Screenshots

| Upload | Translation | Metrics |
|--------|-------------|---------|
| Drag-and-drop document upload | Side-by-side source & translated text | Radar chart of evaluation scores |

---

## Features

- **Document translation** — supports PDF, DOCX, TXT, CSV, XLSX, and JSON
- **Multiple translation backends** — Google Translator (fast), MarianMT, mBART-50, and NLLB-200
- **Auto language detection** via `langdetect`
- **Translation caching** — MD5-keyed JSON cache avoids redundant API/model calls
- **40+ evaluation metrics** across five categories:

| Category | Metrics |
|----------|---------|
| Lexical | BLEU, ROUGE-1/2/L, METEOR, chrF, TER, CER, WER |
| Semantic | Cosine Similarity, Jaccard Similarity, Semantic Similarity (sentence-transformers) |
| Neural | BERTScore (Precision / Recall / F1) |
| Quality | Fluency, Readability, Grammar, Toxicity, Hallucination, Named Entity Accuracy |
| LLM-based | LLM Judge, G-Eval, RAGAS, DeepEval, TruLens, COMET, BLEURT |

- **Report export** — PDF, CSV, XLSX, and JSON
- **SQLite persistence** — every translation and its metrics are stored locally
- **WebSocket progress** — real-time status updates during long translations
- **Interactive radar chart** visualisation of evaluation scores (Chart.js)

---

## Project Context

This project was built in response to the [CeRAI AI Evaluation Tool](https://github.com/cerai-iitm/AIEvaluationTool) assessment task. After evaluating the existing tool, I chose **Option B (Critique & Rebuild)** and implemented a minimal viable alternative focused on:

- Document-level translation (not just conversational endpoints)
- A comprehensive, extensible metrics suite
- A usable web UI with no separate frontend build step

---

## Architecture

```
main2.py  (FastAPI application)
├── /                        → Embedded HTML/JS frontend (served inline)
├── /api/translate-document  → POST: upload & translate a document
├── /api/download-report/{id}→ GET: export report (pdf|csv|xlsx|json)
├── /ws                      → WebSocket: real-time progress
└── /health                  → GET: health check

uploads/   → temporary file storage (auto-cleaned after translation)
reports/   → generated report files
cache/     → MD5-keyed translation cache (JSON)
translation_platform.db  → SQLite database
```

---

## Requirements

- Python 3.9+
- pip

### Install dependencies

```bash
pip install -r requirements.txt
```

> **Note:** First run will download NLTK corpora (`punkt`, `wordnet`, `omw-1.4`) and, if using neural models, the relevant Hugging Face model weights. Ensure you have a stable internet connection and sufficient disk space (~2–10 GB depending on models used).

---

## Running Locally

```bash
python main2.py
```

The server starts on **http://localhost:8000**.

Alternatively, with uvicorn directly:

```bash
uvicorn main2:app --host 0.0.0.0 --port 8000 --reload
```

---

## Usage

1. Open **http://localhost:8000** in your browser.
2. Drag and drop (or click to browse) a document — PDF, DOCX, TXT, CSV, XLSX, or JSON.
3. Select a **Target Language** and a **Translation Model**.
4. Optionally paste a **Reference Translation** for more accurate metric evaluation. If omitted, a synthetic reference is generated automatically.
5. Click **Translate Now**.
6. View the translation, evaluation metrics, and radar chart.
7. Download the report in your preferred format from the Reports section.

---

## Translation Models

| Model | Description | Speed |
|-------|-------------|-------|
| `google` | Google Translator via `deep-translator` | Fast |
| `marian` | Helsinki-NLP MarianMT (language-pair specific) | Medium |
| `mbart` | `facebook/mbart-large-50-many-to-many-mmt` | Slow |
| `nllb` | `facebook/nllb-200-distilled-600M` | Slow |

MarianMT, mBART, and NLLB fall back to Google Translator automatically if the model cannot be loaded.

---

## API Reference

### `POST /api/translate-document`

Translates an uploaded document and computes all evaluation metrics.

**Form fields:**

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `file` | File | ✅ | Document to translate |
| `target_language` | string | ✅ | Target language code (e.g. `hi`, `fr`, `es`) |
| `model_name` | string | | Translation backend (default: `google`) |
| `reference_text` | string | | Reference translation for metric computation |

**Response (JSON):**

```json
{
  "translation_id": "uuid",
  "source_text": "...",
  "translated_text": "...",
  "source_language": "en",
  "target_language": "hi",
  "model_used": "google",
  "confidence_score": 0.85,
  "all_metrics": { "bleu_score": 0.92, "rouge1": 0.88, ... },
  "timestamp": "2026-05-09T18:52:01.940241"
}
```

### `GET /api/download-report/{translation_id}?format=pdf`

Downloads a translation report. Supported formats: `pdf`, `csv`, `xlsx`, `json`.

### `GET /health`

Returns `{"status": "healthy", "timestamp": "..."}`.

---

## Evaluation Metrics — Notes

- **Synthetic reference**: when no reference translation is provided, the platform uses the translated text itself as the reference. This will produce inflated scores (all 1.0) and is intended only to demonstrate the metrics pipeline. For meaningful evaluation, always supply a human reference translation.
- **BERTScore**: runs on CPU by default. GPU acceleration is used automatically if a CUDA device is available.
- **Semantic Similarity**: uses `all-MiniLM-L6-v2` from `sentence-transformers`, downloaded on first run (~80 MB).

---

## Limitations

- The synthetic reference fallback produces perfect scores and should not be treated as a real evaluation.
- BERTScore language is hardcoded to `en`; cross-lingual BERTScore requires model selection per language pair.
- TER and CER implementations are approximate (greedy alignment); they do not use full edit-distance optimisation.
- No authentication or rate limiting — intended for local/internal use only.
- Large documents may hit translation API limits or model context windows (truncated to 512 tokens for neural models).

---

## Dependencies

See [`requirements.txt`](requirements.txt). Key libraries:

- `fastapi` + `uvicorn` — web server
- `transformers`, `torch` — neural translation models
- `deep-translator` — Google Translator wrapper
- `sentence-transformers` — semantic similarity
- `bert-score` — BERTScore
- `rouge-score`, `nltk` — ROUGE, METEOR, BLEU
- `reportlab` — PDF report generation
- `pandas`, `openpyxl` — CSV/XLSX report generation

---

## Submission Details

**Path chosen:** Option B — Critique & Rebuild

**AI use:** Claude (Anthropic) was used to assist with structuring the metrics pipeline, debugging FastAPI route handlers, and generating the embedded frontend HTML/CSS/JS. Course corrections were made when initial metric implementations produced incorrect scores for non-Latin scripts (the BERTScore language parameter and the synthetic reference logic required manual fixes after reviewing output).

---

## License

MIT
