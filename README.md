# PDF Collection Inventory

This branch inventories HZ-related PDFs and checks how reliable their extracted text and metadata are. The current focus is **data understanding**; the inventory does not verify metadata, fetch Publinova records, perform OCR or generate keywords.

## Project folders

```text
CPS/
├── notebooks/   Jupyter notebooks
├── pdfs/        Input PDFs
├── output/      Generated CSVs and text
└── docs/        Manual review sheet and project notes
```

Keep the review sheet at `docs/pdf_collection_review.xlsx`. The inventory writes its CSVs to `output/inventory/`.

pdfs/ and output/ are gitignored, therefore create one on your own in your project root folder

## Run the first notebook

1. Put PDFs in `pdfs/`. Subfolders are supported.
2. Open `notebooks/01_collection_inventory.ipynb` in Jupyter, with `notebooks/` as the working directory.
3. Install missing packages in the notebook kernel with `%pip install pymupdf pandas`, then restart the kernel if prompted.
4. Choose **Run All**. Check that the printed PDF and output paths are correct and that the expected number of files was found.
5. Inspect the overview and review queue. Open the PDFs themselves before accepting candidate metadata.

| Output | What it contains |
|---|---|
| `documents.csv` | One row per unique PDF, including page/text counts and fields for reviewed values. |
| `metadata_observations.csv` | Unreviewed metadata candidates with their source and location. |
| `document_terms.csv` | Individual terms; currently only embedded PDF keywords are extracted automatically. |

`document_id` links the CSVs. Compare titles, authors, dates, DOI and keywords with the visible PDF and Publinova record; record your decisions and any layout problems in the review sheet. Keep publication, event, online and PDF creation dates distinct.

## Common issues

- **Zero PDFs found:** check the printed `pdfs/` path and Jupyter working directory.
- **Import error:** install the named package in the active kernel and run the notebook again.
- **CSV cannot be saved:** close it in Excel and check write access to `output/inventory/`.
- **`extractable` but poor text:** every page returned *some* text; tables, slides and columns can still have broken reading order.
- **Wrong embedded title, author or keywords:** these are candidates, not verified facts. The initial sample included template titles, incomplete authors and meaningless Canva keywords.

On reruns, reviewed fields in `documents.csv` survive while the PDF bytes stay the same. Manually added observation/term rows survive only with `source_type=manual_review` or `publinova_record`; edits to automatically generated rows are overwritten. Keep the review sheet as the durable record.

## Run Notebook 05 — Section Segmentation and Language Preprocessing

This notebook identifies document sections, detects Dutch and English text, and evaluates optional stop-word removal. Original extracted text is preserved for downstream EuroBERT embeddings.

### Install dependencies

Use the same Python environment as your Jupyter notebook. Install missing packages in the notebook kernel:

```python
%pip install pymupdf pandas numpy matplotlib fasttext spacy
```

Restart the kernel if prompted.

### Download the fastText language model

The notebook requires the pretrained `lid.176.bin` model, which is not included in the repository.

From the project root, run:

```bash
mkdir -p models
curl -L https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin -o models/lid.176.bin
```

The model is stored locally in `models/`, which is excluded from Git. spaCy's built-in English and Dutch stop-word lists are used; downloading additional spaCy language models is not necessary.

### Run the notebook

1. Ensure the PDFs are in `pdfs/` and Notebook 01 has generated `output/inventory/documents.csv`.
2. Open `notebooks/05_segmentation_language_preprocessing.ipynb` in JupyterLab, with `notebooks/` as the working directory.
3. Run the cells in order and inspect the segmentation, language detection, and validation results.

### Generated outputs

The notebook saves these files in `output/preprocessing/`:

| Output | Description |
|---|---|
| `pages.jsonl` | Page-level extracted text and document identifiers. |
| `sections.jsonl` | Section blocks containing original text for downstream embeddings. |
| `section_languages.csv` | Language detection results and section metadata. |
| `sections_filtered_experimental.jsonl` | Optional stop-word-filtered representation alongside original text. |

Generated outputs and original PDFs remain local and are excluded from Git.

**Note:** Section identification is heuristic, and short or structured passages may have uncertain language predictions. Stop-word removal is experimental and should not replace original text for EuroBERT.

## Metadata API (Docker)

One FastAPI service (`api/`) with two halves that share the same record format (`Document`):

| Route | What it does | Built by |
|---|---|---|
| `GET /v1/documents`, `/v1/documents/{id}`, `/v1/documents/by-doi/{doi}`, `/v1/authors`, `/v1/keywords`, `/v1/stats` | Read the collection database `output/api/metadata.db` (built by notebook 15). | Georgy |
| `POST /v1/extract` | Upload a new PDF; get the same `Document` fields back, graded with the **same notebook 15 rules** (`api/consolidate.py`), plus `suggested_keywords` from Ministral 3 3B. Nothing is stored. | Marto |
| `GET /` | Browser upload page; shows the `/v1/extract` result JSON. | Marto |
| `GET /health`, `/docs` | Status of the database and the extraction model; interactive API docs. | both |

How `/v1/extract` builds a record:

1. Text, sections and fastText language (notebook 05) - `api/extraction/pdf_text.py`
2. Rule-based DOI and document type (DOI/type notebook on main) - `api/extraction/doi_rules.py`
3. Keyword suggestions: `input_v2` excerpt + numbered-topics prompt on Ministral 3 3B, NF4 4-bit (notebook 13) - `api/extraction/source_prep.py`, `llm.py`
4. Title, authors, year: Ministral JSON prompt (adapted from notebook 07) - `llm.py`
5. `provenance` / `confidence` / `review_flags`: notebook 15's `resolve()` - `api/consolidate.py`.
   Two additions there: a scored rule-based DOI (source `rules`) is preferred over the plain page-text
   regex, and the rule-based document type (source `rules`) is used when the LLM gives none.

`document_id` is `doc_` + the first 16 hex characters of the file's SHA-256, the same ID as the inventory,
so `known_in_database` tells whether the upload is already in `metadata.db`.
`keywords` keeps notebook 15's meaning (keywords printed or embedded by the authors);
the generated suggestions for the website are in `suggested_keywords`.

### Run

Needs Docker Desktop with an NVIDIA GPU (6 GB+), `models/lid.176.bin` and
`models/Ministral-3-3B-Instruct-2512-BF16/` (both gitignored; without the local Ministral folder the
weights are downloaded once from Hugging Face). Build `output/api/metadata.db` with notebook 15 for the read routes.

```bash
docker compose up -d --build
docker compose logs -f
```

If the models or the database live elsewhere, put the paths in a `.env` file next to `docker-compose.yml`
(gitignored), e.g. `MODELS_DIR=C:/path/to/models` and `API_DATA_DIR=C:/path/to/output/api`.
The model loads in about a minute; `/health` shows `"extraction": {"status": "ready"}`.

Without a GPU, the read routes still work locally: `EXTRACTION_ENABLED=0 uvicorn api.main:app` (notebook 16).

### Test

* Browser: <http://localhost:8000>, choose a PDF, *Extract keywords* (20-70 s).
* Command line: `curl.exe -F "file=@pdfs/005_building-relationships-in-higher-education-to-support-students-motivation.pdf" http://localhost:8000/v1/extract`
* Read routes: `curl.exe "http://localhost:8000/v1/documents?q=flood&limit=5"`, `curl.exe http://localhost:8000/v1/stats`
* Errors: 400 not a PDF, 413 too large, 422 no extractable text (scanned; no OCR), 503 model loading.

With `METADATA_API_KEY` set, every `/v1` call needs the `X-API-Key` header; set `DEMO_PAGES=0` as well,
because the browser page does not send a key.
