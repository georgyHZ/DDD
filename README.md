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

## Keyword suggestion experiments (notebooks 10–14)

These notebooks investigate up to 30 keyword/topic suggestions for English and Dutch research papers. Suggestions should match the paper's language and may express supported concepts using different wording.

### Notebook order

| Notebook | Purpose |
|---|---|
| `10_keyword_topic_suggestions_yake_baseline.ipynb` | Initial YAKE baseline on six development papers. |
| `11_keyword_topic_suggestions_input_revision.ipynb` | Revised cleaning and source selection; initial EuroLLM checks. |
| `12_keyword_topic_suggestions_eurollm_numbered_output.ipynb` | Simpler output format and parsing diagnostics for EuroLLM. |
| `13_keyword_topic_suggestions_ministral3_comparison.ipynb` | Ministral 3 3B in 4-bit precision on the same two development excerpts. |
| `14_keybert_multilingual_keyword_extraction.ipynb` | Multilingual KeyBERT baseline (`paraphrase-multilingual-MiniLM-L12-v2`) with author-keyword handling, on the same development and test split. |

Background and decisions are summarised in `docs/about-this-branch.md`.

### Running notebooks 10–13

1. Required inputs are `output/inventory/documents.csv`, `output/preprocessing/sections.jsonl` and `output/preprocessing/section_languages.csv`. Generate these with notebooks 01 and 05 if needed.
2. Use Python 3.12. For a baseline-only run, install `yake==0.4.8`, `pandas`, `matplotlib` and `ipykernel` in the selected notebook environment. Keep `RUN_GENERATOR = False`.
3. Model runs require additional dependencies. The verified Ministral environment used CUDA-enabled PyTorch `2.14.1+cu126`, Transformers `5.22.1`, bitsandbytes `0.50.2`, `accelerate` and mistral-common `1.13.0`. Use this environment for notebook 13; older install cells pinning Transformers below version 5 do not describe that tested setup.
4. Start model diagnostics with `RUN_SCOPE = 'demo'`. Shut down other model kernels to release GPU memory. The first model run downloads weights; keep caches outside Git.
5. Save the executed notebook and the exact folder printed beside `Current run:`. Preserve `output/keywords/sample_manifest.json` so document assignments remain fixed. Keep the test split untouched until settings are settled.

### Running notebook 14 (KeyBERT)

1. Uses the same preprocessing outputs as notebooks 10–13 and the shared `sample_manifest.json` in the project root (identical to `output/keywords/sample_manifest.json`).
2. Install `keybert`, `sentence-transformers` and `spacy` in the notebook kernel. The model downloads once and runs locally.
3. Results are written to `output/keybert/`: candidates, suggestions and selection summaries for the development and test documents, plus `keybert_config.json`.
4. Business rule used: documents with verified author keywords get five additional suggestions; when author keywords cannot be extracted, ten fallback suggestions are generated with the status `extraction_unavailable`. Author keywords and suggestions are kept in separate fields.

### Evidence and decisions

The saved evidence folders under `output/keywords/` use these descriptive names (*can be found in the Teams channel -> Shared -> 09_ARCHIVE -> MODEL TESTING RESULTS*):

| Folder | Original run ID |
|---|---|
| `initial-six-paper-baseline` | `run_85995a125b6b` |
| `six-paper-revised-input-run` | `run_213c7154fac9` |
| `initial-EuroLLM-output-format-failures` | `run_e26de998a8f4` |
| `EuroLLM-numbered-output-experiment` | `run_30a9140d5881` |
| `Ministral-comparison` | `run_02fe7887bd34` |

Set `BASELINE_RUN_NAME = 'initial-six-paper-baseline'` in the notebooks when comparing against the renamed baseline folder. Fresh runs still write configuration-based `run_...` folders; renaming saved evidence does not change the recorded run configuration.

Keep their configurations, source excerpts, suggestions, raw responses, audits, timings and review files. Notebook outputs can be inspected without rerunning the models.

YAKE provided a fast baseline but returned fragments and document artefacts. Input cleaning reduced specific repetition and bibliography artefacts. EuroLLM still struggled with the requested output after a numbered-list revision. Ministral completed both examples normally, returning 20 English and 29 Dutch phrases. KeyBERT returns one- and two-word phrases taken from the text; some are word-order variants or fragments. None of these results establishes keyword accuracy or superiority over Harreman's work; human relevance review is still required.

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
