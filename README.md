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