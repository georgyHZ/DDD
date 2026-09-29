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
