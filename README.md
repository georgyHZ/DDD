# WHAT TO DO

##### Download notebooks 10, 11, 12, 13. Install dependencies when needed. Run all notebooks and fix all errors you come across (from missing libraries).










# Keyword suggestion experiments

These notebooks investigate up to 30 keyword/topic suggestions for English and Dutch research papers. Suggestions should match the paper's language and may express supported concepts using different wording.

## Notebook order

| Notebook | Purpose |
|---|---|
| `10_keyword_topic_suggestions_yake_baseline.ipynb` | Initial YAKE baseline on six development papers. |
| `11_keyword_topic_suggestions_input_revision.ipynb` | Revised cleaning and source selection; initial EuroLLM checks. |
| `12_keyword_topic_suggestions_eurollm_numbered_output.ipynb` | Simpler output format and parsing diagnostics for EuroLLM. |
| `13_keyword_topic_suggestions_ministral3_comparison.ipynb` | Ministral 3 3B in 4-bit precision on the same two development excerpts. |

## Running the experiments

1. Put the notebooks in `notebooks/`. Required inputs are `output/inventory/documents.csv`, `output/preprocessing/sections.jsonl` and `output/preprocessing/section_languages.csv`. Generate these through the earlier inventory and preprocessing notebooks if needed.
2. Use Python 3.12. For a baseline-only run, install `yake==0.4.8`, `pandas`, `matplotlib` and `ipykernel` in the selected notebook environment. Keep `RUN_GENERATOR = False`.
3. Model runs require additional dependencies. The verified Ministral environment used CUDA-enabled PyTorch `2.14.1+cu126`, Transformers `5.22.1`, bitsandbytes `0.50.2`, `accelerate` and mistral-common `1.13.0`. Use this environment for Notebook 13; older install cells pinning Transformers below version 5 do not describe that tested setup.
4. Start model diagnostics with `RUN_SCOPE = 'demo'`. Shut down other model kernels to release GPU memory. The first model run downloads weights; keep caches outside Git.
5. Save the executed notebook and the exact folder printed beside `Current run:`. Preserve `output/keywords/sample_manifest.json` so document assignments remain fixed. Keep the test split untouched until settings are settled.

## Evidence and decisions

The saved evidence folders under `output/keywords/` use these descriptive names (*CAN BE FOUND IN TEAMS CHANNEL -> SHARED -> 09_ARCHIVE -> MODEL TESTING RESULTS*):

| Folder | Original run ID |
|---|---|
| `initial-six-paper-baseline` | `run_85995a125b6b` |
| `six-paper-revised-input-run` | `run_213c7154fac9` | 
| `initial-EuroLLM-output-format-failures` | `run_e26de998a8f4` |
| `EuroLLM-numbered-output-experiment` | `run_30a9140d5881` |
| `Ministral-comparison` | `run_02fe7887bd34` |

Set `BASELINE_RUN_NAME = 'initial-six-paper-baseline'` in the notebooks when comparing against the renamed baseline folder. Fresh runs still write configuration-based `run_...` folders; renaming saved evidence does not change the recorded run configuration. The separate six-paper input-revision evidence is `run_213c7154fac9`; include it under that name if available.

Keep their configurations, source excerpts, suggestions, raw responses, audits, timings and review files. Notebook outputs can be inspected without rerunning the models.

YAKE provided a fast baseline but returned fragments and document artefacts. Input cleaning reduced specific repetition and bibliography artefacts. EuroLLM still struggled with the requested output after a numbered-list revision. Ministral completed both examples normally, returning 20 English and 29 Dutch phrases. This supports further evaluation; it does not establish keyword accuracy or superiority over Harreman's work.

## Next steps

Review relevance, language and redundancy in `keyword_review.csv`, then compare quality at 10, 20 and 30 where enough suggestions exist. Fix the preparation/model configuration before held-out testing. Docker packaging and website integration remain separate implementation work.

Do not commit model weights, caches, virtual environments or temporary debugging files.
