# Keyword research notes

## 1. Define the task

The client wanted more useful keyword suggestions, extending beyond the existing top 10 towards up to 30. Relevance mattered more than filling every position. Suggestions needed to match the paper's English or Dutch language and could include supported concepts that were not written verbatim.

## 2. Establish a baseline

Notebook 10 used YAKE as a fast, reproducible starting point without requiring a generative model. It returned 30 phrases per development paper, but inspection revealed fragments, repeated concepts and document artefacts. Output count alone therefore did not establish usefulness.

## 3. Improve the input

Notebook 11 revised cleaning and source selection while preserving the original text exports. On six development papers, targeted checks found fewer repeated-word and bibliography-marker suggestions. Some contamination and reading-order problems remained. The experiments used selected excerpts, not the complete papers.

## 4. Investigate generative alternatives

Initial EuroLLM responses struggled with the requested JSON format. Notebook 12 tested numbered lists with parsing and repetition controls, but both diagnostic responses reached the token limit and mainly returned unsuitable content. Notebook 13 then tested Ministral on the same two prepared excerpts. It completed normally with 20 English and 29 Dutch phrases, providing a more promising candidate for further evaluation.

## 5. Preserve evidence and evaluate next

The notebooks, configurations, source excerpts and raw outputs were retained so others could inspect the decisions and continue experimenting. The next step was manual review of relevance, language and redundancy before comparing quality at 10, 20 and 30. Further prompt or preparation changes should use development papers and separate saved runs, leaving held-out test papers untouched until settings are fixed. No model fine-tuning or measured accuracy improvement had been established.
