"""One uploaded PDF -> one record in the same shape as the metadata database (`Document`),
plus generated keyword suggestions.

Steps
1. text, sections, language          notebook 05 (pdf_text.py)
2. DOI, document type                 rule-based harvester (doi_rules.py)
3. keyword suggestions                notebook 13: input_v2 excerpt + Ministral numbered topics (llm.py)
4. title / authors / year             Ministral JSON prompt (llm.py), then
5. provenance, confidence, flags      notebook 15 consolidation rules (../consolidate.py)
"""
from __future__ import annotations

import hashlib
import time
import unicodedata

import pymupdf

from .. import consolidate
from . import doi_rules
from .llm import MODEL_ID, PROMPT_VERSION, MinistralModel
from .pdf_text import ExtractedDocument, LanguageDetector, extract_pages, route_language, segment
from .source_prep import MAX_SECTION_CHARS, MAX_SOURCE_CHARS, PREPARATION_VERSION, prepare_source


class PipelineError(Exception):
    """A problem with the uploaded file itself (reported to the client as HTTP 422)."""


def document_id_for(pdf_bytes: bytes) -> tuple[str, str]:
    """Same ID scheme as the inventory (notebook 01): 'doc_' + first 16 hex of the file's SHA-256."""
    sha = hashlib.sha256(pdf_bytes).hexdigest()
    return f"doc_{sha[:16]}", sha


def _language_share(sections) -> dict | None:
    """fastText shares with the filter notebook 15 uses: en/nl, score >= 0.80, >= 100 characters."""
    totals = {"en": 0, "nl": 0}
    for s in sections:
        if s.detected_language in totals and s.language_score >= 0.80 and s.chars >= 100:
            totals[s.detected_language] += s.chars
    total = sum(totals.values())
    return {k: v / total for k, v in totals.items()} if total else None


def _embedded_observations(pdf: pymupdf.Document) -> tuple[dict, list[str]]:
    """PDF properties as candidate values, like notebook 01's pdf_embedded observations."""
    meta = pdf.metadata or {}
    obs: dict[str, list[str]] = {}
    if (meta.get("title") or "").strip():
        obs["title"] = [meta["title"].strip()]
    if (meta.get("author") or "").strip():
        obs["author"] = [meta["author"].strip()]
    keywords = consolidate.norm_list((meta.get("keywords") or "").replace(",", ";"))
    return obs, keywords


def analyse(pdf_bytes: bytes, file_name: str, model: MinistralModel, detector: LanguageDetector) -> dict:
    started = time.perf_counter()
    timings: dict[str, float] = {}
    warnings: list[str] = []
    document_id, sha = document_id_for(pdf_bytes)

    try:
        pdf = pymupdf.open(stream=pdf_bytes, filetype="pdf")
    except Exception as exc:  # noqa: BLE001 - any parse error means "not a usable PDF"
        raise PipelineError(f"The file could not be opened as a PDF ({type(exc).__name__}).") from exc
    with pdf:
        if pdf.needs_pass:
            raise PipelineError("The PDF is password-protected.")

        # 1. text, sections, language ------------------------------------------------------------
        t = time.perf_counter()
        pages = extract_pages(pdf)
        doc = ExtractedDocument(page_count=len(pages), pages=pages)
        if doc.text_status == "no_text":
            raise PipelineError("The PDF contains no extractable text (probably scanned). OCR is not supported yet.")
        if doc.text_status == "partial":
            warnings.append(f"{doc.pages_without_text} of {doc.page_count} pages have no extractable text "
                            "(images or scans); those pages were ignored.")
        doc.sections = segment(pages)
        prompt_language, _, lang_warnings = route_language(doc.sections, detector)   # en/nl for the prompt
        warnings += lang_warnings
        lang_share = _language_share(doc.sections)
        timings["text_and_language"] = round(time.perf_counter() - t, 2)

        # 2. rule-based DOI and document type ----------------------------------------------------
        t = time.perf_counter()
        cands, head, front = doi_rules.read_pdf(pdf)
        rule_doi = doi_rules.choose_doi(cands)
        rule_type = doi_rules.classify_type(head, front)
        if rule_doi["ambiguous"]:
            warnings.append("More than one DOI scored above the threshold; the highest-scoring one is used.")
        obs, emb_keywords = _embedded_observations(pdf)
        timings["rules"] = round(time.perf_counter() - t, 2)

        # 3. keyword suggestions (Ministral, notebook 13) --------------------------------------
        t = time.perf_counter()
        max_chars = MAX_SOURCE_CHARS
        source = prepare_source(doc.sections, max_chars=max_chars)
        if not source["text"]:
            raise PipelineError("No usable body text was found after cleaning (only references or contents?).")
        keyword_result = None
        for _ in range(4):
            try:
                keyword_result = model.keywords(source["text"], prompt_language)
                break
            except ValueError as exc:  # prompt too long for the working context
                max_chars = int(max_chars * 0.75)
                warnings.append(f"Source excerpt shortened to {max_chars} characters to fit the model context ({exc}).")
                source = prepare_source(doc.sections, max_chars=max_chars,
                                        max_section_chars=min(MAX_SECTION_CHARS, max_chars))
        if keyword_result is None:
            raise PipelineError("The document text could not be fitted into the model context.")
        if keyword_result["status"] == "parsed_at_token_limit":
            warnings.append("Keyword generation hit the token limit; a possibly cut-off last suggestion was dropped.")
        if keyword_result["status"] == "no_usable_numbered_topics":
            warnings.append("The model returned no usable numbered keyword list for this document.")
        timings["keywords_model"] = round(time.perf_counter() - t, 2)

        # 4. title / authors / year (Ministral) --------------------------------------------------
        t = time.perf_counter()
        meta = model.metadata(pdf)
        llm_fields, parse_status = consolidate.parse_llm(meta["raw"])
        organisations = consolidate.norm_list(llm_fields.get("organisations"))[:10]
        timings["metadata_model"] = round(time.perf_counter() - t, 2)

    # 5. consolidation: notebook 15 rules ----------------------------------------------------------
    # NFKC turns PDF ligatures (ﬁ, ﬂ) into plain letters so returned values are clean text.
    grading_pages = [unicodedata.normalize("NFKC", p) for p in pages]
    fields, flags, rejected = consolidate.resolve(
        grading_pages, llm=llm_fields, parse_status=parse_status, obs=obs, emb_keywords=emb_keywords,
        lang_share=lang_share, text_status=doc.text_status,
        rule_doi={"doi": rule_doi["doi"], "score": rule_doi["score"]} if rule_doi["doi"] else None,
        rule_type=rule_type)
    if doc.text_status == "partial":
        flags.append("partial_text")
    timings["total"] = round(time.perf_counter() - started, 2)

    value = {f: fields[f]["value"] for f in consolidate.FIELDS}
    return {
        # --- the Document fields of the metadata API (same names and meaning as metadata.db)
        "document_id": document_id,
        "file_name": file_name,
        "publinova_url": None,
        "page_count": doc.page_count,
        "title": value["title"],
        "authors": value["authors"] or [],
        "doi": value["doi"],
        "publication_year": value["publication_year"],
        "document_type": value["document_type"],
        "language": value["language"],
        "keywords": value["keywords"] or [],
        "review_status": "ok" if not flags else "needs_review",
        "review_flags": flags,
        "provenance": {f: {"source": fields[f]["source"], "confidence": fields[f]["confidence"]}
                       for f in consolidate.FIELDS},
        "extraction": {"model": MODEL_ID, "parse_status": parse_status, "rejected_fields": rejected},
        # --- added by the extraction route
        "suggested_keywords": keyword_result["terms"],
        "organisations": organisations,
        "processing": {
            "sha256": sha,
            "text_status": doc.text_status,
            "characters": doc.characters,
            "prompt_language": prompt_language,
            "language_share": {k: round(v, 3) for k, v in (lang_share or {}).items()},
            "doi_evidence": rule_doi["evidence"],
            "document_type_cues": rule_type[1] or None,
            "keyword_model": MODEL_ID,
            "keyword_prompt": PROMPT_VERSION,
            "source_preparation": PREPARATION_VERSION,
            "source_characters_used": source["selected_characters"],
            "source_sections_used": [p["section"] for p in source["provenance"]],
            "keyword_status": keyword_result["status"],
            "keyword_termination": keyword_result["termination"],
            "input_tokens": keyword_result["input_tokens"],
            "generated_tokens": keyword_result["generated_tokens"],
            "gpu": model.device,
            "timing_seconds": timings,
            "warnings": warnings,
        },
    }
