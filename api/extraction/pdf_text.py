"""Page extraction, section segmentation and language detection for one PDF.

Ported from notebook 05 (segmentation_language_preprocessing):
- page text with PyMuPDF `get_text("text", sort=True)`
- refined contents-page rule (explicit heading on pages 1-10)
- front-matter-aware segmentation (section 6.2)
- uncertain front matter after page 1 becomes `unknown` when no main section exists (6.7)
- fastText lid.176 language per section, character-weighted per document
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

import pymupdf

# --- section headings (notebook 05, section 4 and 6) -----------------------
SECTION_PATTERNS = {
    "abstract": [r"abstract", r"samenvatting", r"managementsamenvatting", r"executive summary"],
    "introduction": [r"introduction", r"inleiding", r"introductie", r"achtergrond"],
    "methods": [r"materials? and methods?", r"methods?", r"methodology", r"methodologie",
                r"methode", r"methoden", r"onderzoeksopzet", r"onderzoeksaanpak"],
    "results": [r"results?", r"resultaten", r"bevindingen", r"onderzoeksresultaten"],
    "discussion": [r"discussion", r"discussie", r"beschouwing"],
    "conclusion": [r"conclusions?", r"conclusies?", r"conclusie en aanbevelingen",
                   r"conclusions? and recommendations?", r"aanbevelingen"],
    "references": [r"references", r"bibliography", r"bibliografie", r"literatuur",
                   r"literatuurlijst", r"bronnenlijst"],
}
NUMBER_PREFIX = r"(?:\d+(?:\.\d+)*\.?\s+)?"
HEADING_PATTERNS = {
    section: re.compile(rf"^\s*{NUMBER_PREFIX}(?:{'|'.join(p)})\s*:?\s*$", re.IGNORECASE)
    for section, p in SECTION_PATTERNS.items()
}
FRONT_MATTER_PATTERNS = {
    "keywords": re.compile(r"^\s*(keywords?|trefwoorden|sleutelwoorden)\s*:?\s*$", re.IGNORECASE),
    "contents": re.compile(r"^\s*(table of contents|contents|inhoudsopgave)\s*:?\s*$", re.IGNORECASE),
}
CONTENTS_HEADING = re.compile(r"^\s*(table of contents|contents|inhoudsopgave|inhoud)\s*$", re.IGNORECASE)
MAIN_SECTION_LABELS = {"introduction", "methods", "results", "discussion", "conclusion", "references"}


def detect_section_heading(line: str) -> str | None:
    line = line.strip()
    if not line or len(line) > 100:
        return None
    for section, pattern in HEADING_PATTERNS.items():
        if pattern.fullmatch(line):
            return section
    return None


def detect_extended_heading(line: str) -> str | None:
    line = line.strip()
    if not line or len(line) > 100:
        return None
    for label, pattern in FRONT_MATTER_PATTERNS.items():
        if pattern.fullmatch(line):
            return label
    return detect_section_heading(line)


def is_contents_page(page_number: int, text: str) -> bool:
    if page_number > 10:
        return False
    opening = [l.strip() for l in text.splitlines()[:30] if l.strip()]
    return any(CONTENTS_HEADING.fullmatch(l) for l in opening)


@dataclass
class Section:
    section_number: int
    section_label: str
    page_start: int
    page_end: int
    original_text: str
    detected_language: str = "unknown"
    language_score: float = 0.0

    @property
    def chars(self) -> int:
        return len(self.original_text)


@dataclass
class ExtractedDocument:
    page_count: int
    pages: list[str]
    sections: list[Section] = field(default_factory=list)

    @property
    def characters(self) -> int:
        return sum(len(p) for p in self.pages)

    @property
    def pages_without_text(self) -> int:
        return sum(1 for p in self.pages if not p.strip())

    @property
    def text_status(self) -> str:
        if self.characters == 0 or self.pages_without_text == self.page_count:
            return "no_text"
        return "partial" if self.pages_without_text else "extractable"


def extract_pages(pdf: pymupdf.Document) -> list[str]:
    return [page.get_text("text", sort=True) for page in pdf]


def segment(pages: list[str]) -> list[Section]:
    """Front-matter-aware segmentation; keeps every extracted character."""
    sections: list[Section] = []
    current = "front_matter"
    entered_main = False

    def save(lines: list[str], label: str, page_no: int) -> None:
        text = "".join(lines)
        if text:
            sections.append(Section(len(sections) + 1, label, page_no, page_no, text))

    for page_no, text in enumerate(pages, start=1):
        if is_contents_page(page_no, text):
            if text:
                sections.append(Section(len(sections) + 1, "contents", page_no, page_no, text))
            continue
        buf: list[str] = []
        for line in text.splitlines(keepends=True):
            heading = detect_extended_heading(line)
            if heading in MAIN_SECTION_LABELS:
                save(buf, current, page_no)
                buf, current, entered_main = [], heading, True
            elif heading in {"abstract", "keywords"} and not entered_main:
                save(buf, current, page_no)
                buf, current = [], heading
            buf.append(line)
        save(buf, current, page_no)

    # 6.7: without any main section, front matter after page 1 is uncertain.
    if not any(s.section_label in MAIN_SECTION_LABELS for s in sections):
        for s in sections:
            if s.page_start > 1 and s.section_label == "front_matter":
                s.section_label = "unknown"
    return sections


class LanguageDetector:
    """fastText lid.176 wrapper (notebook 05, section 7)."""

    def __init__(self, model_path: str):
        import fasttext

        self.model = fasttext.load_model(model_path)

    def detect(self, text: str, min_chars: int = 100) -> tuple[str, float]:
        if not isinstance(text, str) or not text.strip():
            return "unknown", 0.0
        clean = " ".join(text.split())
        if len(clean) < min_chars:
            return "uncertain", 0.0
        labels, scores = self.model.predict([clean], k=1)  # batch call: NumPy 2 safe
        return labels[0][0].replace("__label__", ""), float(scores[0][0])


def route_language(sections: list[Section], detector: LanguageDetector,
                   min_share: float = 0.60) -> tuple[str, float, list[str]]:
    """Character-weighted en/nl majority over sections, as in notebook 13 section 3."""
    warnings: list[str] = []
    totals = {"en": 0, "nl": 0}
    all_chars = 0
    for s in sections:
        s.detected_language, s.language_score = detector.detect(s.original_text)
        all_chars += s.chars
        if s.detected_language in totals:
            totals[s.detected_language] += s.chars
    if not all_chars:
        return "en", 0.0, ["No text to detect a language from; defaulted to English."]
    language = max(totals, key=totals.get)
    share = totals[language] / all_chars
    if totals[language] == 0:
        whole, _ = detector.detect(" ".join(s.original_text for s in sections), min_chars=1)
        language = whole if whole in totals else "en"
        warnings.append(f"No section was confidently English or Dutch (whole text: {whole}); using {language}.")
    elif share < min_share:
        warnings.append(f"Language majority is weak ({language} = {share:.0%} of text); "
                        "the paper may be mixed-language.")
    return language, round(share, 3), warnings
