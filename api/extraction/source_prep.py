"""Source preparation `input_v2`, ported from notebook 13 section 5 (`prepare_source`).

Selects up to MAX_SOURCE_CHARS of cleaned text, one excerpt per section category first
(abstract > conclusion > introduction > discussion > results > methods > front matter > unknown),
cut at the bibliography and with headings, page numbers, repeated header/footer lines,
numeric table rows and author-year citations removed.
"""
from __future__ import annotations

import hashlib
import re
import unicodedata
from collections import Counter

from .pdf_text import Section

PREPARATION_VERSION = "input_v2"
MAX_SOURCE_CHARS = 6000
MAX_SECTION_CHARS = 1500
PRIORITY = {"abstract": 0, "conclusion": 1, "introduction": 2, "discussion": 3,
            "results": 4, "methods": 5, "front_matter": 6, "unknown": 7}
REFERENCE_HEADING = re.compile(
    r"^\s*(?:\d+[.)]?\s+)?(?:references|bibliography|referenties|literatuurlijst|bronnen(?:lijst)?)\s*[:.]?\s*$", re.I)
HEADING = re.compile(
    r"^\s*(?:\d+(?:\.\d+)*[.)]?\s*)?(?:abstract|samenvatting|introduction|inleiding|conclusions?|conclusies?|discussion|"
    r"discussie(?: en interpretatie)?|results|resultaten|methods?|methoden?|materials and methods|keywords|trefwoorden|"
    r"sleutelwoorden|voorwoord|funding|acknowledg(?:e)?ments)\s*[:.]?\s*$", re.I)
AUTHOR_YEAR = re.compile(r"\([^()\n]{0,260}\b(?:18|19|20)\d{2}[a-z]?[^()\n]{0,160}\)")
ABSTRACT_CUE = re.compile(r"\b(?:abstract|a\s+b\s+s\s+t\s+r\s+a\s+c\s+t|samenvatting)\b", re.I)


def normalise(s: str) -> str:
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", str(s))).strip()


def edge_key(text: str) -> str:
    return re.sub(r"\d+", "#", normalise(text)).casefold()


def prepare_source(sections: list[Section], max_chars: int = MAX_SOURCE_CHARS,
                   max_section_chars: int = MAX_SECTION_CHARS) -> dict:
    group = sorted(sections, key=lambda s: s.section_number)
    total_chars = sum(s.chars for s in group)
    actions: Counter = Counter()

    edges: Counter = Counter()
    for s in group:
        lines = [x.strip() for x in s.original_text.splitlines() if x.strip()]
        edges.update({edge_key(x) for x in lines[:2] + lines[-2:] if 3 <= len(x) <= 140})
    repeated = {x for x, n in edges.items() if n >= max(3, int(len(group) * 0.30))}

    cutoff = None
    for s in group:
        if s.section_label == "contents":
            continue
        for offset, line in enumerate(s.original_text.splitlines()):
            if REFERENCE_HEADING.fullmatch(line):
                cutoff = (s.section_number, offset)
                break
        if cutoff is not None:
            break

    candidates = []
    for s in group:
        lines = s.original_text.splitlines()
        if cutoff and s.section_number > cutoff[0]:
            actions["after_bibliography_boundary"] += 1
            continue
        if cutoff and s.section_number == cutoff[0]:
            lines = lines[:cutoff[1]]
            actions["bibliography_boundary"] += 1
        if s.section_label in {"references", "contents"}:
            actions["excluded_section_label"] += 1
            continue
        nonempty = [x.strip() for x in lines if x.strip()]
        if nonempty and re.fullmatch(r"voorwoord|foreword|preface", nonempty[0], re.I):
            actions["foreword_block"] += 1
            continue
        effective = "unknown" if s.section_label == "keywords" else s.section_label
        joined = "\n".join(lines)
        cue = ABSTRACT_CUE.search(joined[:3000]) if s.section_label == "front_matter" else None
        if cue:
            actions["before_abstract_cue"] += 1
            joined = joined[cue.end():]
            lines = joined.splitlines()
            effective = "abstract"

        cleaned = []
        for line in lines:
            line = unicodedata.normalize("NFKC", line).replace("­", "")
            stripped = line.strip()
            if not stripped:
                cleaned.append("\n")
                continue
            if HEADING.fullmatch(stripped) or re.fullmatch(r"\d{1,4}", stripped):
                actions["heading_or_page_number"] += 1
                continue
            if edge_key(stripped) in repeated:
                actions["repeated_edge_line"] += 1
                continue
            if re.match(r"^(?:keywords|trefwoorden|sleutelwoorden)\s*:", stripped, re.I):
                cols = re.split(r"\s{4,}", stripped)
                actions["explicit_keyword_line"] += 1
                if len(cols) <= 1:
                    continue
                stripped = "; ".join(cols[1:])
            nums = re.findall(r"\b\d+(?:[.,]\d+)?\b", stripped)
            if len(nums) >= 3 and len(nums) / max(1, len(stripped.split())) >= 0.30:
                actions["numeric_table_line"] += 1
                continue

            def citation_repl(m: re.Match) -> str:
                if re.search(r"[A-Za-zÀ-ÿ]", m.group()):
                    actions["parenthetical_year_citation"] += 1
                    return " "
                return m.group()

            stripped = AUTHOR_YEAR.sub(citation_repl, stripped)
            stripped = re.sub(r"\s{4,}", "; ", stripped)
            if stripped.strip():
                cleaned.append(stripped.strip())
        text = "\n".join(cleaned)
        text = re.sub(r"(?<=[A-Za-zÀ-ÿ])-\s*\n\s*(?=[a-zà-ÿ])", "", text)
        text = re.sub(r"\n\s*\n+", ";\n\n", text)
        text = re.sub(r"[ \t]+", " ", text).strip(" ;\n")
        if len(text) < 40:
            continue
        candidates.append({"section_number": s.section_number, "effective_label": effective,
                           "page_start": s.page_start, "text": text,
                           "priority": PRIORITY.get(effective, 8)})

    candidates.sort(key=lambda x: (x["priority"], x["section_number"]))
    first, rest, seen = [], [], set()
    for c in candidates:
        (rest if c["effective_label"] in seen else first).append(c)
        seen.add(c["effective_label"])

    pieces, provenance, used = [], [], 0
    for item in first + rest:
        remaining = max_chars - used
        if remaining <= 0:
            break
        excerpt = item["text"][:min(max_section_chars, remaining)].rstrip()
        if not excerpt:
            continue
        if len(excerpt) < len(item["text"]) and " " in excerpt:
            excerpt = excerpt.rsplit(" ", 1)[0]
        used += len(excerpt)
        pieces.append(excerpt)
        provenance.append({"section": item["effective_label"], "page": item["page_start"],
                           "characters": len(excerpt)})
    text = ";\n\n".join(pieces)
    return {
        "text": text,
        "source_sha256": hashlib.sha256(text.encode()).hexdigest(),
        "selected_characters": used,
        "original_characters": total_chars,
        "bibliography_cutoff_found": cutoff is not None,
        "provenance": provenance,
        "cleaning_actions": dict(actions),
    }
