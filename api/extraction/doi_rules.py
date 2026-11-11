"""Rule-based DOI, document type and author keywords, ported from notebook 08.

A DOI is an exact string, so it is harvested from metadata, links and page zones and
scored (own DOI vs cited DOI) instead of asking the language model for it.
"""
from __future__ import annotations

import re
import unicodedata
from collections import defaultdict
from urllib.parse import unquote

import pymupdf

FULL_SCAN_PAGES = 6
LINK_PAGES = 3
EDGE = 0.12
MIN_SCORE = 3

DOI_RE = re.compile(r'10\.\d{4,9}/[^\s"<>]+', re.I)
REF_HEADING = re.compile(
    r"^\W*(?:\d+\.?\s*)?(references?|bibliography|literatuur(?:lijst)?|referenties|bronnen(?:lijst)?)\W*$", re.I)
EDGE_ZONES = {"HEADER", "FOOTER", "MARGIN_LEFT", "MARGIN_RIGHT", "SIDEBAR_LEFT", "SIDEBAR_RIGHT"}
LOCATION_ORDER = ["METADATA", "LINK", "HEADER", "FOOTER", "SIDEBAR_LEFT", "SIDEBAR_RIGHT",
                  "MARGIN_LEFT", "MARGIN_RIGHT", "BODY", "REFERENCES"]


def clean(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    return re.sub(r"[\x00-\x08\x0b-\x1f-�]", "", text).strip()


def unwrap(text: str) -> str:
    text = re.sub(r"(?i)(doi\.org/|doi\s*:?)[ \t]+(?=10\.\d)", r"\1", text)
    return re.sub(r"(10\.\d{4,9}/\S*[-./_])[ \t]*\n[ \t]*(?=[0-9a-z])", r"\1", text)


def trim_doi(doi: str) -> str:
    doi = doi.rstrip(".,;:]}'\"")
    while doi.endswith(")") and doi.count(")") > doi.count("("):
        doi = doi[:-1].rstrip(".,;:")
    return doi.lower()


def doi_matches(text: str) -> list[tuple[str, bool]]:
    text = unwrap(text)
    return [(trim_doi(m.group(0)), "doi" in text[max(0, m.start() - 12):m.start()].lower())
            for m in DOI_RE.finditer(text)]


def score_doi(cs: list[dict]) -> tuple[int, list[str]]:
    score, why = 0, []
    text_hits = [c for c in cs if c["source"] == "text"]
    edge_pages = {c["page"] for c in text_hits if c["zone"] in EDGE_ZONES}
    body_pages = {c["page"] for c in text_hits if c["zone"] == "BODY"}
    if any(c["source"] == "metadata" for c in cs):
        score += 6; why.append("in PDF metadata")
    if any(c["source"] == "link" and c["page"] <= 2 for c in cs):
        score += 3; why.append("clickable link on page 1-2")
    if 1 in edge_pages:
        score += 4; why.append("header/footer/side on page 1")
    elif 2 in edge_pages:
        score += 2; why.append("header/footer/side on page 2")
    if len(edge_pages) >= 3:
        score += 4; why.append(f"repeated on {len(edge_pages)} page edges")
    elif len(edge_pages) == 2:
        score += 2; why.append("repeated on 2 page edges")
    if body_pages & {1, 2}:
        score += 2; why.append("body text on page 1-2")
    elif body_pages:
        score += 1; why.append("body text on page 3-6")
    if any(c["marked"] for c in text_hits):
        score += 2; why.append('labelled "doi"')
    only_refs = text_hits and all(c["zone"] == "REFERENCES" for c in text_hits)
    if only_refs and not any(c["source"] in ("metadata", "link") for c in cs):
        score -= 6; why.append("only in the reference list")
    return score, why


def choose_doi(cands: list[dict]) -> dict:
    by_doi: dict[str, list[dict]] = defaultdict(list)
    for c in cands:
        by_doi[c["doi"]].append(c)
    ranked = []
    for doi, cs in by_doi.items():
        score, why = score_doi(cs)
        best = min(cs, key=lambda c: (LOCATION_ORDER.index(c["zone"]), c["page"]))
        ranked.append({"doi": doi, "score": score, "evidence": "; ".join(why),
                       "zone": best["zone"], "page": best["page"]})
    ranked.sort(key=lambda r: (-r["score"], r["page"]))
    out = {"doi": None, "score": None, "evidence": None, "ambiguous": False, "candidates": len(ranked)}
    if ranked and ranked[0]["score"] >= MIN_SCORE:
        b = ranked[0]
        out.update(doi=b["doi"], score=b["score"], evidence=b["evidence"])
        out["ambiguous"] = sum(r["score"] >= MIN_SCORE for r in ranked) > 1
    return out


def zone_of(bbox, direction, W: float, H: float) -> str:
    x0, y0, x1, y1 = bbox
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    if abs(direction[0]) < 0.7:
        if cx <= 2 * EDGE * W:
            return "SIDEBAR_LEFT"
        if cx >= (1 - 2 * EDGE) * W:
            return "SIDEBAR_RIGHT"
        return "BODY"
    if cy <= EDGE * H:
        return "HEADER"
    if cy >= (1 - EDGE) * H:
        return "FOOTER"
    if x1 <= EDGE * W:
        return "MARGIN_LEFT"
    if x0 >= (1 - EDGE) * W:
        return "MARGIN_RIGHT"
    return "BODY"


def read_pdf(pdf: pymupdf.Document) -> tuple[list[dict], str, str]:
    """Returns (doi candidates, head text, front text) for one open PDF."""
    cands, head, front = [], [], []

    def cand(doi, source, page, zone, marked):
        cands.append({"doi": doi, "source": source, "page": page, "zone": zone, "marked": marked})

    info = " ".join(str(v) for v in (pdf.metadata or {}).values() if v)
    xmp = pdf.get_xml_metadata() or ""
    for doi, marked in doi_matches(info + "\n" + xmp):
        cand(doi, "metadata", 0, "METADATA", marked)

    in_refs = False
    for number, page in enumerate(pdf, start=1):
        W, H = page.rect.width, page.rect.height
        if number <= LINK_PAGES:
            for link in page.get_links():
                for doi, _ in doi_matches(unquote(link.get("uri") or "")):
                    cand(doi, "link", number, "LINK", True)
        ordered = []
        for block in page.get_text("dict", sort=True)["blocks"]:
            if block.get("type") != 0:
                continue
            for line in block["lines"]:
                text = clean("".join(s["text"] for s in line["spans"]))
                if not text:
                    continue
                zone = zone_of(line["bbox"], line["dir"], W, H)
                if zone == "BODY" and REF_HEADING.match(text):
                    in_refs = True
                if zone == "BODY" and in_refs:
                    zone = "REFERENCES"
                ordered.append((zone, text))
        groups: dict[str, list[str]] = defaultdict(list)
        for zone, text in ordered:
            groups[zone].append(text)
        for zone, lines in groups.items():
            if number > FULL_SCAN_PAGES and zone not in EDGE_ZONES:
                continue
            for doi, marked in doi_matches("\n".join(lines)):
                cand(doi, "text", number, zone, marked)
        if number <= 2:
            page_text = "\n".join(t for _, t in ordered)
            front.append(page_text)
            if number == 1:
                head.append(page_text[:1500])
            head.append("\n".join(t for z, t in ordered if z in EDGE_ZONES))
    return cands, "\n".join(head), "\n".join(front)


TYPE_CUES = {
    "thesis": [r"\bthesis\b", r"\bdissertation\b", r"\bproefschrift\b", r"\bscriptie\b",
               r"\bafstudeer\w*", r"\bin partial fulfil\w*"],
    "policy_brief": [r"\bpolicy brief\b", r"\bpolicy paper\b", r"\bbeleidsnotitie\b",
                     r"\bbeleidsbrief\b", r"\bfact ?sheet\b"],
    "conference_paper": [r"\bproceedings\b", r"\bconference\b", r"\bcongress\b", r"\bsymposium\b",
                         r"\bworkshop\b", r"\bconferentie\b", r"\bcongres\b"],
    "journal_article": [r"\bjournal\b", r"\btijdschrift\b", r"\bissn\b", r"\bvol(?:ume|\.)\s*\d",
                        r"\breceived\b[^\n]{0,80}\baccepted\b", r"\bpublished by\b",
                        r"\b(?:elsevier|springer|wiley|mdpi|sage|frontiers)\b",
                        r"\btaylor\s*&\s*francis\b", r"\bcorresponding author\b", r"\bopen access\b"],
    "report": [r"\brapport\w*", r"\breport\b", r"\beindrapportage\b", r"\bdeliverable\b",
               r"\bprojectplan\b", r"\bcolofon\b", r"\bcolophon\b", r"\bvoorwoord\b",
               r"\bopdrachtgever\b", r"\binhoudsopgave\b", r"\bmanagementsamenvatting\b"],
}
TYPE_ORDER = ["thesis", "policy_brief", "conference_paper", "journal_article", "report"]


def classify_type(head: str, front: str) -> tuple[str, str]:
    scores, cues = {}, {}
    for doc_type, patterns in TYPE_CUES.items():
        points, hits = 0, []
        for pattern in patterns:
            if re.search(pattern, head, re.I):
                points += 3; hits.append(pattern.replace(r"\b", ""))
            elif re.search(pattern, front, re.I):
                points += 1; hits.append(pattern.replace(r"\b", "") + " (front)")
        scores[doc_type], cues[doc_type] = points, hits
    best = max(TYPE_ORDER, key=lambda t: (scores[t], -TYPE_ORDER.index(t)))
    if scores[best] < 2:
        return "other", ""
    return best, "; ".join(cues[best])


KEYWORD_LINE = re.compile(
    r"^\s*(?:key\s*words?|keywords?|index terms|trefwoorden|sleutelwoorden)\s*[:—\-–]\s*(.+)$", re.I | re.M)


def author_keywords(front: str) -> list[str]:
    """Keywords the authors printed themselves ("Keywords: a; b; c"), first match only."""
    m = KEYWORD_LINE.search(front)
    if not m:
        return []
    # The list can wrap onto the next lines; stop at a blank line or a heading-like line.
    tail = front[m.start(1):].splitlines()
    parts = [tail[0]]
    for line in tail[1:4]:
        s = line.strip()
        if not s or re.match(r"^(?:\d+\.?\s+)?[A-Z][a-z]+$", s) or len(s) > 120:
            break
        if not re.search(r"[;,·•]", s) and not parts[-1].rstrip().endswith((",", ";")):
            break
        parts.append(s)
    text = " ".join(parts)
    sep = ";" if ";" in text else ("·" if "·" in text else ("•" if "•" in text else ","))
    terms = [t.strip(" .;,·•") for t in text.split(sep)]
    return [t for t in terms if 1 < len(t) <= 80][:20]
