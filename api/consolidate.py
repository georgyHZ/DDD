"""Field consolidation and confidence rules from notebook 08 (consolidate_and_build_database).

Moved out of the notebook so the API's upload route (`POST /v1/extract`) grades a freshly
uploaded PDF with exactly the same rules as the documents in `metadata.db`:

* `high`   - LLM value found (exactly) in the page text
* `medium` - found approximately/partially, or taken from a regex / PDF-embedded / rule fallback
* `low`    - not verifiable (e.g. document_type) or kept although ungrounded

Changes compared with the notebook (marked "API addition" below):
* `resolve()` takes its inputs as arguments instead of reading the notebook's global tables.
* Optional `rule_doi`: the scored DOI from the rule-based harvester (notebook 08 on main,
  DOI/type extraction). It is used after an LLM DOI and before the plain page-text regex, which
  can pick up a cited DOI. Source `rules`.
* Optional `rule_type`: the cue-based document type from the same notebook, used when the LLM
  gives none. Source `rules`.
"""
from __future__ import annotations

import datetime
import difflib
import json
import re
import unicodedata

KEEP_UNGROUNDED = False     # True publishes LLM values that are NOT found in the page text (confidence 'low')
GROUNDING_PAGES = 4         # same pages the extraction prompt saw
MIN_TITLE_COVERAGE = 0.90   # share of the title that must be found in the page text
DOC_TYPES = ['journal_article', 'report', 'thesis', 'conference_paper', 'book_chapter', 'policy_brief', 'other']
FIELDS = ['title', 'authors', 'doi', 'publication_year', 'document_type', 'language', 'keywords']
CURRENT_YEAR = datetime.date.today().year

# ---------------------------------------------------------------- cleaning and grounding helpers
NULL_STRINGS = {'', 'null', 'none', 'n/a', 'na', 'nan', 'undefined', 'unknown', '-', '[]', '{}'}


def squash(s):
    s = unicodedata.normalize('NFKD', str(s)).lower()
    return re.sub(r'[^a-z0-9]+', '', ''.join(c for c in s if not unicodedata.combining(c)))


def is_null(v):
    if v is None:
        return True
    if isinstance(v, str):
        return v.strip().lower() in NULL_STRINGS
    if isinstance(v, (list, dict)):
        return len(v) == 0
    return False


def norm_str(v):
    if is_null(v):
        return None
    s = re.sub(r'\s+', ' ', str(v)).strip()
    return None if s.lower() in NULL_STRINGS else s


def norm_list(v):
    if is_null(v):
        return []
    if isinstance(v, str):
        parts = re.split(r'[;|\n]', v)
        if len(parts) == 1 and re.search(r'\s(?:and|en|&)\s', v):
            parts = re.split(r'\s(?:and|en|&)\s', v)
        v = parts
    elif isinstance(v, dict):
        v = [v]
    items = []
    for x in v:
        if isinstance(x, dict):
            x = (x.get('name') or x.get('full_name')
                 or ' '.join(str(p) for p in (x.get('given'), x.get('family')) if p))
        s = norm_str(x)
        if s and s not in items:
            items.append(s)
    return items


def clean_author(name):
    name = re.sub(r'[\*†‡§\d]+$', '', name).strip(' ,;')
    parts = [p.strip() for p in name.split(',')]
    if (len(parts) == 2 and all(re.search(r'[^\W\d_]', p) for p in parts)
            and not re.fullmatch(r'(?i)(jr|sr|phd|md)\.?', parts[1])):
        name = f'{parts[1]} {parts[0]}'          # "Buijs, Jean-Marie" -> "Jean-Marie Buijs"
    return name or None


DOI_RE = re.compile(r'10\.\d{4,9}/[^\s"<>]+', re.I)


def clean_doi(v):
    s = norm_str(v)
    if not s:
        return None
    s = re.sub(r'(?i)^(https?://(dx\.)?doi\.org/|doi:\s*)', '', s)
    m = DOI_RE.search(s)
    if not m:
        return None
    d = m.group(0).rstrip('.,;]')
    while d.endswith(')') and d.count(')') > d.count('('):
        d = d[:-1]
    return d.lower()


def clean_year(v):
    s = norm_str(v)
    m = re.search(r'\b(1[89]\d{2}|20\d{2})\b', s) if s else None
    y = int(m.group(1)) if m else None
    return y if y and 1900 <= y <= CURRENT_YEAR + 1 else None


TYPE_ALIASES = {'article': 'journal_article', 'journal': 'journal_article', 'artikel': 'journal_article',
                'paper': 'conference_paper', 'proceedings': 'conference_paper',
                'chapter': 'book_chapter', 'rapport': 'report', 'dissertation': 'thesis',
                'proefschrift': 'thesis', 'master_thesis': 'thesis', 'bachelor_thesis': 'thesis',
                'brief': 'policy_brief'}


def norm_type(v):
    s = norm_str(v)
    if not s:
        return None
    s = re.sub(r'[\s\-]+', '_', s.lower())
    return s if s in DOC_TYPES else TYPE_ALIASES.get(s)


LANG_ALIASES = {'english': 'en', 'en-us': 'en', 'en-gb': 'en', 'eng': 'en',
                'dutch': 'nl', 'nederlands': 'nl', 'nld': 'nl', 'dut': 'nl',
                'mixed': 'mixed', 'bilingual': 'mixed', 'en': 'en', 'nl': 'nl'}


def norm_lang(v):
    s = norm_str(v)
    return LANG_ALIASES.get(s.lower()) if s else None


def coverage(needle, hay_sq):
    n = squash(needle)
    if not n or not hay_sq:
        return 0.0
    if n in hay_sq:
        return 1.0
    m = difflib.SequenceMatcher(None, hay_sq, n, autojunk=False).find_longest_match(0, len(hay_sq), 0, len(n))
    return m.size / len(n)


def author_grounded(name, hay_sq):
    tokens = re.findall(r"[^\W\d_]{2,}", name)
    return bool(tokens) and all(squash(t) in hay_sq for t in tokens)


def junk_title(t):
    low = t.lower()
    return (len(t) < 8 or low.startswith('microsoft word') or low in {'untitled', 'undefined'}
            or low.endswith(('.doc', '.docx', '.pdf', '.indd', '.tex')))


def year_from_text(text):
    tiers = [re.compile(r'(?i)published|available online|gepubliceerd|publicatiedatum'),
             re.compile(r'(?i)©|copyright')]
    for pat in tiers:
        for line in text.splitlines():
            if pat.search(line):
                ys = [int(y) for y in re.findall(r'\b(19\d{2}|20\d{2})\b', line) if int(y) <= CURRENT_YEAR + 1]
                if ys:
                    return max(ys)
    return None


KEYWORD_LINE = re.compile(r'(?im)^\s*(?:key\s*words?|trefwoorden|sleutelwoorden)\s*[:\-–—]\s*(.+)$')


# ---------------------------------------------------------------- parsing the raw LLM answer
def salvage(text):
    out = {}
    for key in ('title', 'doi', 'document_type', 'language'):
        m = re.search(rf'"{key}"\s*:\s*"((?:[^"\\]|\\.)*)"', text)
        if m:
            try:
                out[key] = json.loads('"' + m.group(1) + '"')
            except json.JSONDecodeError:
                out[key] = m.group(1)
    m = re.search(r'"publication_year"\s*:\s*"?(\d{4})', text)
    if m:
        out['publication_year'] = m.group(1)
    for key in ('authors', 'keywords'):
        m = re.search(rf'"{key}"\s*:\s*\[(.*?)(?:\]|$)', text, re.S)
        if m:
            out[key] = [s for s in re.findall(r'"((?:[^"\\]|\\.)*)"', m.group(1))]
    return out


def parse_llm(raw):
    """Returns (fields, status) with status json | salvaged | failed."""
    text = re.sub(r'```(?:json)?', '', raw or '').strip()
    candidates = []
    start = text.find('{')
    if start >= 0:
        candidates.append(text[start:])
    elif text.startswith('"'):
        candidates.append('{' + text)
    for cand in candidates:
        try:
            data, _ = json.JSONDecoder().raw_decode(cand)
            if isinstance(data, dict):
                return data, 'json'
        except json.JSONDecodeError:
            pass
    data = salvage(text)
    return (data, 'salvaged') if data else ({}, 'failed')


# ---------------------------------------------------------------- resolving one document
def R(value, source, confidence):
    return {'value': value, 'source': source, 'confidence': confidence}


EMPTY = R(None, None, None)

TYPE_SIGNALS = {'thesis': r'thesis|proefschrift|scriptie|dissertation', 'report': r'\breport\b|rapport',
                'conference_paper': r'proceedings|conference|congres', 'policy_brief': r'policy brief',
                'book_chapter': r'\bchapter\b|hoofdstuk', 'journal_article': r'\bjournal\b|tijdschrift'}


def fasttext_language(share):
    """share = {'en': fraction, 'nl': fraction} of reliably classified text (or None)."""
    if not share:
        return None, 0.0
    en, nl = share.get('en', 0.0), share.get('nl', 0.0)
    if en >= 0.9: return 'en', en
    if nl >= 0.9: return 'nl', nl
    if en >= 0.1 and nl >= 0.1: return 'mixed', max(en, nl)
    return ('en', en) if en >= nl else ('nl', nl)


def resolve(pages, llm=None, parse_status='no_llm_output', obs=None, emb_keywords=None,
            lang_share=None, text_status=None, rule_doi=None, rule_type=None):
    """Consolidate one document.

    pages        page texts, first page first (only GROUNDING_PAGES are used)
    llm          parsed LLM answer (dict) and its parse_status
    obs          candidate values {'title': [...], 'author': [...], 'doi': [...], 'online_date': [...]}
    emb_keywords keywords embedded in the PDF properties
    lang_share   {'en': x, 'nl': y} fastText shares
    rule_doi     API addition: {'doi': str, 'score': int} from the rule-based DOI harvester
    rule_type    API addition: (document_type, cues) from the rule-based type classifier
    Returns (fields, review_flags, rejected_fields).
    """
    llm, obs, emb_keywords = llm or {}, obs or {}, emb_keywords or []
    pgs = list(pages[:GROUNDING_PAGES])
    text = '\n'.join(pgs)
    sq = squash(text)
    rejected, out = [], {}

    # ---- title
    title = EMPTY
    t = norm_str(llm.get('title'))
    if t:
        cov = coverage(t, sq)
        if cov >= 0.999: title = R(t, 'llm', 'high')
        elif cov >= MIN_TITLE_COVERAGE: title = R(t, 'llm', 'medium')
        elif KEEP_UNGROUNDED: title = R(t, 'llm', 'low'); rejected.append('title')
        else: rejected.append('title')
    if title['value'] is None:
        for e in obs.get('title', []):
            e = norm_str(e)
            if e and not junk_title(e) and coverage(e, sq) >= MIN_TITLE_COVERAGE:
                title = R(e, 'pdf_embedded', 'medium'); break
    out['title'] = title

    # ---- authors
    def grade_authors(names, source):
        names = [n for n in (clean_author(x) for x in names) if n]
        good = [n for n in names if author_grounded(n, sq)]
        if names and len(good) == len(names): return R(good, source, 'high'), False
        if good: return R(good, source, 'medium'), True
        if names and KEEP_UNGROUNDED: return R(names, source, 'low'), True
        return EMPTY, bool(names)
    authors, dropped = grade_authors(norm_list(llm.get('authors')), 'llm')
    if dropped: rejected.append('authors')
    if authors['value'] is None:
        authors, _ = grade_authors(norm_list(';'.join(obs.get('author', []))), 'pdf_embedded')
        if authors['value']: authors['confidence'] = 'medium'
    out['authors'] = authors

    # ---- doi
    doi = EMPTY
    x = clean_doi(llm.get('doi'))
    if x:
        if squash(x) in sq: doi = R(x, 'llm', 'high')
        elif KEEP_UNGROUNDED: doi = R(x, 'llm', 'low'); rejected.append('doi')
        else: rejected.append('doi')
    if doi['value'] is None and rule_doi and clean_doi(rule_doi.get('doi')):   # API addition
        doi = R(clean_doi(rule_doi['doi']), 'rules', 'high' if (rule_doi.get('score') or 0) >= 7 else 'medium')
    if doi['value'] is None:
        for pg in pgs:                          # page 1 first: later pages often cite other papers
            m = DOI_RE.search(pg)
            if m:
                doi = R(clean_doi(m.group(0)), 'regex', 'medium'); break
    if doi['value'] is None:
        for e in obs.get('doi', []):
            if clean_doi(e):
                doi = R(clean_doi(e), 'regex', 'medium'); break
    out['doi'] = doi

    # ---- publication year
    year = EMPTY
    y, ty = clean_year(llm.get('publication_year')), year_from_text(text)
    if y:
        if ty == y: year = R(y, 'llm', 'high')
        elif str(y) in text: year = R(y, 'llm', 'medium')
        elif KEEP_UNGROUNDED: year = R(y, 'llm', 'low'); rejected.append('publication_year')
        else: rejected.append('publication_year')
    if year['value'] is None and ty:
        year = R(ty, 'regex', 'medium')
    if year['value'] is None:
        for e in obs.get('online_date', []):
            oy = clean_year(e)
            if oy:
                year = R(oy, 'regex', 'medium'); break
    out['publication_year'] = year

    # ---- document type (not verifiable from text alone: corroborate with simple signals)
    dtype = EMPTY
    ty_llm = norm_type(llm.get('document_type'))
    if ty_llm:
        corroborated = bool(re.search(TYPE_SIGNALS.get(ty_llm, r'(?!x)x'), pgs[0] if pgs else '', re.I)
                            or (ty_llm == 'journal_article' and doi['value']))
        dtype = R(ty_llm, 'llm', 'medium' if corroborated else 'low')
    elif not is_null(llm.get('document_type')):
        rejected.append('document_type')
    if dtype['value'] is None and rule_type and rule_type[0] in DOC_TYPES and rule_type[0] != 'other':  # API addition
        cues = rule_type[1] or ''
        head_cue = any(c and not c.endswith('(front)') for c in cues.split('; '))
        dtype = R(rule_type[0], 'rules', 'medium' if head_cue else 'low')
    out['document_type'] = dtype

    # ---- language: LLM, cross-checked with fastText
    lang = EMPTY
    l_llm = norm_lang(llm.get('language'))
    l_ft, strength = fasttext_language(lang_share)
    if l_ft:
        if l_llm == l_ft: lang = R(l_ft, 'llm+fasttext', 'high')
        else:
            lang = R(l_ft, 'fasttext', 'high' if strength >= 0.9 else 'medium')
            if l_llm: rejected.append('language')
    elif l_llm:
        lang = R(l_llm, 'llm', 'low')
    out['language'] = lang

    # ---- keywords (the ones the authors printed or embedded; generated suggestions are separate)
    kws = [k for k in norm_list(llm.get('keywords')) if len(squash(k)) >= 3]
    good = [k for k in kws if squash(k) in sq]
    if good: keywords = R(good, 'llm', 'high' if len(good) == len(kws) else 'medium')
    elif kws and KEEP_UNGROUNDED: keywords = R(kws, 'llm', 'low')
    else: keywords = EMPTY
    if kws and len(good) < len(kws):
        rejected.append('keywords')
    if keywords['value'] is None:
        found = [k for m in KEYWORD_LINE.finditer(text) for k in re.split(r'[;,·•]', m.group(1)) if k.strip()]
        if found: keywords = R([k.strip() for k in found], 'regex', 'medium')
    if keywords['value'] is None and emb_keywords:
        keywords = R(norm_list(';'.join(emb_keywords)), 'pdf_embedded', 'medium')
    out['keywords'] = keywords

    # ---- review flags
    flags = []
    if text_status in ('no_text', 'error'): flags.append('no_text')
    if parse_status == 'no_llm_output': flags.append('no_llm_output')
    elif parse_status == 'failed': flags.append('llm_parse_failed')
    for f in ('title', 'authors', 'publication_year'):
        if out[f]['value'] in (None, []): flags.append(f'no_{f}')
        elif out[f]['confidence'] == 'low': flags.append(f'low_confidence_{f}')
    return out, flags, rejected
