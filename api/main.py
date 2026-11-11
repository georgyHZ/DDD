"""Metadata API: read the collection database, extract new uploads.

Read routes serve the SQLite file built by notebook 08 (output/api/metadata.db).
The API never writes to it: notebooks stay the only writer, so a rerun of
notebook 08 atomically replaces the database and the API picks it up
on the next request.

POST /v1/extract takes a new PDF, runs the extraction pipeline (api/extraction,
Ministral 3B on a CUDA GPU) and returns a record in the same `Document` shape,
graded with the same notebook 08 rules (api/consolidate.py), plus generated
`suggested_keywords`. The result is returned only, not stored.

Run from the project root:
    uvicorn api.main:app --reload            (read routes only: EXTRACTION_ENABLED=0)
    docker compose up -d --build             (everything, with GPU)

Environment variables (all optional):
    METADATA_DB        path to the SQLite file
    METADATA_API_KEY   if set, every /v1 request needs header X-API-Key
    CORS_ORIGINS       comma-separated list of allowed browser origins
    EXTRACTION_ENABLED 1 = load the model and enable /v1/extract (default 0 outside Docker)
    MODEL_DIR          local Ministral folder (else downloaded from Hugging Face)
    LID_MODEL          path to fastText lid.176.bin
    MAX_UPLOAD_MB      upload limit for /v1/extract (default 100)
    DEMO_PAGES         1 = serve the browser upload page at / (default 1)
"""
from __future__ import annotations

import html
import json
import logging
import os
import secrets
import sqlite3
import threading
import traceback
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Iterator, Literal, Optional

from fastapi import APIRouter, Depends, FastAPI, File, HTTPException, Query, Security, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.security import APIKeyHeader
from pydantic import BaseModel

DB_PATH = Path(os.environ.get(
    "METADATA_DB",
    Path(__file__).resolve().parents[1] / "output" / "api" / "metadata.db",
))
API_KEY = os.environ.get("METADATA_API_KEY")

Confidence = Literal["high", "medium", "low"]
FIELDS = ["title", "authors", "doi", "publication_year",
          "document_type", "language", "keywords"]


# ---------------------------------------------------------------- models
class FieldProvenance(BaseModel):
    source: Optional[str] = None
    confidence: Optional[Confidence] = None


class Extraction(BaseModel):
    model: Optional[str] = None
    parse_status: Optional[str] = None
    rejected_fields: list[str] = []


class Document(BaseModel):
    document_id: str
    file_name: Optional[str] = None
    publinova_url: Optional[str] = None
    page_count: Optional[int] = None
    title: Optional[str] = None
    authors: list[str] = []
    doi: Optional[str] = None
    publication_year: Optional[int] = None
    document_type: Optional[str] = None
    language: Optional[str] = None
    keywords: list[str] = []
    review_status: Literal["ok", "needs_review"]
    review_flags: list[str] = []
    provenance: dict[str, FieldProvenance]
    extraction: Extraction


class ExtractResult(Document):
    """A freshly extracted upload: the database fields plus what only the extraction produces."""
    suggested_keywords: list[str] = []
    organisations: list[str] = []
    known_in_database: bool = False
    processing: dict = {}


class DocumentPage(BaseModel):
    total: int
    limit: int
    offset: int
    items: list[Document]


class CountItem(BaseModel):
    value: str
    count: int


# ------------------------------------------------------------------- db
def get_db() -> Iterator[sqlite3.Connection]:
    if not DB_PATH.is_file():
        raise HTTPException(503, "Metadata database not built yet. Run notebook 08.")
    con = sqlite3.connect(f"{DB_PATH.as_uri()}?mode=ro", uri=True,
                          check_same_thread=False)
    con.row_factory = sqlite3.Row
    try:
        yield con
    finally:
        con.close()


api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def require_key(key: Optional[str] = Security(api_key_header)) -> None:
    if API_KEY and not secrets.compare_digest(key or "", API_KEY):
        raise HTTPException(401, "Missing or invalid API key.")


def row_to_document(row: sqlite3.Row) -> Document:
    return Document(
        document_id=row["document_id"],
        file_name=row["file_name"],
        publinova_url=row["publinova_url"],
        page_count=row["page_count"],
        title=row["title"],
        authors=json.loads(row["authors_json"] or "[]"),
        doi=row["doi"],
        publication_year=row["publication_year"],
        document_type=row["document_type"],
        language=row["language"],
        keywords=json.loads(row["keywords_json"] or "[]"),
        review_status=row["review_status"],
        review_flags=json.loads(row["review_flags_json"] or "[]"),
        provenance={f: FieldProvenance(source=row[f"{f}_source"],
                                       confidence=row[f"{f}_confidence"])
                    for f in FIELDS},
        extraction=Extraction(
            model=row["llm_model"], parse_status=row["llm_parse_status"],
            rejected_fields=json.loads(row["llm_rejected_json"] or "[]")),
    )


def like(text: str) -> str:
    escaped = text.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
    return f"%{escaped}%"


def normalise_doi(value: str) -> str:
    value = value.strip().lower()
    for prefix in ("https://doi.org/", "http://doi.org/",
                   "https://dx.doi.org/", "http://dx.doi.org/", "doi:"):
        if value.startswith(prefix):
            value = value[len(prefix):]
    return value


RANK = "(CASE {c} WHEN 'high' THEN 3 WHEN 'medium' THEN 2 WHEN 'low' THEN 1 ELSE 0 END)"
CORE_CONFIDENCE = "MIN({}, {}, {})".format(
    RANK.format(c="title_confidence"), RANK.format(c="authors_confidence"),
    RANK.format(c="publication_year_confidence"))
SORTS = {
    "title": "title IS NULL, title COLLATE NOCASE ASC",
    "-title": "title IS NULL, title COLLATE NOCASE DESC",
    "year": "publication_year IS NULL, publication_year ASC",
    "-year": "publication_year IS NULL, publication_year DESC",
}

# ------------------------------------------------------------ extraction
log = logging.getLogger("metadata-api")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
EXTRACTION_ENABLED = os.environ.get("EXTRACTION_ENABLED", "0") == "1"
DEMO_PAGES = os.environ.get("DEMO_PAGES", "1") == "1"
MAX_UPLOAD_MB = int(os.environ.get("MAX_UPLOAD_MB", "100"))


class Extractor:
    """Holds the models; loaded once in a background thread so the read routes start at once."""
    status = "disabled"            # disabled | loading | ready | error
    error: Optional[str] = None
    model = None
    detector = None


def _load_extractor() -> None:
    try:
        from .extraction.llm import MinistralModel
        from .extraction.pdf_text import LanguageDetector
        lid = os.environ.get("LID_MODEL", "/models/lid.176.bin")
        log.info("Loading fastText language model from %s", lid)
        Extractor.detector = LanguageDetector(lid)
        Extractor.model = MinistralModel(os.environ.get("MODEL_DIR", "/models/Ministral-3-3B-Instruct-2512-BF16"))
        log.info("Loading Ministral (4-bit)")
        Extractor.model.load()
        Extractor.status = "ready"
        log.info("Extraction ready on %s after %ss (%s)", Extractor.model.device,
                 Extractor.model.load_seconds, Extractor.model.source)
    except Exception as exc:  # noqa: BLE001 - surface any load failure on /health
        Extractor.status, Extractor.error = "error", f"{type(exc).__name__}: {exc}"
        log.error("Extraction model failed to load:\n%s", traceback.format_exc())


@asynccontextmanager
async def lifespan(_: FastAPI):
    if EXTRACTION_ENABLED:
        Extractor.status = "loading"
        threading.Thread(target=_load_extractor, daemon=True).start()
    yield


def known_in_database(document_id: str) -> bool:
    if not DB_PATH.is_file():
        return False
    with sqlite3.connect(f"{DB_PATH.as_uri()}?mode=ro", uri=True) as con:
        return con.execute("SELECT 1 FROM documents WHERE document_id = ?", (document_id,)).fetchone() is not None


def run_extraction(file: UploadFile) -> ExtractResult:
    if Extractor.status == "disabled":
        raise HTTPException(503, "Extraction is disabled on this server (EXTRACTION_ENABLED=0).")
    if Extractor.status == "loading":
        raise HTTPException(503, "The extraction model is still loading. Try again in a minute (see /health).")
    if Extractor.status == "error":
        raise HTTPException(503, f"The extraction model failed to load: {Extractor.error}")
    data = file.file.read(MAX_UPLOAD_MB * 1024 * 1024 + 1)
    if len(data) > MAX_UPLOAD_MB * 1024 * 1024:
        raise HTTPException(413, f"The file is larger than {MAX_UPLOAD_MB} MB.")
    if not data:
        raise HTTPException(400, "The uploaded file is empty.")
    if b"%PDF-" not in data[:1024]:
        raise HTTPException(400, "The uploaded file is not a PDF.")
    from .extraction.pipeline import PipelineError, analyse
    name = os.path.basename(file.filename or "upload.pdf")
    log.info("Extracting %s (%.1f MB)", name, len(data) / 1e6)
    try:
        result = analyse(data, name, Extractor.model, Extractor.detector)
    except PipelineError as exc:
        raise HTTPException(422, str(exc)) from exc
    result["known_in_database"] = known_in_database(result["document_id"])
    log.info("Done %s: %d suggested keywords in %ss", name, len(result["suggested_keywords"]),
             result["processing"]["timing_seconds"]["total"])
    return ExtractResult(**result)


# ------------------------------------------------------------------ app
app = FastAPI(
    title="Publication metadata API",
    version="1.1.0",
    description=("Bibliographic metadata extracted from the PDF collection with an "
                 "LLM, checked against the page text. Every field carries its "
                 "`source` and `confidence` under `provenance`. `POST /v1/extract` "
                 "processes a new PDF the same way and adds generated `suggested_keywords`."),
    lifespan=lifespan,
)
if os.environ.get("CORS_ORIGINS"):
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[o.strip() for o in os.environ["CORS_ORIGINS"].split(",")],
        allow_methods=["GET", "POST"], allow_headers=["X-API-Key"])

router = APIRouter(prefix="/v1", dependencies=[Depends(require_key)])


@app.get("/health", tags=["meta"])
def health() -> dict:
    extraction = {"status": Extractor.status, "error": Extractor.error,
                  "gpu": getattr(Extractor.model, "device", None),
                  "model_source": getattr(Extractor.model, "source", None)}
    if not DB_PATH.is_file():
        return {"status": "no_database", "extraction": extraction}
    with sqlite3.connect(f"{DB_PATH.as_uri()}?mode=ro", uri=True) as con:
        n = con.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
        built = con.execute("SELECT value FROM meta WHERE key='built_at'").fetchone()
    return {"status": "ok", "documents": n, "built_at": built[0] if built else None, "extraction": extraction}


@router.post("/extract", response_model=ExtractResult, tags=["extraction"])
def extract(file: UploadFile = File(..., description="A PDF publication (English or Dutch)")) -> ExtractResult:
    """Extract metadata and keyword suggestions from an uploaded PDF. Nothing is stored."""
    return run_extraction(file)


@router.get("/documents", response_model=DocumentPage, tags=["documents"])
def list_documents(
    db: sqlite3.Connection = Depends(get_db),
    q: Optional[str] = Query(None, description="Title contains (case-insensitive)."),
    author: Optional[str] = Query(None, description="Author name contains."),
    keyword: Optional[str] = Query(None, description="Exact keyword (case-insensitive)."),
    doi: Optional[str] = None,
    year: Optional[int] = None,
    year_from: Optional[int] = None,
    year_to: Optional[int] = None,
    language: Optional[Literal["en", "nl", "mixed"]] = None,
    document_type: Optional[str] = None,
    min_confidence: Optional[Confidence] = Query(
        None, description="Minimum confidence of title, authors AND publication_year."),
    review_status: Optional[Literal["ok", "needs_review"]] = None,
    sort: Literal["title", "-title", "year", "-year"] = "title",
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
) -> DocumentPage:
    where, args = [], []
    if q:
        where.append("title LIKE ? ESCAPE '\\'"); args.append(like(q))
    if author:
        where.append("EXISTS (SELECT 1 FROM authors a WHERE a.document_id = documents.document_id "
                     "AND a.name LIKE ? ESCAPE '\\')"); args.append(like(author))
    if keyword:
        where.append("EXISTS (SELECT 1 FROM keywords k WHERE k.document_id = documents.document_id "
                     "AND k.keyword = ? COLLATE NOCASE)"); args.append(keyword)
    if doi:
        where.append("doi = ?"); args.append(normalise_doi(doi))
    if year is not None:
        where.append("publication_year = ?"); args.append(year)
    if year_from is not None:
        where.append("publication_year >= ?"); args.append(year_from)
    if year_to is not None:
        where.append("publication_year <= ?"); args.append(year_to)
    if language:
        where.append("language = ?"); args.append(language)
    if document_type:
        where.append("document_type = ?"); args.append(document_type)
    if review_status:
        where.append("review_status = ?"); args.append(review_status)
    if min_confidence:
        where.append(f"{CORE_CONFIDENCE} >= ?")
        args.append({"high": 3, "medium": 2, "low": 1}[min_confidence])
    clause = f"WHERE {' AND '.join(where)}" if where else ""
    total = db.execute(f"SELECT COUNT(*) FROM documents {clause}", args).fetchone()[0]
    rows = db.execute(
        f"SELECT * FROM documents {clause} ORDER BY {SORTS[sort]}, document_id LIMIT ? OFFSET ?",
        [*args, limit, offset]).fetchall()
    return DocumentPage(total=total, limit=limit, offset=offset,
                        items=[row_to_document(r) for r in rows])


@router.get("/documents/by-doi/{doi:path}", response_model=Document, tags=["documents"])
def get_document_by_doi(doi: str, db: sqlite3.Connection = Depends(get_db)) -> Document:
    row = db.execute("SELECT * FROM documents WHERE doi = ?", (normalise_doi(doi),)).fetchone()
    if not row:
        raise HTTPException(404, "No document with this DOI.")
    return row_to_document(row)


@router.get("/documents/{document_id}", response_model=Document, tags=["documents"])
def get_document(document_id: str, db: sqlite3.Connection = Depends(get_db)) -> Document:
    row = db.execute("SELECT * FROM documents WHERE document_id = ?", (document_id,)).fetchone()
    if not row:
        raise HTTPException(404, "Unknown document_id.")
    return row_to_document(row)


@router.get("/authors", response_model=list[CountItem], tags=["facets"])
def list_authors(db: sqlite3.Connection = Depends(get_db), q: Optional[str] = None,
                 limit: int = Query(100, ge=1, le=500)) -> list[CountItem]:
    clause, args = ("WHERE name LIKE ? ESCAPE '\\'", [like(q)]) if q else ("", [])
    rows = db.execute(
        f"SELECT name AS value, COUNT(*) AS count FROM authors {clause} "
        "GROUP BY name COLLATE NOCASE ORDER BY count DESC, value LIMIT ?", [*args, limit]).fetchall()
    return [CountItem(**dict(r)) for r in rows]


@router.get("/keywords", response_model=list[CountItem], tags=["facets"])
def list_keywords(db: sqlite3.Connection = Depends(get_db), q: Optional[str] = None,
                  limit: int = Query(100, ge=1, le=500)) -> list[CountItem]:
    clause, args = ("WHERE keyword LIKE ? ESCAPE '\\'", [like(q)]) if q else ("", [])
    rows = db.execute(
        f"SELECT keyword AS value, COUNT(*) AS count FROM keywords {clause} "
        "GROUP BY keyword COLLATE NOCASE ORDER BY count DESC, value LIMIT ?", [*args, limit]).fetchall()
    return [CountItem(**dict(r)) for r in rows]


@router.get("/stats", tags=["meta"])
def stats(db: sqlite3.Connection = Depends(get_db)) -> dict:
    scalar_cols = {"title": "title", "authors": "authors_json", "doi": "doi",
                   "publication_year": "publication_year", "document_type": "document_type",
                   "language": "language", "keywords": "keywords_json"}
    total = db.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
    filled = {}
    for field, col in scalar_cols.items():
        empty = "'[]'" if col.endswith("_json") else "''"
        filled[field] = db.execute(
            f"SELECT COUNT(*) FROM documents WHERE {col} IS NOT NULL AND {col} != {empty}").fetchone()[0]
    sources = {f: {r[0] or "none": r[1] for r in db.execute(
        f"SELECT {f}_source, COUNT(*) FROM documents GROUP BY {f}_source")} for f in FIELDS}
    confidence = {f: {r[0] or "none": r[1] for r in db.execute(
        f"SELECT {f}_confidence, COUNT(*) FROM documents GROUP BY {f}_confidence")} for f in FIELDS}

    def counts(col: str) -> dict:
        return {str(r[0]): r[1] for r in db.execute(
            f"SELECT {col}, COUNT(*) FROM documents GROUP BY {col} ORDER BY {col}")}

    return {
        "documents": total,
        "fields_filled": filled,
        "sources": sources,
        "confidence": confidence,
        "review_status": counts("review_status"),
        "languages": counts("language"),
        "years": counts("publication_year"),
        "build": {r[0]: r[1] for r in db.execute("SELECT key, value FROM meta")},
    }


app.include_router(router)


# ---------------------------------------------------------- browser demo pages
PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>{refresh}
<style>
 body {{ font-family: system-ui, sans-serif; max-width: 900px; margin: 2rem auto; padding: 0 1rem; color: #1d1d1f; }}
 pre {{ background: #f4f4f6; padding: 1rem; border-radius: 6px; overflow-x: auto; white-space: pre-wrap; }}
 .muted {{ color: #666; }} .err {{ color: #b00020; }}
 button {{ padding: .5rem 1.2rem; font-size: 1rem; }}
</style></head><body>{body}</body></html>"""


def _page(title: str, body: str, refresh: bool = False, status: int = 200) -> HTMLResponse:
    meta = '<meta http-equiv="refresh" content="10">' if refresh else ""
    return HTMLResponse(PAGE.format(title=html.escape(title), refresh=meta, body=body), status_code=status)


if DEMO_PAGES:
    @app.get("/", response_class=HTMLResponse, include_in_schema=False)
    def index() -> HTMLResponse:
        state = {
            "ready": f'<p class="muted">Model ready on {html.escape(getattr(Extractor.model, "device", None) or "GPU")}.</p>',
            "loading": '<p class="muted">The model is loading (about a minute after start). This page refreshes itself.</p>',
            "error": f'<p class="err">The model failed to load: {html.escape(Extractor.error or "")}</p>',
            "disabled": '<p class="err">Extraction is disabled on this server (EXTRACTION_ENABLED=0).</p>',
        }[Extractor.status]
        body = f"""<h1>Keyword extraction</h1>
<p>Upload a research publication (PDF, English or Dutch). Processing takes roughly 20&ndash;70 seconds.</p>
{state}
<form action="/extract" method="post" enctype="multipart/form-data"
      onsubmit="this.querySelector('button').disabled=true;document.getElementById('wait').hidden=false;">
  <p><input type="file" name="file" accept="application/pdf,.pdf" required></p>
  <p><button type="submit" {'disabled' if Extractor.status != 'ready' else ''}>Extract keywords</button></p>
  <p id="wait" class="muted" hidden>Processing&hellip; keep this tab open.</p>
</form>
<p class="muted">API: <code>POST /v1/extract</code> (form field <code>file</code>) &middot;
<code>GET /v1/documents</code> &middot; <a href="/docs">/docs</a> &middot; <a href="/health">/health</a></p>"""
        return _page("Keyword extraction", body, refresh=Extractor.status == "loading")

    @app.post("/extract", response_class=HTMLResponse, include_in_schema=False)
    def extract_page(file: UploadFile = File(...)) -> HTMLResponse:
        try:
            result = run_extraction(file)
        except HTTPException as exc:
            body = (f'<h1>Could not process the file</h1><p class="err">{html.escape(str(exc.detail))}</p>'
                    '<p><a href="/">Upload another PDF</a></p>')
            return _page("Error", body, status=exc.status_code)
        pretty = html.escape(json.dumps(result.model_dump(), indent=2, ensure_ascii=False))
        body = f'<h1>Result</h1><p><a href="/">Upload another PDF</a></p><pre>{pretty}</pre>'
        return _page(f"Result - {result.file_name}", body)
