#!/usr/bin/env python3
"""deepdive_kai0 showcase — public-facing demo/progress page.

Lightweight FastAPI single-process server. No ROS / CUDA / model imports —
runs anywhere with just `pip install fastapi uvicorn`.

Usage:
    python web/showcase/server.py [--port 8765] [--host 0.0.0.0]
"""
from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Optional

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles


# ── Paths ──
_web_dir = Path(__file__).resolve().parent
_project_root = _web_dir.parent.parent
_docs_root = _project_root / "docs"
_content_dir = _web_dir / "content"


# ── App ──
app = FastAPI(
    title="deepdive_kai0 Showcase",
    version="0.1.0",
    description="Public-facing project showcase for deepdive_kai0 (kai0/π0.5 deployment).",
)


# ── Static & templates ──
app.mount("/static", StaticFiles(directory=str(_web_dir / "static")), name="static")


@app.get("/", response_class=HTMLResponse)
async def index():
    return FileResponse(str(_web_dir / "templates" / "index.html"))


@app.get("/api/health")
async def health():
    return {
        "status": "ok",
        "version": app.version,
        "content": {
            "features": (_content_dir / "features.json").is_file(),
            "milestones": (_content_dir / "milestones.json").is_file(),
            "docs_index": (_content_dir / "docs_index.json").is_file(),
        },
        "docs_root": str(_docs_root),
    }


# ── Content endpoints (read JSON from content/) ──
def _read_json(name: str) -> dict:
    path = _content_dir / name
    if not path.is_file():
        raise HTTPException(status_code=404, detail=f"{name} not found")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=500, detail=f"{name} parse error: {e}")


@app.get("/api/features")
async def get_features():
    return _read_json("features.json")


@app.get("/api/milestones")
async def get_milestones():
    return _read_json("milestones.json")


@app.get("/api/docs/index")
async def get_docs_index():
    return _read_json("docs_index.json")


# ── Doc serving (markdown raw text) ──
# Resolves docs/<subdir>/<name>.md where subdir ∈ {deployment, training, security}.
# Sanitized name → no path traversal possible.
_DOC_NAME_RE = re.compile(r"^[a-zA-Z0-9_\-]+\.md$")
_DOC_SUBDIRS = ("deployment", "training", "security")


def _resolve_doc(name: str) -> Optional[Path]:
    if not _DOC_NAME_RE.match(name):
        return None
    for sub in _DOC_SUBDIRS:
        candidate = _docs_root / sub / name
        if candidate.is_file():
            return candidate
    # Root-level doc (e.g. project_complete_guide.md)
    candidate = _docs_root / name
    if candidate.is_file():
        return candidate
    return None


@app.get("/api/doc/{name}", response_class=PlainTextResponse)
async def get_doc(name: str):
    path = _resolve_doc(name)
    if path is None:
        raise HTTPException(status_code=404, detail=f"doc {name!r} not found")
    return PlainTextResponse(path.read_text(encoding="utf-8"))


@app.get("/api/readme", response_class=PlainTextResponse)
async def get_readme(lang: str = "zh"):
    candidates = []
    if lang == "en":
        candidates = ["README_en.md", "README.md"]
    else:
        candidates = ["README.md", "README_zh.md"]
    for name in candidates:
        path = _project_root / name
        if path.is_file():
            return PlainTextResponse(path.read_text(encoding="utf-8"))
    return PlainTextResponse(
        f"README ({lang}) not found at {_project_root}", status_code=404
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=int(os.environ.get("SHOWCASE_PORT", 8765)))
    parser.add_argument("--host", default=os.environ.get("SHOWCASE_HOST", "0.0.0.0"))
    parser.add_argument("--reload", action="store_true", help="dev: auto-reload on edit")
    args = parser.parse_args()

    print("=" * 60)
    print("  deepdive_kai0 Showcase")
    print(f"  http://{args.host}:{args.port}/")
    print(f"  docs_root  = {_docs_root}")
    print(f"  content    = {_content_dir}")
    print("=" * 60)

    uvicorn.run(
        "web.showcase.server:app" if args.reload else app,
        host=args.host, port=args.port, reload=args.reload,
        log_level="info",
    )


if __name__ == "__main__":
    main()
