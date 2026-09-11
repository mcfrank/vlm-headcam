"""Blind human check of the Gemini referential-alignment annotation.

The rater sees exactly what Gemini saw (one frame + the utterance) and answers yes/no to Gemini's
question at its 50-point anchor (object visible, even if small/partial/one of many). Gemini's own answers are NOT in the data dir; the analysis joins on item_id later.

State is entirely server-side, one JSON file per (rater, item) under
DATA_DIR/responses/<rater>/<item_id>.json, so a rater can stop and resume from any browser and
several raters can work at once (no shared file is ever rewritten). Items are handed out
fewest-ratings-first so coverage stays balanced, ties broken by a per-rater seeded permutation.

Rater identity: the IAP header when deployed on Cloud Run behind Identity-Aware Proxy; otherwise
the ?rater= name the client stores (ssh-tunnel deployment on ccn2)."""
import hashlib
import json
import os
import re
import time
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel

DATA = Path(os.environ.get("DATA_DIR", "data"))
STATIC = Path(__file__).parent / "static"
ITEMS = {it["id"]: it["text"] for it in json.loads((DATA / "items.json").read_text())}
ORDER = sorted(ITEMS)
RESP = DATA / "responses"
TARGET = int(os.environ.get("RATERS_PER_ITEM", 3))
ANSWERS = {"no": 0, "yes": 100, "cant_tell": None,
           "none": 0, "partial": 50, "clear": 100}  # last three: the pre-2026-09-11 three-level format

app = FastAPI()


def rater_id(request: Request, rater: str | None) -> str:
    iap = request.headers.get("x-goog-authenticated-user-email")  # "accounts.google.com:me@x"
    name = iap.split(":")[-1] if iap else (rater or "")
    slug = re.sub(r"[^a-z0-9._@-]", "_", name.strip().lower())
    if not slug:
        raise HTTPException(400, "rater name required")
    return slug


_CACHE: dict[str, dict[str, dict]] = {}   # rater -> item_id -> response
_CACHE_AT = 0.0
RESCAN_S = 120


def all_responses() -> dict[str, dict[str, dict]]:
    """Responses by rater. Kept in memory: on the Cloud Run bucket mount every file read is a
    GCS request, and rereading ~2k files per /api/state call made the app unusable once one
    rater had finished. Other instances' writes are picked up by rescanning the listing every
    RESCAN_S seconds and reading only files not yet cached."""
    global _CACHE_AT
    if time.time() - _CACHE_AT > RESCAN_S:
        if RESP.exists():
            for rdir in RESP.iterdir():
                if not rdir.is_dir():
                    continue
                have = _CACHE.setdefault(rdir.name, {})
                for p in rdir.glob("*.json"):
                    if p.stem in ITEMS and p.stem not in have:
                        try:
                            have[p.stem] = json.loads(p.read_text())
                        except (OSError, ValueError):
                            pass
        _CACHE_AT = time.time()
    return _CACHE


def queue_for(rater: str, resp: dict) -> list[str]:
    seed = int(hashlib.sha256(rater.encode()).hexdigest()[:8], 16)
    perm = np.random.default_rng(seed).permutation(len(ORDER))
    rank = {ORDER[i]: r for r, i in enumerate(perm)}
    done = resp.get(rater, {})
    counts = {i: 0 for i in ORDER}
    for r, d in resp.items():
        if r != rater:
            for i in d:
                counts[i] += 1
    todo = [i for i in ORDER if i not in done]
    return sorted(todo, key=lambda i: (counts[i], rank[i]))


def coverage(resp: dict) -> dict:
    n = {i: 0 for i in ORDER}
    for d in resp.values():
        for i in d:
            n[i] += 1
    vals = list(n.values())
    return {"target": TARGET, "items": len(ORDER), "complete": sum(v >= TARGET for v in vals),
            "mean": round(float(np.mean(vals)), 2),
            "raters": {r: len(d) for r, d in sorted(resp.items())}}


@app.get("/", response_class=HTMLResponse)
def index():
    return (STATIC / "index.html").read_text()


@app.get("/api/me")
def me(request: Request):
    iap = request.headers.get("x-goog-authenticated-user-email")
    return {"rater": iap.split(":")[-1].lower() if iap else None}


@app.get("/api/state")
def state(request: Request, rater: str | None = None):
    r = rater_id(request, rater)
    resp = all_responses()
    return {"rater": r, "total": len(ORDER), "done": resp.get(r, {}),
            "queue": queue_for(r, resp), "coverage": coverage(resp)}


@app.get("/api/item/{item_id}")
def item(item_id: str):
    if item_id not in ITEMS:
        raise HTTPException(404)
    return {"id": item_id, "text": ITEMS[item_id]}


@app.get("/frames/{item_id}.jpg")
def frame(item_id: str):
    if item_id not in ITEMS:
        raise HTTPException(404)
    return FileResponse(DATA / "frames" / f"{item_id}.jpg", media_type="image/jpeg",
                        headers={"Cache-Control": "private, max-age=3600"})


class Response(BaseModel):
    rater: str | None = None
    item_id: str
    answer: str
    referent: str = ""
    rt_ms: int | None = None
    client_ts: str | None = None


@app.post("/api/response")
def respond(body: Response, request: Request):
    r = rater_id(request, body.rater)
    if body.item_id not in ITEMS:
        raise HTTPException(404, "unknown item")
    if body.answer not in ANSWERS:
        raise HTTPException(400, "bad answer")
    rec = {"rater": r, "item_id": body.item_id, "answer": body.answer,
           "score": ANSWERS[body.answer], "referent": body.referent.strip().lower()[:60],
           "rt_ms": body.rt_ms, "client_ts": body.client_ts, "server_ts": time.time()}
    d = RESP / r
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{body.item_id}.json").write_text(json.dumps(rec))
    _CACHE.setdefault(r, {})[body.item_id] = rec
    return {"ok": True}


@app.get("/api/progress")
def progress():
    return coverage(all_responses())
