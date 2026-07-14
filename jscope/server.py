"""J-Scope server: one process, one model in memory, a handful of forward passes.

    .venv/bin/python jscope/server.py            # real model, device auto-picked
    .venv/bin/python jscope/server.py --mock     # no torch, no GPU, fake data

Then open http://127.0.0.1:7801.

Every endpoint is a small number of forward passes over a short prompt, so this
is an interactive tool, not a batch job. Results are cached by request body,
which also gives us the freeze: the cache *is* the session.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import re
import sys
import threading

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

sys.path.insert(0, str(pathlib.Path(__file__).parent))

from engine import MockEngine, RealEngine, band_layers  # noqa: E402

HERE = pathlib.Path(__file__).parent
STATIC = HERE / "static"
FROZEN = HERE / "frozen"

app = FastAPI(title="J-Scope")
ENGINE = None
LOCK = threading.Lock()          # one model, one forward pass at a time
CACHE: dict[str, dict] = {}      # request body -> response, and the freeze source


def key_of(path: str, body: dict) -> str:
    return path + "|" + json.dumps(body, sort_keys=True, separators=(",", ":"))


def cached(path: str, body: dict, fn):
    k = key_of(path, body)
    if k not in CACHE:
        with LOCK:
            try:
                CACHE[k] = fn()
            except ValueError as e:
                raise HTTPException(status_code=400, detail=str(e))
    return CACHE[k]


class ReadoutReq(BaseModel):
    prompt: str
    concepts: list[str] = []
    layers: list[int] | None = None
    use_jacobian: bool = True


class InjectReq(BaseModel):
    prompt: str
    concept: str
    layers: list[int]
    positions: list[int]
    alpha: float
    mode: str = "strength"
    watch: list[str] = []          # what a correct answer looks like, if any


class SweepReq(BaseModel):
    prompt: str
    concept: str
    layers: list[int]
    positions: list[int]
    alphas: list[float]
    mode: str = "strength"
    watch: list[str] = []


class NaturalReq(BaseModel):
    prompt: str
    concept: str
    layers: list[int]


class FreezeReq(BaseModel):
    name: str
    entries: dict[str, dict]


@app.get("/api/meta")
def meta():
    return ENGINE.meta()


@app.post("/api/tokenize")
def tokenize(req: ReadoutReq):
    return cached("/api/tokenize", {"prompt": req.prompt},
                  lambda: {"tokens": ENGINE.tokenize(req.prompt)})


@app.post("/api/readout")
def readout(req: ReadoutReq):
    body = req.model_dump()
    return cached("/api/readout", body, lambda: ENGINE.readout(
        req.prompt, req.concepts, req.layers, req.use_jacobian))


@app.post("/api/inject")
def inject(req: InjectReq):
    body = req.model_dump()
    return cached("/api/inject", body, lambda: ENGINE.inject(
        req.prompt, req.concept, req.layers, req.positions, req.alpha, req.mode,
        watch=req.watch))


@app.post("/api/sweep")
def sweep(req: SweepReq):
    body = req.model_dump()
    return cached("/api/sweep", body, lambda: {"points": ENGINE.sweep(
        req.prompt, req.concept, req.layers, req.positions, req.alphas, req.mode,
        watch=req.watch)})


@app.post("/api/natural")
def natural(req: NaturalReq):
    body = req.model_dump()
    return cached("/api/natural", body, lambda: {
        "natural": ENGINE.natural_alpha(req.prompt, req.concept, req.layers)})


@app.post("/api/freeze")
def freeze(req: FreezeReq):
    """Write a standalone copy of the UI with this session's answers baked in.

    The frozen page is the same index.html; it just finds a pre-filled cache on
    `window.__JSCOPE_FROZEN__` and never calls the network. That is what makes it
    survive without a GPU.
    """
    safe = re.sub(r"[^a-z0-9_-]", "-", req.name.strip().lower()) or "session"
    FROZEN.mkdir(exist_ok=True)
    html = (STATIC / "index.html").read_text()
    blob = json.dumps(req.entries, separators=(",", ":"))
    inject_script = (
        "<script>window.__JSCOPE_FROZEN__ = "
        + blob.replace("</", "<\\/")
        + ";</script>\n"
    )
    if "</head>" not in html:
        raise HTTPException(500, "index.html has no </head> to graft onto")
    out = FROZEN / f"{safe}.html"
    out.write_text(html.replace("</head>", inject_script + "</head>", 1))
    return {"path": str(out), "entries": len(req.entries), "bytes": out.stat().st_size}


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")


def main():
    global ENGINE
    ap = argparse.ArgumentParser()
    ap.add_argument("--mock", action="store_true", help="no torch, no model, fake data")
    ap.add_argument("--model", default=None)
    ap.add_argument("--device", default=None, help="cuda / mps / cpu (auto if unset)")
    ap.add_argument("--port", type=int, default=7801)
    ap.add_argument("--host", default="127.0.0.1")
    args = ap.parse_args()

    if args.mock:
        print("J-Scope: MOCK engine. Numbers are fiction.")
        ENGINE = MockEngine()
    else:
        from engine import MODEL_NAME
        print(f"J-Scope: loading {args.model or MODEL_NAME} ...")
        ENGINE = RealEngine(model_name=args.model or MODEL_NAME, device=args.device)
        print(f"J-Scope: ready on {ENGINE.device} ({ENGINE.dtype})")

    app.mount("/static", StaticFiles(directory=STATIC), name="static")

    import uvicorn
    print(f"J-Scope: http://{args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
