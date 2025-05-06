import asyncio, os, logging, httpx
from typing import List, Union, Dict, Any
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
import base64
import langid

TRITON = os.getenv("TRITON_HTTP", "http://triton:8081")
READY_EP = f"{TRITON}/v2/health/ready"

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("gateway")

# ---------- helper ----------------------------------------------------------
async def wait_triton_ready(timeout: int = 180) -> None:
    async with httpx.AsyncClient(timeout=5.0) as cli:
        for sec in range(timeout):
            try:
                r = await cli.get(READY_EP)
                if r.status_code == 200:
                    log.info("Triton READY (%s с)", sec)
                    return
            except httpx.RequestError:
                pass
            await asyncio.sleep(1)
    raise RuntimeError("Triton didn’t become ready")

async def triton_infer(model: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    async with httpx.AsyncClient(timeout=30.0) as cli:
        r = await cli.post(f"{TRITON}/v2/models/{model}/infer", json=payload)
    if r.status_code != 200:
        raise HTTPException(502, f"Triton {r.status_code}: {r.text}")
    return r.json()

# ---------- FastAPI ---------------------------------------------------------
app = FastAPI()

@app.on_event("startup")
async def _startup():          # дождались Triton при запуске
    await wait_triton_ready()

# === Новые Pydantic-схемы ===
class LangIDResp(BaseModel):
    lang: str
    prob: float

class TextReq(BaseModel):
    text: Union[str, List[str]]


# === 1) /langid → {lang, prob} ===
@app.get("/langid", response_model=LangIDResp)
async def detect_lang(q: str = Query(..., description="Text to detect language for")):
    """
    Определяет язык текста локально через langid.classify.
    """
    lang, prob = langid.classify(q)
    return LangIDResp(lang=lang, prob=prob)


# === 2) /translate/uzlat → перевод на uz_Latn ===
@app.post("/translate/uzlat")
async def translate_to_uzlat(req: TextReq):
    """
    Переводит текст(ы) на узбекскую латиницу через Triton-модель nllb_200_translate.
    Исходный язык определяется через langid.classify, целевой всегда uz_Latn.
    """
    # 1) нормализация на список
    texts = req.text if isinstance(req.text, list) else [req.text]
    batch_size = len(texts)

    # 2) определяем SRC_LANG для каждого фрагмента
    src_langs = []
    for t in texts:
        code, _ = langid.classify(t)
        code_map = {
            "uz": "uz_Latn",
            "ru": "ru_Cyrl",
            "en": "en_XX",
            # добавьте при необходимости
        }
        src = code_map.get(code)
        if not src:
            raise HTTPException(400, f"Unsupported source language: {code}")
        src_langs.append(src)

    # 3) готовим Triton payload
    payload = {
        "inputs": [
            {
                "name": "TEXT", "datatype": "BYTES",
                "shape": [batch_size, 1],
                "data": [[t] for t in texts],
                "parameters": {"content_type": "str"}
            },
            {
                "name": "SRC_LANG", "datatype": "BYTES",
                "shape": [batch_size, 1],
                "data": [[s] for s in src_langs],
                "parameters": {"content_type": "str"}
            },
            {
                "name": "TGT_LANG", "datatype": "BYTES",
                "shape": [batch_size, 1],
                "data": [["uz_Latn"] for _ in range(batch_size)],
                "parameters": {"content_type": "str"}
            }
        ],
        "outputs": [{"name": "TRANSLATION"}]
    }

    resp = await triton_infer("nllb_200_translate", payload)

    # 4) декодируем (Triton отдаёт base64-строки)
    raw = resp["outputs"][0]["data"]
    results = []
    for row in raw:
        token = row[0] if isinstance(row, (list, tuple)) else row
        try:
            txt = base64.b64decode(token).decode("utf-8")
        except Exception:
            txt = token.decode("utf-8", errors="ignore")
        results.append(txt)

    return results[0] if isinstance(req.text, str) else results


# ---------- /embed ----------------------------------------------------------
class EmbedReq(BaseModel):
    text: Union[str, List[str]]

@app.post("/embed/bge_m3")
async def embed(req: EmbedReq):
    texts = req.text if isinstance(req.text, list) else [req.text]
    payload = {
        "inputs": [{
            "name": "TEXT", "datatype": "BYTES",
            "shape": [len(texts), 1],              # <-- 2-D
            "data": [[t] for t in texts],          # <-- [[str]]
            "parameters": {"content_type": "str"}
        }],
        "outputs": [{"name": "DENSE"},
                    {"name": "SPARSE_VALUES"},
                    {"name": "SPARSE_INDICES"}]
    }
    return (await triton_infer("bge_m3", payload))["outputs"]

# ---------- /rerank ---------------------------------------------------------
class RerankReq(BaseModel):
    query: str
    doc: str

@app.post("/rerank/bge_reranker_v2_m3")
async def rerank(req: RerankReq):
    payload = {
        "inputs": [
            {"name": "QUERY", "datatype": "BYTES", "shape": [1,1],
         "data": [[req.query]], "parameters": {"content_type": "str"}},
        {"name": "DOC",   "datatype": "BYTES", "shape": [1,1],
         "data": [[req.doc]],   "parameters": {"content_type": "str"}}
        ],
        "outputs": [{"name": "SCORE"}]
    }
    out = (await triton_infer("bge_reranker_v2_m3", payload))["outputs"][0]["data"][0]
    return {"score": float(out)}
