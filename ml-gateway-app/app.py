import asyncio, os, logging, httpx
from typing import List, Dict, Any
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

TRITON = os.getenv("TRITON_HTTP", "http://triton:8081")
READY_EP = f"{TRITON}/v2/health/ready"

log = logging.getLogger("ml-gateway")
logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")

# ---------- helpers ----------------------------------------------------------

async def wait_triton_ready(timeout: int = 180) -> None:
    """Ждём, пока Triton вернёт 200 на /health/ready."""
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

async def triton_infer(model: str, inputs: Dict[str, Any]) -> Dict[str, Any]:
    url = f"{TRITON}/v2/models/{model}/infer"
    async with httpx.AsyncClient(timeout=20.0) as cli:
        r = await cli.post(url, json=inputs)
    if r.status_code != 200:
        raise HTTPException(502, f"Triton error {r.status_code}: {r.text}")
    return r.json()

# ---------- FastAPI ----------------------------------------------------------

app = FastAPI(lifespan=lambda app: wait_triton_ready())  # dunder-hack: lifespan короче

# -------- /embed -------------------------------------------------------------

class EmbedRequest(BaseModel):
    texts: List[str]

@app.post("/embed/bge_m3")
async def embed(req: EmbedRequest):
    inp = {
        "inputs": [{"name": "TEXT", "datatype": "BYTES", "shape": [len(req.texts)],
                    "data": [t.encode() for t in req.texts], "parameters": {"binary_data": False}}],
        "outputs": [
            {"name": "DENSE"}, {"name": "SPARSE_VALUES"}, {"name": "SPARSE_INDICES"}]
    }
    res = await triton_infer("bge_m3", inp)
    return res["outputs"]

# -------- /rerank ------------------------------------------------------------

class RerankRequest(BaseModel):
    query: str
    doc: str

@app.post("/rerank/bge_reranker_v2_m3")
async def rerank(req: RerankRequest):
    inp = {
        "inputs": [
            {"name": "QUERY", "datatype": "BYTES", "shape": [1],
             "data": [req.query.encode()], "parameters": {"binary_data": False}},
            {"name": "DOC", "datatype": "BYTES", "shape": [1],
             "data": [req.doc.encode()], "parameters": {"binary_data": False}},
        ],
        "outputs": [{"name": "SCORE"}],
    }
    res = await triton_infer("bge_reranker_v2_m3", inp)
    return res["outputs"][0]["data"][0]

# -------- /translate ---------------------------------------------------------

class TranslateRequest(BaseModel):
    text: str
    src_lang: str = "eng"
    tgt_lang: str = "rus"

@app.post("/translate/nllb_200")
async def translate(req: TranslateRequest):
    inp = {
        "inputs": [
            {"name": "SOURCE", "datatype": "BYTES", "shape": [1],
             "data": [req.text.encode()], "parameters": {"binary_data": False}},
            {"name": "SRC_LANG", "datatype": "BYTES", "shape": [1],
             "data": [req.src_lang.encode()], "parameters": {"binary_data": False}},
            {"name": "TGT_LANG", "datatype": "BYTES", "shape": [1],
             "data": [req.tgt_lang.encode()], "parameters": {"binary_data": False}},
        ],
        "outputs": [{"name": "TARGET"}],
    }
    res = await triton_infer("nllb_200_translate", inp)
    return res["outputs"][0]["data"][0].decode()
