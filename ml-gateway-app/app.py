import asyncio, os, logging, httpx
from typing import List, Union, Dict, Any
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

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

# ---------- /embed ----------------------------------------------------------
class EmbedReq(BaseModel):
    text: Union[str, List[str]]

@app.post("/embed/bge_m3")
async def embed(req: EmbedReq):
    texts = req.text if isinstance(req.text, list) else [req.text]

    payload = {
        "inputs": [{
            "name": "TEXT", "datatype": "BYTES", "shape": [len(texts)],
            "data": texts,                       # <-- СТРОКИ, не bytes
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
            {"name": "QUERY", "datatype": "BYTES", "shape": [1],
             "data": [req.query], "parameters": {"content_type": "str"}},
            {"name": "DOC",   "datatype": "BYTES", "shape": [1],
             "data": [req.doc],   "parameters": {"content_type": "str"}}
        ],
        "outputs": [{"name": "SCORE"}]
    }
    out = (await triton_infer("bge_reranker_v2_m3", payload))["outputs"][0]["data"][0]
    return {"score": float(out)}
