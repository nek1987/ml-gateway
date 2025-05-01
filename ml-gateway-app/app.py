import os, json, numpy as np, httpx, asyncio
from fastapi import FastAPI
from pydantic import BaseModel, Field

TRITON = os.getenv("TRITON_HTTP", "http://triton:8081")
cli = httpx.AsyncClient(timeout=30)

# ---------- FastAPI -------------------------------------------------
app = FastAPI(title="Mini Triton Gateway")

# ▸ ждём готовности Triton один раз при старте контейнера
@app.on_event("startup")
async def wait_triton_ready() -> None:
    for _ in range(40):          # ≈ 4 с
        try:
            r = await cli.get(f"{TRITON}/v2/health/ready")
            if r.status_code == 200:
                return
        except httpx.RequestError:
            pass
        await asyncio.sleep(0.1)
    raise RuntimeError("Triton didn’t become ready")

# -------------------- схемы -----------------------------------------
class EmbedReq(BaseModel):
    text: str = Field(..., examples=["Salom dunyo"])

class RerankReq(BaseModel):
    query: str
    doc:   str
# --------------------------------------------------------------------


# -------------------- helpers ---------------------------------------
async def _rpc(model: str, inputs: list, outputs: list):
    r = await cli.post(f"{TRITON}/v2/models/{model}/infer",
                       json={"inputs": inputs, "outputs": outputs})
    r.raise_for_status()
    return r.json()["outputs"]
# --------------------------------------------------------------------


# -------------------- /embed ----------------------------------------
@app.post("/embed/{model}")
async def embed(model: str, body: EmbedReq):
    ins = [{
        "name": "TEXT", "datatype": "BYTES", "shape": [1],
        "data": [body.text], "parameters": {"content_type": "str"}
    }]
    outs = [{"name": n} for n in ("DENSE", "SPARSE_VALUES", "SPARSE_INDICES")]

    o = await _rpc
