# ml-gateway-app/app.py
import os, json, asyncio, httpx, numpy as np
from fastapi import FastAPI
from pydantic import BaseModel, Field

TRITON = os.getenv("TRITON_HTTP", "http://triton:8081")
cli    = httpx.AsyncClient(timeout=30)

app = FastAPI(title="Mini Triton Gateway")

# -----------------------------------------------------------------------------
# стартап: ждём пока Triton ответит 200 на /v2/health/ready
# -----------------------------------------------------------------------------
@app.on_event("startup")
async def wait_triton_ready() -> None:
    for _ in range(60):                      # ~6 сек.
        try:
            r = await cli.get(f"{TRITON}/v2/health/ready")
            if r.status_code == 200:
                return                       # Triton готов
        except httpx.RequestError:
            pass
        await asyncio.sleep(0.1)
    raise RuntimeError("Triton didn’t become ready")
# -----------------------------------------------------------------------------

# ---------------------- схемы -------------------------------------------------
class EmbedReq(BaseModel):
    text: str = Field(..., examples=["Salom dunyo"])

class RerankReq(BaseModel):
    query: str
    doc:   str
# -----------------------------------------------------------------------------


# ---------------------- общий вызов Triton -----------------------------------
async def _rpc(model: str, inputs: list, outputs: list):
    """Отправляем /v2/models/{model}/infer и возвращаем outputs"""
    r = await cli.post(
        f"{TRITON}/v2/models/{model}/infer",
        json={"inputs": inputs, "outputs": outputs},
    )
    r.raise_for_status()                     # 4xx / 5xx → исключение
    return r.json()["outputs"]
# -----------------------------------------------------------------------------


# ---------------------- /embed ------------------------------------------------
@app.post("/embed/{model}")
async def embed(model: str, body: EmbedReq):
    inputs = [{
        "name": "TEXT", "datatype": "BYTES", "shape": [1],
        "data": [body.text], "parameters": {"content_type": "str"}
    }]
    outputs = [{"name": n} for n in
               ("DENSE", "SPARSE_VALUES", "SPARSE_INDICES")]

    o = await _rpc(model, inputs, outputs)

    return {
        "dense":           np.asarray(o[0]["data"], dtype=np.float32).tolist(),
        "sparse_values":   np.asarray(o[1]["data"], dtype=np.float32).tolist(),
        "sparse_indices":  np.asarray(o[2]["data"], dtype=np.int32 ).tolist(),
    }
# -----------------------------------------------------------------------------


# ---------------------- /rerank ----------------------------------------------
@app.post("/rerank/{model}")
async def rerank(model: str, body: RerankReq):
    inputs = [
        {"name": "QUERY", "datatype": "BYTES", "shape": [1],
         "data": [body.query], "parameters": {"content_type": "str"}},
        {"name": "DOC",   "datatype": "BYTES", "shape": [1],
         "data": [body.doc],   "parameters": {"content_type": "str"}}
    ]
    score = (await _rpc(model, inputs, [{"name": "SCORE"}]))[0]["data"][0]
    return {"score": float(score)}
# -----------------------------------------------------------------------------
