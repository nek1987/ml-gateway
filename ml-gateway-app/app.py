import asyncio, os, json, numpy as np, httpx
from fastapi import FastAPI
from pydantic import BaseModel, Field

TRITON = os.getenv("TRITON_HTTP", "http://triton:8081")
cli = httpx.AsyncClient(timeout=30)

# ---------- ждём пока Triton сообщит, что он готов ----------
async def _wait_ready():
    for _ in range(40):                       # ≈ 4 сек.
        try:
            r = await cli.get(f"{TRITON}/v2/health/ready", timeout=0.2)
            if r.status_code == 200:
                return
        except httpx.RequestError:
            pass
        await asyncio.sleep(0.1)

asyncio.run(_wait_ready())
# -----------------------------------------------------------

app = FastAPI(title="Mini Triton Gateway")

# ------------------- Pydantic схемы ------------------------
class EmbedReq(BaseModel):
    text: str = Field(..., examples=["Salom dunyo"])

class RerankReq(BaseModel):
    query: str
    doc: str
# -----------------------------------------------------------


def _rpc(model: str, inputs: list, outputs: list):
    return cli.post(f"{TRITON}/v2/models/{model}/infer",
                    json={"inputs": inputs, "outputs": outputs})


# ---------------------- EMBED ------------------------------
@app.post("/embed/{model}")
async def embed(model: str, body: EmbedReq):
    payload_in = [{
        "name": "TEXT",
        "datatype": "BYTES",
        "shape": [1],
        "data": [body.text],
        "parameters": {"content_type": "str"}
    }]
    outs = [{"name": "DENSE"},
            {"name": "SPARSE_VALUES"},
            {"name": "SPARSE_INDICES"}]

    r = await _rpc(model, payload_in, outs)
    r.raise_for_status()

    out = r.json()["outputs"]
    dense  = np.asarray(out[0]["data"], dtype=np.float32).tolist()
    s_vals = np.asarray(out[1]["data"], dtype=np.float32).tolist()
    s_idx  = np.asarray(out[2]["data"], dtype=np.int32 ).tolist()
    return {"dense": dense, "sparse_values": s_vals, "sparse_indices": s_idx}
# -----------------------------------------------------------


# ---------------------- RERANK -----------------------------
@app.post("/rerank/{model}")
async def rerank(model: str, body: RerankReq):
    payload_in = [
        {"name": "QUERY", "datatype": "BYTES", "shape": [1],
         "data": [body.query], "parameters": {"content_type": "str"}},
        {"name": "DOC",   "datatype": "BYTES", "shape": [1],
         "data": [body.doc],   "parameters": {"content_type": "str"}}
    ]
    r = await _rpc(model, payload_in,
                   [{"name": "SCORE"}])
    r.raise_for_status()
    score = float(r.json()["outputs"][0]["data"][0])
    return {"score": score}
# -----------------------------------------------------------
