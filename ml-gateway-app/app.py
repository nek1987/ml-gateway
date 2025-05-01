import asyncio, os, httpx, json
# import os, httpx, triton_python_backend_utils as pb # ← УБРАТЬ
asyncio.run(_wait_ready()) 
from fastapi import FastAPI
from pydantic import BaseModel

TRITON = os.getenv("TRITON_HTTP", "http://triton:8081")
async def _wait_ready():
    async with httpx.AsyncClient() as cli:
        for _ in range(20):                            # ≤ 2 с
            try:
                r = await cli.get(f"{TRITON}/v2/health/ready", timeout=0.2)
                if r.status_code == 200:
                    return
            except httpx.RequestError:
                pass
            await asyncio.sleep(0.1)

         

app = FastAPI(title="ML-Gateway")

async def _triton(model: str, body: dict):
    async with httpx.AsyncClient(timeout=120) as c:
        r = await c.post(f"{TRITON}/v2/models/{model}/infer", json=body)
        r.raise_for_status()
        return r.json()

# ---------- Embedding ----------
class TextReq(BaseModel):
    text: str
    sparse: bool = False

@app.post("/embed/m3")
async def embed(r: TextReq):
    body = {
      "inputs":[{"name":"TEXT","datatype":"BYTES","shape":[1],"data":[r.text]}],
      "outputs":[
        {"name":"DENSE"},
        {"name":"SPARSE_VALUES"},
        {"name":"SPARSE_INDICES"}
      ]
    }
    j = await _triton("bge_m3", body)
    dense = j["outputs"][0]["data"][0]
    if not r.sparse:
        return {"dense": dense}
    return {
      "dense": dense,
      "sparse":{
        "values": j["outputs"][1]["data"],
        "indices":j["outputs"][2]["data"]
      }
    }

# ---------- Rerank ----------
class RerankReq(BaseModel):
    query: str
    docs: list[str]

@app.post("/rerank/m3")
async def rerank(r: RerankReq):
    body = {
      "inputs":[
        {"name":"QUERY","datatype":"BYTES","shape":[1],"data":[r.query]},
        {"name":"DOC","datatype":"BYTES","shape":[len(r.docs)],"data":r.docs}
      ],
      "outputs":[{"name":"SCORE"}]
    }
    j = await _triton("bge_reranker_v2_m3", body)
    return {"score": j["outputs"][0]["data"]}

# ---------- Translate ----------
class TReq(BaseModel):
    text: str
    src: str
    tgt: str

@app.post("/translate")
async def translate(r: TReq):
    body = {
      "inputs":[
        {"name":"TEXT","datatype":"BYTES","shape":[1],"data":[r.text]},
        {"name":"SRC_LANG","datatype":"BYTES","shape":[1],"data":[r.src]},
        {"name":"TGT_LANG","datatype":"BYTES","shape":[1],"data":[r.tgt]}
      ],
      "outputs":[{"name":"TRANSLATION"}]
    }
    j = await _triton("nllb_200_translate", body)
    return {"translation": j["outputs"][0]["data"][0]}
