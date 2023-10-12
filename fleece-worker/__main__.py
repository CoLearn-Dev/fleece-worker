from typing import List, Tuple
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
from .worker import Worker

app = FastAPI()
worker = Worker(cache_dir="/home/ubuntu/llama")  # TODO


class LayersRequest(BaseModel):
    layer_names: List[str]


@app.post("/preload_layers")
async def preload_layers(
    req: LayersRequest
):
    try:
        await worker.preload_laters(req.layer_names)
        return None
    except Exception as e:
        print(e)
        raise HTTPException(status_code=500, detail="Internal Server Error")


@app.post("/unload_layers")
async def unload_layers(
    req: LayersRequest
):
    try:
        await worker.unload_layers(req.layer_names)
        return None
    except Exception as e:
        print(e)
        raise HTTPException(status_code=500, detail="Internal Server Error")


class ForwardRequest(BaseModel):
    task_id: str
    is_new_task: bool
    plan: List[Tuple[str, str]]
    payload: List[float]


@app.post("/forward")
async def forward(
    req: ForwardRequest
):
    try:
        await worker.forward(req)
        return None
    except Exception as e:
        print(e)
        raise HTTPException(status_code=500, detail="Internal Server Error")


class GetInfoRequest(BaseModel):
    node_list: List[str] = None
    timeout: int = 30


@app.post("/get_info")
async def get_info(
    req: GetInfoRequest
):
    try:
        await worker.get_info(req)
        return None
    except Exception as e:
        print(e)
        raise HTTPException(status_code=500, detail="Internal Server Error")

if __name__ == '__main__':
    uvicorn.run(app, host="0.0.0.0", port=8080)
