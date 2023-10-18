from typing import List, Tuple
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
import uvicorn
import sys
from .worker import Worker

app = FastAPI()
worker = Worker("http://127.0.0.1:8080", cache_dir="/home/ubuntu/llama")  # TODO


class LayersRequest(BaseModel):
    layer_names: List[str]


@app.post("/preload_layers")
def preload_layers(
    req: LayersRequest
):
    try:
        worker.preload_layers(req.layer_names)
        return None
    except Exception as e:
        print(e)
        raise HTTPException(status_code=500, detail="Internal Server Error")


@app.post("/unload_layers")
def unload_layers(
    req: LayersRequest
):
    try:
        worker.unload_layers(req.layer_names)
        return None
    except Exception as e:
        print(e)
        raise HTTPException(status_code=500, detail="Internal Server Error")


class ForwardRequest(BaseModel):
    task_id: str
    is_new_task: bool
    plan: List[Tuple[str, List[str]]]
    payload: List = None


@app.post("/forward")
def forward(
    req: ForwardRequest,
    background_tasks: BackgroundTasks
):
    try:
        background_tasks.add_task(worker.forward, req.task_id, req.is_new_task, req.plan, req.payload)
        return None
    except Exception as e:
        print(e)
        raise HTTPException(status_code=500, detail="Internal Server Error")


class GetInfoRequest(BaseModel):
    node_list: List[str] = None
    timeout: int = 30


@app.post("/get_info")
def get_info(
    req: GetInfoRequest
):
    try:
        worker.get_info(req)
        return None
    except Exception as e:
        print(e)
        raise HTTPException(status_code=500, detail="Internal Server Error")


if __name__ == '__main__':
    if len(sys.argv) >= 2:
        worker.my_url = sys.argv[1]
        parsed = worker.my_url.split(':')
        if len(parsed) >= 3:
            port = int(parsed[2])
        else:
            port = 8080
        uvicorn.run(app, host="0.0.0.0", port=port)
    else:
        uvicorn.run(app, host="0.0.0.0", port=8080)
