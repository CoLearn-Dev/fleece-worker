from typing import List, Tuple, Optional
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
import uvicorn
import sys
from .worker import Worker
import argparse
import requests
import json

app = FastAPI()
worker = Worker("http://127.0.0.1:8080")


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
    step: int
    round: int = -1
    payload: List = None


@app.post("/forward")
def forward(
    req: ForwardRequest,
    background_tasks: BackgroundTasks
):
    try:
        background_tasks.add_task(worker.forward, req.task_id, req.is_new_task, req.plan, req.step, req.round, req.payload)
        return None
    except Exception as e:
        print(e)
        raise HTTPException(status_code=500, detail="Internal Server Error")


class GetInfoRequest(BaseModel):
    node_list: List[str] = []
    timeout: int = 30


class GetInfoResponse(BaseModel):
    gpu_mem_info: Tuple[int, int] = [0, 0]
    latency_list: List[Optional[float]] = []


@app.post("/get_info")
def get_info(
    req: GetInfoRequest,
    response_model=GetInfoResponse
):
    try:
        gpu_mem_info, latency_list = worker.get_info(req.node_list, req.timeout)
        return GetInfoResponse(gpu_mem_info=gpu_mem_info, latency_list=latency_list)
    except Exception as e:
        print(e)
        raise HTTPException(status_code=500, detail="Internal Server Error")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--controller-url")
    parser.add_argument("-w", "--worker-url")
    args = parser.parse_args()
    if args.worker_url is not None:
        worker.worker_url = args.worker_url
        parsed = worker.worker_url.split(':')
        if len(parsed) >= 3:
            port = int(parsed[2])
        else:
            port = 8080
    else:
        port = 8080
    if args.controller_url is not None:
        worker.controller_url = args.controller_url
        r = requests.post(f"{args.controller_url}/register_worker",
                          json={
                              "worker_url": worker.worker_url,
                          })
        worker.worker_token = json.loads(r.content)["access_token"]
    uvicorn.run(app, host="0.0.0.0", port=port)
    if args.controller_url is not None:
        r = requests.post(f"{args.controller_url}/deregister_worker",
                          headers={"worker-token": worker.worker_token})
