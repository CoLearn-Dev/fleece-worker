from typing import List, Tuple, Optional
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
import uvicorn
from .worker import Worker
import argparse
import requests
import json

app = FastAPI()
worker = Worker()


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
    plan: List[Tuple[str, List[str]]]
    step: int
    round: int = -1
    payload: Optional[List] = None
    max_total_len: int = 1024
    temperature: float = 0.0
    top_p: float = 0.9


@app.post("/forward")
def forward(
    req: ForwardRequest,
    background_tasks: BackgroundTasks
):
    try:
        background_tasks.add_task(worker.forward, req.task_id, req.plan, req.step, req.round, req.payload, req.max_total_len, req.temperature, req.top_p)
        return None
    except Exception as e:
        print(e)
        raise HTTPException(status_code=500, detail="Internal Server Error")


class GetInfoRequest(BaseModel):
    node_list: List[str] = []
    timeout: int = 30


class GetInfoResponse(BaseModel):
    worker_nickname: str
    gpu_mem_info: Tuple[int, int] = [0, 0]
    latency_list: List[Optional[float]] = []


@app.post("/get_info")
def get_info(
    req: GetInfoRequest,
    response_model=GetInfoResponse
):
    try:
        worker_nickname, gpu_mem_info, latency_list = worker.get_info(req.node_list, req.timeout)
        return GetInfoResponse(worker_nickname=worker_nickname, gpu_mem_info=gpu_mem_info, latency_list=latency_list)
    except Exception as e:
        print(e)
        raise HTTPException(status_code=500, detail="Internal Server Error")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--controller-url")
    parser.add_argument("-w", "--worker-url")
    parser.add_argument("-t", "--api-token")
    parser.add_argument("--worker-nickname")
    parser.add_argument("--heartbeat-interval")
    args = parser.parse_args()
    if args.worker_url is not None:
        worker.worker_url = args.worker_url
        parsed = worker.worker_url.split(':')
        if len(parsed) >= 3:
            port = int(parsed[2])
        else:
            port = 8080
    else:
        worker.worker_url = "http://127.0.0.1:8080"
        port = 8080
    if args.api_token is not None:
        worker.api_token = args.api_token
    if args.worker_nickname is not None:
        worker.worker_nickname = args.worker_nickname
    if args.heartbeat_interval is not None:
        worker.heartbeat_interval = int(args.heartbeat_interval)
    if args.controller_url is not None:
        worker.controller_url = args.controller_url
        data = {
            "url": worker.worker_url,
        }
        if worker.worker_nickname is not None:
            data["nickname"] = worker.worker_nickname
        r = requests.post(f"{args.controller_url}/register_worker",
                          json=data,
                          headers={"api-token": worker.api_token})
        worker.start_heartbeat_daemon()
    uvicorn.run(app, host="0.0.0.0", port=port, access_log=True)
    if args.controller_url is not None:
        r = requests.post(f"{args.controller_url}/deregister_worker",
                          json={
                              "url": worker.worker_url,
                          },
                          headers={"api-token": worker.api_token})
