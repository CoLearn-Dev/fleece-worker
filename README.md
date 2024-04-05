## Installation

### Install From PyPI
```
pip install fleece-worker
```

### Install From Source
```
pip install -e .
```

## Connect to a controller

```
python -m fleece-worker -c <controller_url>  -t <api_token>
```
Optional: `--worker-nickname abc`, `--heartbeat-interval 10`, `-w <worker_url>`

For example:

```
python -m fleece-worker -c https://serving-api.colearn.cloud:8443 -t <api_token>
```

## Try it out (deprecated)

```
python -m fleece-worker
```

```
curl localhost:8080/forward -H 'Content-Type: application/json' -d '{"task_id":"123","step":0,"round":0,"plan":[["http://127.0.0.1:8080",["llama-2-7b-chat-slice/tok_embeddings", "llama-2-7b-chat-slice/layers.0", "llama-2-7b-chat-slice/layers.1", "llama-2-7b-chat-slice/layers.2", "llama-2-7b-chat-slice/layers.3", "llama-2-7b-chat-slice/layers.4", "llama-2-7b-chat-slice/layers.5", "llama-2-7b-chat-slice/layers.6", "llama-2-7b-chat-slice/layers.7", "llama-2-7b-chat-slice/layers.8", "llama-2-7b-chat-slice/layers.9", "llama-2-7b-chat-slice/layers.10", "llama-2-7b-chat-slice/layers.11", "llama-2-7b-chat-slice/layers.12", "llama-2-7b-chat-slice/layers.13", "llama-2-7b-chat-slice/layers.14", "llama-2-7b-chat-slice/layers.15", "llama-2-7b-chat-slice/layers.16", "llama-2-7b-chat-slice/layers.17", "llama-2-7b-chat-slice/layers.18", "llama-2-7b-chat-slice/layers.19", "llama-2-7b-chat-slice/layers.20", "llama-2-7b-chat-slice/layers.21", "llama-2-7b-chat-slice/layers.22", "llama-2-7b-chat-slice/layers.23", "llama-2-7b-chat-slice/layers.24", "llama-2-7b-chat-slice/layers.25", "llama-2-7b-chat-slice/layers.26", "llama-2-7b-chat-slice/layers.27", "llama-2-7b-chat-slice/layers.28", "llama-2-7b-chat-slice/layers.29", "llama-2-7b-chat-slice/layers.30", "llama-2-7b-chat-slice/layers.31", "llama-2-7b-chat-slice/norm", "llama-2-7b-chat-slice/output"]]],"payload":[[1, 518, 25580, 29962, 825, 338, 278,  9522, 412, 310, 1122, 11586, 895, 29973, 518, 29914, 25580, 29962]]}'
```
```
curl localhost:8080/forward -H 'Content-Type: application/json' -d '{"task_id":"123","step":0,"round":0,"plan":[["http://127.0.0.1:8080",["llama-2-7b-chat-slice/tok_embeddings", "llama-2-7b-chat-slice/layers.0", "llama-2-7b-chat-slice/layers.1", "llama-2-7b-chat-slice/layers.2", "llama-2-7b-chat-slice/layers.3", "llama-2-7b-chat-slice/layers.4", "llama-2-7b-chat-slice/layers.5", "llama-2-7b-chat-slice/layers.6", "llama-2-7b-chat-slice/layers.7", "llama-2-7b-chat-slice/layers.8", "llama-2-7b-chat-slice/layers.9", "llama-2-7b-chat-slice/layers.10", "llama-2-7b-chat-slice/layers.11", "llama-2-7b-chat-slice/layers.12", "llama-2-7b-chat-slice/layers.13", "llama-2-7b-chat-slice/layers.14", "llama-2-7b-chat-slice/layers.15", "llama-2-7b-chat-slice/layers.16", "llama-2-7b-chat-slice/layers.17", "llama-2-7b-chat-slice/layers.18", "llama-2-7b-chat-slice/layers.19", "llama-2-7b-chat-slice/layers.20", "llama-2-7b-chat-slice/layers.21", "llama-2-7b-chat-slice/layers.22", "llama-2-7b-chat-slice/layers.23", "llama-2-7b-chat-slice/layers.24", "llama-2-7b-chat-slice/layers.25", "llama-2-7b-chat-slice/layers.26", "llama-2-7b-chat-slice/layers.27", "llama-2-7b-chat-slice/layers.28", "llama-2-7b-chat-slice/layers.29", "llama-2-7b-chat-slice/layers.30", "llama-2-7b-chat-slice/layers.31", "llama-2-7b-chat-slice/norm", "llama-2-7b-chat-slice/output"]]],"payload":[[1, 518, 25580, 29962, 825, 338, 278, 9522, 412, 310, 1122, 11586, 895, 29973, 518, 29914, 25580, 29962], [1, 518, 25580, 29962, 3532, 14816, 29903, 6778, 13, 2499, 1994, 1234, 411, 5952, 18282, 13, 29966, 829, 14816, 29903, 6778, 13, 13, 29902, 626, 2675, 304, 3681, 29892, 825, 881, 306, 1074, 29973, 518, 29914, 25580, 29962], [1, 518, 25580, 29962, 3532, 14816, 29903, 6778, 13, 2499, 1994, 1234, 411, 953, 3848, 275, 13, 29966, 829, 14816, 29903, 6778, 13, 13, 5328, 304, 748, 515, 1522, 823, 292, 304, 23526, 29973, 518, 29914, 25580, 29962]]}'
```
> note that the model will be automatically downloaded to `~/.cache`
