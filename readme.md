## Installation

```
pip install -e .
```

## Try it out

```
python -m fleece-worker
```

```
curl localhost:8080/forward -H 'Content-Type: application/json' -d '{"task_id":"123","is_new_task":true,"step":0,"round":0,"plan":[["http://127.0.0.1:8080",["llama-2-7b-chat-slice/tok_embeddings", "llama-2-7b-chat-slice/layers.0", "llama-2-7b-chat-slice/layers.1", "llama-2-7b-chat-slice/layers.2", "llama-2-7b-chat-slice/layers.3", "llama-2-7b-chat-slice/layers.4", "llama-2-7b-chat-slice/layers.5", "llama-2-7b-chat-slice/layers.6", "llama-2-7b-chat-slice/layers.7", "llama-2-7b-chat-slice/layers.8", "llama-2-7b-chat-slice/layers.9", "llama-2-7b-chat-slice/layers.10", "llama-2-7b-chat-slice/layers.11", "llama-2-7b-chat-slice/layers.12", "llama-2-7b-chat-slice/layers.13", "llama-2-7b-chat-slice/layers.14", "llama-2-7b-chat-slice/layers.15", "llama-2-7b-chat-slice/layers.16", "llama-2-7b-chat-slice/layers.17", "llama-2-7b-chat-slice/layers.18", "llama-2-7b-chat-slice/layers.19", "llama-2-7b-chat-slice/layers.20", "llama-2-7b-chat-slice/layers.21", "llama-2-7b-chat-slice/layers.22", "llama-2-7b-chat-slice/layers.23", "llama-2-7b-chat-slice/layers.24", "llama-2-7b-chat-slice/layers.25", "llama-2-7b-chat-slice/layers.26", "llama-2-7b-chat-slice/layers.27", "llama-2-7b-chat-slice/layers.28", "llama-2-7b-chat-slice/layers.29", "llama-2-7b-chat-slice/layers.30", "llama-2-7b-chat-slice/layers.31", "llama-2-7b-chat-slice/norm", "llama-2-7b-chat-slice/output"]]],"payload":[[1, 518, 25580, 29962, 825, 338, 278,  9522, 412, 310, 1122, 11586, 895, 29973, 518, 29914, 25580, 29962]]}'
```

> note that the model will be automatically downloaded to `~/.cache`

## Connect to a controller

```
python -m fleece-worker -c <controller_url> -w <worker_url>
```

For example (with port forwarding `ssh -R 8080:localhost:8080 <username>@34.219.82.248`):

```
python -m fleece-worker -c http://34.219.82.248:8000 -w http://127.0.0.1:8080
```
