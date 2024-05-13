import torch
fleece_worker = __import__("fleece-worker")

worker = fleece_worker.Worker()

worker.start_layer_forward_engine()

# example 1
# layer_names = ["llama-2-7b-chat-slice/tok_embeddings", "llama-2-7b-chat-slice/layers.0", "llama-2-7b-chat-slice/layers.1", "llama-2-7b-chat-slice/layers.2", "llama-2-7b-chat-slice/layers.3", "llama-2-7b-chat-slice/layers.4", "llama-2-7b-chat-slice/layers.5", "llama-2-7b-chat-slice/layers.6", "llama-2-7b-chat-slice/layers.7", "llama-2-7b-chat-slice/layers.8", "llama-2-7b-chat-slice/layers.9", "llama-2-7b-chat-slice/layers.10", "llama-2-7b-chat-slice/layers.11", "llama-2-7b-chat-slice/layers.12", "llama-2-7b-chat-slice/layers.13", "llama-2-7b-chat-slice/layers.14", "llama-2-7b-chat-slice/layers.15",
#                "llama-2-7b-chat-slice/layers.16", "llama-2-7b-chat-slice/layers.17", "llama-2-7b-chat-slice/layers.18", "llama-2-7b-chat-slice/layers.19", "llama-2-7b-chat-slice/layers.20", "llama-2-7b-chat-slice/layers.21", "llama-2-7b-chat-slice/layers.22", "llama-2-7b-chat-slice/layers.23", "llama-2-7b-chat-slice/layers.24", "llama-2-7b-chat-slice/layers.25", "llama-2-7b-chat-slice/layers.26", "llama-2-7b-chat-slice/layers.27", "llama-2-7b-chat-slice/layers.28", "llama-2-7b-chat-slice/layers.29", "llama-2-7b-chat-slice/layers.30", "llama-2-7b-chat-slice/layers.31", "llama-2-7b-chat-slice/norm", "llama-2-7b-chat-slice/output"]
# layer_names = [
#     "llama-2-70b-chat-slice/tok_embeddings",
#     *[f"llama-2-70b-chat-slice/layers.{i}" for i in range(40)],
# ]
layer_names = [
    "llama-3-70b-instruct-slice/tok_embeddings",
    *[f"llama-3-70b-instruct-slice/layers.{i}" for i in range(10)],
]
worker.preload_layers(layer_names)
# h = torch.tensor([[1, 518, 25580, 29962, 825, 338, 278,  9522, 412, 310, 1122, 11586, 895, 29973, 518, 29914, 25580, 29962]], device="cuda")
for _bsz in [1, 2, 4, 8, 16, 32, 64, 128, 256]:
    h = torch.randint(0, 32000, (_bsz, 18), dtype=torch.long, device="cuda")
    start_pos = 0
    is_new_task = start_pos == 0
    kv_cache_dict = dict()
    for i in range(16):
        bsz = h.shape[0]
        seqlen = h.shape[1]
        _, kv_cache_dict = worker.layers_forward(h, layer_names, bsz, is_new_task, 0, start_pos, seqlen, kv_cache_dict)
        is_new_task = False
        start_pos += seqlen
        # h = torch.tensor([[29962]], device="cuda")
        h = torch.randint(0, 32000, (_bsz, 1), dtype=torch.long, device="cuda")
    print(_bsz, sum(worker.perf_bench[6:])/len(worker.perf_bench[6:]), worker.perf_bench)
    worker.perf_bench = []
