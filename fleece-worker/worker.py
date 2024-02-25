from typing import List, Tuple, Dict, Any, Set
import os
import torch
from torch import nn
from .model import ModelArgs, TransformerBlock, RMSNorm, precompute_freqs_cis
# from .tokenizer import Tokenizer
import requests
import threading
import concurrent.futures
import time
import socket
from urllib.parse import urlparse
import json
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.primitives import hashes

torch.set_default_device("cpu")
torch.set_default_dtype(torch.float16)

llama_2_7b_args = {"dim": 4096, "multiple_of": 256, "n_heads": 32, "n_layers": 32, "norm_eps": 1e-06, "vocab_size": 32000}
llama_2_13b_args = {"dim": 5120, "multiple_of": 256, "n_heads": 40, "n_layers": 40, "norm_eps": 1e-05, "vocab_size": 32000}
llama_2_70b_args = {"dim": 8192, "multiple_of": 4096, "ffn_dim_multiplier": 1.3, "n_heads": 64, "n_kv_heads": 8, "n_layers": 80, "norm_eps": 1e-05, "vocab_size": 32000}

if torch.cuda.is_available():
    global_freqs_cis = precompute_freqs_cis(128, 4096).to("cuda")
else:
    global_freqs_cis = precompute_freqs_cis(128, 4096)
# tokenizer = Tokenizer(model_path="/home/ubuntu/llama/tokenizer.model")
# print(tokenizer.bos_id) 1
# print(tokenizer.eos_id) 2
# print(tokenizer.pad_id) -1
# print(tokenizer.n_words) 32000


def parse_layer_name(layer_name: str):
    s = layer_name.split('/')
    return s[0], s[1]


KV_CACHE_BLOCK = 512


def get_kv_cache_length(cur, seqlen):
    while cur < seqlen:
        cur += KV_CACHE_BLOCK
    return cur


def get_kv_cache(x, start_pos, kv_cache, model):
    bsz, seqlen, _ = x.shape
    if kv_cache is None:
        length = get_kv_cache_length(0, start_pos + seqlen)
        cache_k = torch.zeros(
            (
                bsz,
                length,
                model.attention.n_local_kv_heads,
                model.attention.head_dim,
            ),
            device="cuda"
        )
        cache_v = torch.zeros(
            (
                bsz,
                length,
                model.attention.n_local_kv_heads,
                model.attention.head_dim,
            ),
            device="cuda"
        )
        return (cache_k, cache_v)
    old_cache_k, old_cache_v = kv_cache
    if start_pos + seqlen > old_cache_k.shape[1]:
        length = get_kv_cache_length(old_cache_k.shape[1], start_pos + seqlen)
        cache_k = torch.zeros(
            (
                bsz,
                length,
                model.attention.n_local_kv_heads,
                model.attention.head_dim,
            ),
            device="cuda"
        )
        cache_v = torch.zeros(
            (
                bsz,
                length,
                model.attention.n_local_kv_heads,
                model.attention.head_dim,
            ),
            device="cuda"
        )
        cache_k[:, :start_pos, :, :], cache_v[:, :start_pos, :, :] = old_cache_k[:, :start_pos, :, :], old_cache_v[:, :start_pos, :, :]
        del_tensor(old_cache_k)
        del_tensor(old_cache_v)
        del kv_cache
        return (cache_k, cache_v)
    else:
        return kv_cache


def del_tensor(t):
    t.detach()
    t.grad = None
    t.untyped_storage().resize_(0)


executor = concurrent.futures.ThreadPoolExecutor(max_workers=40)
executor_forward = concurrent.futures.ThreadPoolExecutor(max_workers=40)


def requests_post(url, headers=None, json=None, worker=None):
    try:
        r = requests.post(url, headers=headers, json=json)
        assert r.status_code == 200
    except:
        if worker is not None:
            worker.cancel_task(json["task_id"])


def send_request(url, headers=None, json=None, exec=None, worker=None):
    if exec is None:
        executor.submit(requests_post, url, headers, json, worker)
    else:
        exec.submit(requests_post, url, headers, json, worker)


executor_latency_test = concurrent.futures.ThreadPoolExecutor(max_workers=40)


def latency_test(host: str, port: int, timeout=60):
    st = time.monotonic()
    try:
        s = socket.create_connection((host, port), timeout=timeout)
        s.shutdown(socket.SHUT_RD)
    except socket.timeout:
        return None
    except OSError:
        return None
    en = time.monotonic()
    return (en-st)*1000


def measure_latency(node_list: List[str], timeout):
    # executor_latency_test
    jobs = []
    for node in node_list:
        parsed_url = urlparse(node)
        host = parsed_url.hostname
        if parsed_url.port is not None:
            port = parsed_url.port
        elif parsed_url.scheme == "http":
            port = 80
        elif parsed_url.scheme == "https":
            port = 443
        else:
            port = 22
        jobs.append(executor_latency_test.submit(latency_test, host, port, timeout))
    ans = []
    for job in jobs:
        ans.append(job.result())
    print(ans)
    return ans


class Worker:
    def __init__(
            self,
            worker_id: str = None,
            # mirror_url: str = "TODO",
            cache_dir: str = "~/.cache/fleece-worker/models",
    ):
        self.worker_id = worker_id
        # self.mirror_url = mirror_url
        self.controller_url = None
        self.api_token = None
        self.worker_nickname = worker_id
        self.heartbeat_interval = 300
        self.tm_pubkeys = {}
        self.worker_urls = {}
        self.cache_dir = os.path.expanduser(cache_dir)
        self.layers = dict()
        self.task_info: Dict[(str, int), Tuple[int, Dict[str, Any]]] = dict()
        self.mutex = threading.Lock()
        self.task_prompt_tokens: Dict[str, torch.Tensor] = dict()
        self.task_eos_reached: Dict[str, torch.Tensor] = dict()
        self.task_local_steps: Dict[str, List[int]] = dict()
        self.canceled_task: Set[str] = set()

    def fetch_layer(self, full_layer_name):
        model_name, layer_name = parse_layer_name(full_layer_name)
        if model_name.startswith("dummy"):
            return None
        path = os.path.join(self.cache_dir, model_name, f"{layer_name}.pt")
        if not os.path.exists(path):  # TODO lock
            os.makedirs(os.path.join(self.cache_dir, model_name), exist_ok=True)
            with requests.get(f"https://huggingface.co/colearn/{model_name}/resolve/main/{layer_name}.pt", stream=True) as r:
                r.raise_for_status()
                with open(path, 'wb') as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        f.write(chunk)
        return path

    def preload_layers(self, layer_names: List[str]):
        with self.mutex:  # TODO
            ths = []
            for full_layer_name in layer_names:
                if full_layer_name in self.layers:
                    continue
                th = executor.submit(self.fetch_layer, full_layer_name)
                ths.append((full_layer_name, th))
            for full_layer_name, th in ths:
                path = th.result()
                model_name, layer_name = parse_layer_name(full_layer_name)
                if model_name.startswith("dummy"):
                    continue
                if model_name.startswith("llama-2-7b"):
                    model_args = ModelArgs(**llama_2_7b_args)
                elif model_name.startswith("llama-2-13b"):
                    model_args = ModelArgs(**llama_2_13b_args)
                elif model_name.startswith("llama-2-70b"):
                    model_args = ModelArgs(**llama_2_70b_args)
                else:
                    raise NotImplementedError("Unknown model")
                if layer_name == "tok_embeddings":
                    l = torch.nn.utils.skip_init(nn.Embedding, model_args.vocab_size, model_args.dim)
                elif layer_name.startswith("layer"):
                    l = TransformerBlock(model_args)
                elif layer_name == "norm":
                    l = RMSNorm(model_args.dim, eps=model_args.norm_eps)
                elif layer_name == "output":
                    l = torch.nn.utils.skip_init(nn.Linear, model_args.dim, model_args.vocab_size, bias=False)
                else:
                    raise NotImplementedError("Unknown layers")
                l.load_state_dict(torch.load(path, map_location="cpu"))
                l.to("cuda")
                self.layers[full_layer_name] = l

    def unload_layers(self, layer_names: List[str]):
        for full_layer_name in layer_names:
            if full_layer_name not in self.layers:
                continue  # TODO continue or warning?
            del self.layers[full_layer_name]
            torch.cuda.empty_cache()

    def cancel_task(self, task_id: str):
        self.del_task(task_id)
        self.canceled_task.add(task_id)

    def del_task(self, task_id: str):
        steps = self.task_local_steps.pop(task_id, None)
        if steps is None:
            return
        if task_id in self.task_prompt_tokens:
            del self.task_prompt_tokens[task_id]
        if task_id in self.task_eos_reached:
            del self.task_eos_reached[task_id]
        for step in steps:
            _, kv_cache_dict = self.task_info[(task_id, step)]
            for _, kv_cache in kv_cache_dict.items():
                k_cache, v_cache = kv_cache
                del_tensor(k_cache)
                del_tensor(v_cache)
            del self.task_info[(task_id, step)]
        torch.cuda.empty_cache()

    def pull_worker_url(self):
        r = requests.get(f"{self.controller_url}/get_worker_list",
                         headers={"api-token": self.api_token})
        res = json.loads(r.content)
        for worker in res["workers"]:
            self.worker_urls[worker["worker_id"]] = worker["url"]

    def get_worker_url(self, worker_id):
        if worker_id not in self.worker_urls:
            self.pull_worker_url()
        return self.worker_urls[worker_id]

    def verify(self, tm_url, task_id, plan, timestamp, signature_hex):
        public_key_bytes = bytes.fromhex(self.tm_pubkeys[tm_url])
        public_key = ec.EllipticCurvePublicKey.from_encoded_point(
            ec.SECP256K1(), public_key_bytes
        )
        signed_bytes = task_id.encode()+str(timestamp).encode()
        for x in plan:
            signed_bytes += x[0].encode()
            for y in x[1]:
                signed_bytes += y.encode()
        try:
            public_key.verify(
                bytes.fromhex(signature_hex),
                signed_bytes,
                ec.ECDSA(hashes.SHA256())
            )
            return True
        except:
            return False

    def forward(self,
                task_id: str,
                plan: List[Tuple[str, List[str]]],
                step: int,
                round: int,
                payload: List,
                max_total_len: int,
                temperature: float,
                top_p: float,
                task_manager_url: str,
                signature: str,
                timestamp: int,
                ):
        self.verify(task_manager_url, task_id, plan, timestamp, signature)

        index = step
        is_new_task = round == 0
        if payload is None or task_id in self.canceled_task:
            self.del_task(task_id)
            if index < len(plan)-1:
                # next node
                send_request(
                    f"{self.get_worker_url(plan[index+1][0])}/forward",
                    json={
                        "task_id": task_id,
                        "plan": plan,
                        "step": step+1,
                        "task_manager_url": task_manager_url,
                        "signature": signature,
                        "timestamp": timestamp,
                    },
                    exec=executor_forward)
            return

        if is_new_task:
            if task_id in self.task_local_steps:
                self.task_local_steps[task_id].append(step)
            else:
                self.task_local_steps[task_id] = [step]
            self.task_info[(task_id, step)] = (0, dict())
        else:
            if not task_id in self.task_local_steps:
                return
        start_pos, kv_cache_dict = self.task_info[(task_id, step)]

        # first node
        if index == 0:
            bsz = len(payload)
            if is_new_task:
                min_prompt_len = min(len(t) for t in payload)
                self.task_prompt_tokens[task_id] = payload
                tokens = torch.zeros((bsz, min_prompt_len), dtype=torch.long)
                for k, t in enumerate(payload):
                    tokens[k, :] = torch.tensor(t[:min_prompt_len], dtype=torch.long)
                if torch.cuda.is_available():
                    h = tokens.to("cuda")
                else:
                    h = tokens
            else:
                prompt_tokens = self.task_prompt_tokens[task_id]
                tokens = torch.zeros((bsz, 1), dtype=torch.long)
                for k, t in enumerate(prompt_tokens):
                    if len(t) > start_pos:
                        tokens[k, :] = torch.tensor([t[start_pos]], dtype=torch.long)
                    else:
                        tokens[k, :] = torch.tensor([payload[k]], dtype=torch.long)
                if torch.cuda.is_available():
                    h = tokens.to("cuda")
                else:
                    h = tokens
            # print(h)
            bsz, seqlen = h.shape
        else:
            h = torch.tensor(payload, dtype=torch.float16, device="cuda")
            bsz, seqlen, _ = h.shape

        # forward
        freqs_cis = global_freqs_cis[start_pos: start_pos + seqlen]
        mask = None
        if seqlen > 1:
            mask = torch.full(
                (1, 1, seqlen, seqlen), float("-inf"), device=h.device
            )
            mask = torch.triu(mask, diagonal=start_pos + 1).type_as(h)
        # layer
        _, layer_names = plan[index]
        self.preload_layers(layer_names)  # preload
        with torch.inference_mode():
            with self.mutex:
                for full_layer_name in layer_names:
                    model_name, layer_name = parse_layer_name(full_layer_name)
                    if model_name.startswith("dummy"):
                        if layer_name == "output":
                            h = torch.rand((bsz, 1, 32000), dtype=torch.float16)
                            if round >= 32:
                                h = torch.zeros((bsz, 1, 32000), dtype=torch.float16)
                                h[:, :, 2] = 1.0
                        continue
                    if layer_name == "tok_embeddings":
                        h = self.layers[full_layer_name](h)
                    elif layer_name.startswith("layers."):
                        if is_new_task:
                            kv_cache = get_kv_cache(h, start_pos, None, self.layers[full_layer_name])
                        else:
                            kv_cache = get_kv_cache(h, start_pos, kv_cache_dict[full_layer_name], self.layers[full_layer_name])
                        h = self.layers[full_layer_name](h, start_pos, freqs_cis, mask, kv_cache)
                        kv_cache_dict[full_layer_name] = kv_cache
                    elif layer_name == "norm":
                        h = self.layers[full_layer_name](h)
                    elif layer_name == "output":
                        h = self.layers[full_layer_name](h)
                    else:
                        raise NotImplementedError("Unknown layers")
        self.task_info[(task_id, step)] = (start_pos+seqlen, kv_cache_dict)

        # last node
        if index == len(plan)-1:
            if temperature > 0:
                probs = torch.softmax(h[:, -1] / temperature, dim=-1)
                next_token = sample_top_p(probs, top_p)
            else:
                next_token = torch.argmax(h[:, -1], dim=-1)
            next_token = next_token.reshape(-1)
            if start_pos > max_total_len:
                next_token = torch.tensor([2] * bsz)  # FIXME fake max length limit
            # print(next_token)
            next_token = next_token.to("cpu")

            # eos_reached
            if is_new_task:
                self.task_eos_reached[task_id] = torch.tensor([False] * bsz)
            self.task_eos_reached[task_id] |= next_token == 2  # eos_id
            if not all(self.task_eos_reached[task_id]):
                # next node
                send_request(
                    f"{self.get_worker_url(plan[0][0])}/forward",
                    json={
                        "task_id": task_id,
                        "plan": plan,
                        "step": 0,
                        "round": round+1,
                        "payload": next_token.tolist(),
                        "max_total_len": max_total_len,
                        "temperature": temperature,
                        "top_p": top_p,
                        "task_manager_url": task_manager_url,
                        "signature": signature,
                        "timestamp": timestamp,
                    },
                    exec=executor_forward,
                    worker=self)
            else:
                self.cancel_task(task_id)
                send_request(
                    f"{self.get_worker_url(plan[0][0])}/forward",
                    json={
                        "task_id": task_id,
                        "plan": plan,
                        "step": 0,
                        "task_manager_url": task_manager_url,
                        "signature": signature,
                        "timestamp": timestamp,
                    },
                    exec=executor_forward)
            # update
            if task_manager_url is not None:
                send_request(
                    f"{task_manager_url}/update_task",
                    headers={"worker-id": self.worker_id, "api-token": self.api_token},
                    json={
                        "task_id": task_id,
                        "plan_current_step": step,
                        "plan_current_round": round,
                        "output_tokens": next_token.tolist(),
                    },
                    worker=self)
        else:
            # next node
            send_request(
                f"{self.get_worker_url(plan[index+1][0])}/forward",
                json={
                    "task_id": task_id,
                    "plan": plan,
                    "step": step+1,
                    "round": round,
                    "payload": h.tolist(),
                    "max_total_len": max_total_len,
                    "temperature": temperature,
                    "top_p": top_p,
                    "task_manager_url": task_manager_url,
                    "signature": signature,
                    "timestamp": timestamp,
                },
                exec=executor_forward,
                worker=self)
            # update
            if task_manager_url is not None:
                send_request(
                    f"{task_manager_url}/update_task",
                    headers={"worker-id": self.worker_id, "api-token": self.api_token},
                    json={
                        "task_id": task_id,
                        "plan_current_step": step,
                        "plan_current_round": round,
                    },
                    worker=self)

    def get_info(self, node_list, timeout):
        gpu_mem_info = torch.cuda.mem_get_info()
        latency_list = measure_latency(node_list, timeout)
        return self.worker_nickname, gpu_mem_info, latency_list

    def send_heartbeat(self):
        data = {

        }
        r = requests.post(f"{self.controller_url}/worker_heartbeat",
                          json=data,
                          headers={"worker-id": self.worker_id, "api-token": self.api_token})
        res = json.loads(r.content)
        self.tm_pubkeys = res["pubkeys"]

    def start_heartbeat_daemon(self):
        def heartbeat_thread():
            while True:
                self.send_heartbeat()
                time.sleep(self.heartbeat_interval)
        heartbeat_thread = threading.Thread(target=heartbeat_thread)
        heartbeat_thread.daemon = True
        heartbeat_thread.start()


def sample_top_p(probs, p):
    """
    Perform top-p (nucleus) sampling on a probability distribution.

    Args:
        probs (torch.Tensor): Probability distribution tensor.
        p (float): Probability threshold for top-p sampling.

    Returns:
        torch.Tensor: Sampled token indices.

    Note:
        Top-p sampling selects the smallest set of tokens whose cumulative probability mass
        exceeds the threshold p. The distribution is renormalized based on the selected tokens.

    """
    probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
    probs_sum = torch.cumsum(probs_sort, dim=-1)
    mask = probs_sum - probs_sort > p
    probs_sort[mask] = 0.0
    probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
    next_token = torch.multinomial(probs_sort, num_samples=1)
    next_token = torch.gather(probs_idx, -1, next_token)
    return next_token
