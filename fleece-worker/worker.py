from typing import List, Tuple, Dict, Any
import os
import torch
from torch import nn
from .model import ModelArgs, TransformerBlock, RMSNorm, precompute_freqs_cis
# from .tokenizer import Tokenizer
import requests
import threading
import concurrent.futures

torch.set_default_device('cpu')
torch.set_default_dtype(torch.float16)

llama_2_7b_args = {"dim": 4096, "multiple_of": 256, "n_heads": 32, "n_layers": 32, "norm_eps": 1e-06, "vocab_size": 32000}
llama_2_13b_args = {"dim": 5120, "multiple_of": 256, "n_heads": 40, "n_layers": 40, "norm_eps": 1e-05, "vocab_size": 32000}
llama_2_70b_args = {"dim": 8192, "multiple_of": 4096, "ffn_dim_multiplier": 1.3, "n_heads": 64, "n_kv_heads": 8, "n_layers": 80, "norm_eps": 1e-05, "vocab_size": 32000}

global_freqs_cis = precompute_freqs_cis(128, 4096).to("cuda")
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


def requests_post(url, headers=None, json=None):
    requests.post(url, headers=headers, json=json)


def send_request(url, headers=None, json=None):
    executor.submit(requests_post, url, headers, json)


class Worker:
    def __init__(
            self,
            worker_url: str,
            mirror_url: str = "TODO",
            cache_dir: str = "~/.cache/fleece-worker/models",
    ):
        self.worker_url = worker_url
        self.mirror_url = mirror_url
        self.controller_url = None
        self.worker_token = None
        self.cache_dir = os.path.expanduser(cache_dir)
        self.layers = dict()
        self.task_info: Dict[(str, int), Tuple[int, Dict[str, Any]]] = dict()
        self.mutex = threading.Lock()

    def fetch_layer(self, full_layer_name):
        model_name, layer_name = parse_layer_name(full_layer_name)
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
            for full_layer_name in layer_names:
                if full_layer_name in self.layers:
                    continue
                path = self.fetch_layer(full_layer_name)
                model_name, layer_name = parse_layer_name(full_layer_name)
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

    def forward(self,
                task_id: str,
                is_new_task: bool,
                plan: List[Tuple[str, List[str]]],
                step: int,
                round: int,
                payload: List
                ):
        index = step
        if payload is None:
            _, kv_cache_dict = self.task_info[(task_id, step)]
            for _, kv_cache in kv_cache_dict.items():
                k_cache, v_cache = kv_cache
                del_tensor(k_cache)
                del_tensor(v_cache)
            del self.task_info[(task_id, step)]
            torch.cuda.empty_cache()
            if index < len(plan)-1:
                # next node
                send_request(
                    f"{plan[index+1][0]}/forward",
                    json={
                        "task_id": task_id,
                        "is_new_task": is_new_task,
                        "plan": plan,
                        "step": step+1,
                    })
            return
        # first node
        if index == 0:
            # bsz=1
            if is_new_task:
                pass  # TODO batch
            h = torch.tensor(payload, dtype=torch.int64, device="cuda")
            _bsz, seqlen = h.shape
        else:
            h = torch.tensor(payload, dtype=torch.float16, device="cuda")
            _bsz, seqlen, _ = h.shape
        # forward
        if is_new_task:
            self.task_info[(task_id, step)] = (0, dict())

        start_pos, kv_cache_dict = self.task_info[(task_id, step)]
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
            # TODO temperature
            next_token = torch.argmax(h[:, -1], dim=-1)
            if start_pos > 4000:
                next_token = torch.tensor([2])  # FIXME fake max length limit
            print(next_token)
            # eos_id
            if next_token[0] != 2:
                # next node
                send_request(
                    f"{plan[0][0]}/forward",
                    json={
                        "task_id": task_id,
                        "is_new_task": False,
                        "plan": plan,
                        "step": 0,
                        "round": round+1,
                        "payload": [next_token.tolist()],
                    })
            else:
                send_request(
                    f"{plan[0][0]}/forward",
                    json={
                        "task_id": task_id,
                        "is_new_task": False,
                        "plan": plan,
                        "step": 0,
                    })
            # update
            if self.controller_url is not None:
                send_request(
                    f"{self.controller_url}/update_task",
                    headers={"worker-token": self.worker_token},
                    json={
                        "t_id": task_id,
                        "plan_current_step": step,
                        "plan_current_round": round,
                        "output_tokens": next_token.tolist(),
                    })
        else:
            # next node
            send_request(
                f"{plan[index+1][0]}/forward",
                json={
                    "task_id": task_id,
                    "is_new_task": is_new_task,
                    "plan": plan,
                    "step": step+1,
                    "round": round,
                    "payload": h.tolist(),
                })
            # update
            if self.controller_url is not None:
                send_request(
                    f"{self.controller_url}/update_task",
                    headers={"worker-token": self.worker_token},
                    json={
                        "t_id": task_id,
                        "plan_current_step": step,
                        "plan_current_round": round,
                    })

    def get_info(self, req):
        pass
