from typing import List, Tuple
import os
import torch
from torch import nn
from .model import ModelArgs, TransformerBlock, RMSNorm, precompute_freqs_cis

torch.set_default_device('cpu')
torch.set_default_dtype(torch.float16)

llama_2_7b_args = {"dim": 4096, "multiple_of": 256, "n_heads": 32, "n_layers": 32, "norm_eps": 1e-06, "vocab_size": 32000}
llama_2_13b_args = {"dim": 5120, "multiple_of": 256, "n_heads": 40, "n_layers": 40, "norm_eps": 1e-05, "vocab_size": 32000}
llama_2_70b_args = {"dim": 8192, "multiple_of": 4096, "ffn_dim_multiplier": 1.3, "n_heads": 64, "n_kv_heads": 8, "n_layers": 80, "norm_eps": 1e-05, "vocab_size": 32000}

global_freqs_cis = precompute_freqs_cis(128, 4096).to("cuda")


def parse_layer_name(layer_name: str):
    s = layer_name.split('/')
    return s[0], s[1]


class Worker:
    def __init__(
            self,
            my_url: str,
            mirror_url: str = "TODO",
            cache_dir: str = "~/.cache/fleece-worker/data",
    ):
        self.my_url = my_url
        self.mirror_url = mirror_url
        self.cache_dir = os.path.expanduser(cache_dir)
        self.layers = dict()
        self.task_info = dict()

    async def download_layer(self, full_layer_name):
        model_name, layer_name = parse_layer_name(full_layer_name)
        return os.path.join(self.cache_dir, "llama-2-7b-chat-slice", f"{layer_name}.pt")

    async def preload_layers(self, layer_names: List[str]):
        for full_layer_name in layer_names:
            if full_layer_name in self.layers:
                continue
            path = await self.download_layer(full_layer_name)
            model_name, layer_name = parse_layer_name(full_layer_name)
            model_args = ModelArgs(**llama_2_7b_args)  # TODO
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

    async def unload_layers(self, layer_names: List[str]):
        for full_layer_name in layer_names:
            if full_layer_name not in self.layers:
                continue  # TODO continue or warning?
            del self.layers[full_layer_name]
            torch.cuda.empty_cache()

    async def forward(self,
                      task_id: str,
                      is_new_task: bool,
                      plan: List[Tuple[str, List[str]]],
                      payload: List
                      ):
        indices = [index for index, (_url, _) in enumerate(plan) if _url == self.my_url]
        assert len(indices) == 1
        index = indices[0]
        if index == 0:
            pass
        # forward
        h = torch.HalfTensor(payload, device="cuda")
        _bsz, seqlen = h.shape
        start_pos = 0  # TODO and KV cache
        freqs_cis = global_freqs_cis[start_pos: start_pos + seqlen]
        mask = None
        if seqlen > 1:
            mask = torch.full(
                (1, 1, seqlen, seqlen), float("-inf"), device=h.device
            )
            mask = torch.triu(mask, diagonal=start_pos + 1).type_as(h)
        # layer
        _, layer_names = plan[index]
        for full_layer_name in layer_names:
            model_name, layer_name = parse_layer_name(full_layer_name)
            if layer_name == "tok_embeddings":
                h = self.layers[full_layer_name](h)
            elif layer_name.startswith("layers."):
                h, kv_cache = self.layers[full_layer_name](h, start_pos, freqs_cis, mask)
            elif layer_name == "norm":
                h = self.layers[full_layer_name](h)
            elif layer_name == "output":
                h = self.layers[full_layer_name](h)
            else:
                raise NotImplementedError("Unknown layers")
        if index == len(plan):
            pass

    async def get_info(self, req):
        pass
