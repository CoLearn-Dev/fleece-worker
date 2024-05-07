import pandas as pd
import time
from uuid import uuid4

class DataPlane:
    def __init__(self, path = './fleece-worker/dummy_gpu'):
        self.gpu_data = pd.read_csv(f'{path}/gpu.csv')
        self.mem_data = pd.read_csv(f'{path}/mem.csv')
        self.time_data = pd.read_csv(f'{path}/time.csv')

    def get_total_mem(self, device) -> int:
        if device not in self.gpu_data['Machine'].values:
            raise ValueError(f'GPU {device} not Support')
        return self.gpu_data[self.gpu_data['Machine'] == device]['Memory'].iloc[0]

class DummyGPU:

    data_plane = DataPlane()
    def __init__(self, device):
        self.device = device
        self.total_mem = DummyGPU.data_plane.get_total_mem(device)
        self.curr_mem = 0
        self.tensor_map = {}

    def load(self, layer_name):
        loading_time = DummyGPU.data_plane.time_data[(DummyGPU.data_plane.time_data['Spec'] == self.device) & (DummyGPU.data_plane.time_data['Layer'] == layer_name)]['Loading_time'].iloc[0]
        mem_usage = DummyGPU.data_plane.mem_data[DummyGPU.data_plane.mem_data['Layer'] == layer_name]['Mem_model'].iloc[0]
        if self.curr_mem + mem_usage > self.total_mem:
            raise Exception('Out of Memory in Loading')
        self.curr_mem += mem_usage
        time.sleep(loading_time)
        return
    
    def unload(self, layer_name):
        mem_usage = DummyGPU.data_plane.mem_data[DummyGPU.data_plane.mem_data['Layer'] == layer_name]['Mem_model'].iloc[0]
        self.curr_mem -= mem_usage
        return
    
    def forward(self, layer_name):
        forward_time = DummyGPU.data_plane.time_data[(DummyGPU.data_plane.time_data['Spec'] == self.device) & (DummyGPU.data_plane.time_data['Layer'] == layer_name)]['Latency_with_cache'].iloc[0]
        time.sleep(forward_time)
        return

    def create_tensor(self, shape, dtype_size = 16) -> str:
        tensor_id = str(uuid4())
        size = dtype_size
        for dim in shape:
            size *= dim
        if self.curr_mem + size > self.total_mem:
            raise Exception('Out of Memory in Tensor Creation')
        self.curr_mem += size
        self.tensor_map[tensor_id] = size
        return tensor_id

    def del_tensor(self, tensor_id: str):
        if tensor_id not in self.tensor_map:
            raise Exception('Tensor not found')
        self.curr_mem -= self.tensor_map[tensor_id]
        del self.tensor_map[tensor_id]
        return

    def available_mem(self):
        return self.total_mem - self.curr_mem

if __name__ == '__main__':
    a100 = DummyGPU('A100')
    t = time.time()
    print(a100.curr_mem)
    a100.load('llama-2-7b-chat-slice/tok_embeddings')
    print(a100.curr_mem)
    print(time.time() - t)
    t = time.time()
    a100.forward('llama-2-7b-chat-slice/tok_embeddings')
    print(a100.curr_mem)
    print(time.time() - t)
    t = time.time()
    a100.forward('llama-2-7b-chat-slice/layers')
    print(a100.curr_mem)
    print(time.time() - t)

