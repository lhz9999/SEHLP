# @Author       : Li Haozhou
# @Time         : 2025/3/26 15:56
# @Description  : 自定义Dataset

import torch
from torch.utils.data import Dataset


class MyDataset(Dataset):
    def __init__(self, data_list):
        self.data_list = data_list

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        return self.data_list[index]


class MyDataCollator:
    def __init__(self, tokenizer, device="cuda"):
        self.tokenizer = tokenizer
        self.device = device

    def __call__(self, data_batch):
        self.tokenizer.padding_size = "left"
        context_batch = self.tokenizer.pad(data_batch, padding=True, pad_to_multiple_of=8)
        for k, v in context_batch.items():
            if not isinstance(v, torch.Tensor):
                context_batch[k] = torch.as_tensor(v, device=self.device)
        return context_batch
