# @Author       : Li Haozhou
# @Time         : 2025/3/26 12:23
# @Description  :
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_file

SAFETENSORS_WEIGHTS_NAME = "adapter_model.safetensors"
WEIGHTS_NAME = "adapter_model.bin"

class SummaryEnhanced(nn.Module):
    def __init__(self, num_layers=22, token_dim=256, prefix_length=40, bert_hidden_size=768):
        super(SummaryEnhanced, self).__init__()
        self.dd1 = nn.Linear(bert_hidden_size, bert_hidden_size, bias=False)

        self.lstm_layer = nn.LSTM(bert_hidden_size, 384, num_layers=1, batch_first=True,
                                  bidirectional=True, bias=False)  # 双向LSTM
        self.prefix_embeds = nn.Parameter(torch.empty(prefix_length, bert_hidden_size))
        # nn.init.kaiming_normal_(self.prefix_embeds, a=0, mode='fan_in', nonlinearity='leaky_relu')
        nn.init.normal_(self.prefix_embeds)
        self.prefix_proj = nn.Sequential(
            nn.Linear(bert_hidden_size, 200),
            nn.ReLU(),
            nn.Linear(200, bert_hidden_size)
        )
        self.fc_final_2 = torch.nn.Sequential(
                torch.nn.Linear(bert_hidden_size, 200),
                nn.ReLU(),
                torch.nn.Linear(200, num_layers * 2 * token_dim),# layer 4  token_dim 4096
            )

    def forward(self, pooler_output):

        sum_lstm_output, (final_hidden_state1, final_cell_state1) = self.lstm_layer(pooler_output) #[batch, lenth, 512]

        summary_cls = pooler_output[:, 0, :] # summary_cls  
        dd_out = summary_cls
        g1 = self.dd1(dd_out)
        dd_out = g1 * dd_out + g1
        g2 = self.dd1(dd_out)
        dd_out = g2 * dd_out + g2


        cls_sum_out = dd_out
        cls_sum_out = cls_sum_out.reshape(cls_sum_out.shape[0], -1, cls_sum_out.shape[1])    # 添加一个维度 [batch, 1, 768]
        prompt_embeds = torch.cat([cls_sum_out,sum_lstm_output], dim=1)   #[batch, lenth +1, 768]

        # 添加soft prefix
        prefix_embeds = self.prefix_proj(self.prefix_embeds) + self.prefix_embeds
        # prefix_embeds = self.prefix_embeds
        prefix_embeds = prefix_embeds.expand(prompt_embeds.shape[0], -1, -1)
        prompt_embeds = torch.cat([prefix_embeds, prompt_embeds], dim=1)  #[batch, (prefix lenth + sumary length), 768]
        prompt_embeds = self.fc_final_2(prompt_embeds)
        return prompt_embeds # [batch, (100 + summary length+1), 32768]

    def load(self, load_path, device="cuda"):
        if os.path.exists(os.path.join(load_path, SAFETENSORS_WEIGHTS_NAME)):
            filename = os.path.join(load_path, SAFETENSORS_WEIGHTS_NAME)
            use_safetensors = True
        else:
            filename = os.path.join(load_path, WEIGHTS_NAME)
            use_safetensors = False
        if use_safetensors:
            adapters_weights = load_file(filename, device=device)
        else:
            adapters_weights = torch.load(filename, map_location=torch.device(device))
        # 修改字典的key值
        new_adapter_weights = dict()
        for name, weight in adapters_weights.items():
            if 'prompt_encoder.prefix_model.' in name:
                new_name = name.split("prompt_encoder.prefix_model.")[1]
                new_adapter_weights[new_name] = adapters_weights[name]
        del adapters_weights
        missing_keys, unexpected_keys = self.load_state_dict(new_adapter_weights, strict=False)
        print(missing_keys, unexpected_keys)