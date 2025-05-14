# @Author       : Li Haozhou
# @Time         : 2025/1/14 19:27
# @Description  :对原始peft_model函数进行重写。
"""
1.实现对prefix_tuning 和 lora的兼容
2.修改保存模型的方式
"""
import collections
import os
import warnings
from typing import Any, Dict, List, Optional, Union
import Summary_Enhanced_Model
import torch
import torch.nn as nn
import transformers
from transformers import BertModel
from peft import PeftModelForCausalLM, PeftConfig, PeftType, get_peft_model_state_dict, PeftModel, LoraConfig, \
    PrefixTuningConfig, TRANSFORMERS_MODELS_TO_PREFIX_TUNING_POSTPROCESS_MAPPING
import packaging.version
from peft.utils import id_tensor_storage
from safetensors.torch import save_file as safe_save_file

SAFETENSORS_WEIGHTS_NAME = "adapter_model.safetensors"
WEIGHTS_NAME = "adapter_model.bin"

# 1.重写save_pretrained
# 2.重写forward
# 3.重写generate
class MyPeftModelForCausalLM(PeftModel):
    # 不变
    def __init__(self, model, peft_config: PeftConfig, adapter_name: str = "default") -> None:
        super().__init__(model, peft_config, adapter_name)
        self.base_model_prepare_inputs_for_generation = self.base_model.prepare_inputs_for_generation
        if isinstance(peft_config, PrefixTuningConfig):
            self.prefix_tuning_config = peft_config
            self.bert = BertModel.from_pretrained("/media/wjz/新加卷/lhz/PLMs/llama_chinese/pycache/model/code/__pycache__/cache/FinBERT/").requires_grad_(False).to("cuda")
            prefix_tuning_model = Summary_Enhanced_Model.SummaryEnhanced(num_layers=peft_config.num_layers,
                                                                      token_dim=peft_config.token_dim,
                                                                      prefix_length=peft_config.num_virtual_tokens,
                                                                      bert_hidden_size=768).to("cuda")
            self.prompt_encoder.update(torch.nn.ModuleDict({adapter_name: prefix_tuning_model}))
            self.prefix_length = peft_config.num_virtual_tokens  # 100 prefix lenth
            self.prefix_hidden_dim = peft_config.num_layers * 2 * peft_config.token_dim # 4 *2 *4096

    def forward(self, full_prompt, labels, summary, **kwargs):
        input_ids, attention_mask, labels = full_prompt["input_ids"], full_prompt["attention_mask"], full_prompt["labels"]
        batch_size = _get_batch_size(input_ids, None)
        # 1.调用prefix tuning 移动位置
        past_key_values, add_summary = self.get_prefix_prompt(batch_size, summary)
        if attention_mask is not None:
            # 修改输入的序列长度
            prefix_attention_mask = torch.ones(batch_size, self.prefix_tuning_config.num_virtual_tokens + add_summary + 1).to(   #加上summary length
                attention_mask.device)
            attention_mask = torch.cat((prefix_attention_mask, attention_mask), dim=1)

        return self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            past_key_values=past_key_values,
            **kwargs,
        )

    def generate(self, **kwargs):
        self.base_model.prepare_inputs_for_generation = self.prepare_inputs_for_generation
        if hasattr(self.base_model, "model"):
            self.base_model.model.generation_config = self.generation_config
        else:
            self.base_model.generation_config = self.generation_config
        try:
            outputs = self.base_model.generate(**kwargs)
        except:
            self.base_model.prepare_inputs_for_generation = self.base_model_prepare_inputs_for_generation
            raise
        else:
            self.base_model.prepare_inputs_for_generation = self.base_model_prepare_inputs_for_generation
            return outputs

    # summary
    def get_prefix_prompt(self, batch_size: int, summary) -> torch.Tensor:
# #######=======================================Fin BERT======================================================#############
        input_ids, attention_mask, token_type_ids = summary["input_ids"], summary["attention_mask"], summary["token_type_ids"]
        with torch.no_grad():
            summary_bert_output = self.bert(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids)
            summary_bert_output = summary_bert_output.last_hidden_state #shape是 (batch_size, sequence_length, hidden_size)，hidden_size=768,它是模型最后一层输出的隐藏状态
            add_summary_length = summary_bert_output.shape[1]            

        prompt_encoder = self.prompt_encoder[self.active_adapter]
        past_key_values= prompt_encoder(summary_bert_output) # [bs, 51 , num_layers * token_dim * 2]  summary_bert_output.shape[0]
        if self.base_model_torch_dtype is not None:
            past_key_values = past_key_values.to(self.base_model_torch_dtype)
        past_key_values = past_key_values.view(
            batch_size,#16
            self.prefix_tuning_config.num_virtual_tokens + add_summary_length + 1,           # prefix 的长度 ### 加长度  [16, 100, 8, 32, 128]' + sum length
            self.prefix_tuning_config.num_layers * 2, #8
            self.prefix_tuning_config.num_attention_heads, #32
            self.prefix_tuning_config.token_dim // self.prefix_tuning_config.num_attention_heads, # 4096/32=128
        )
        past_key_values = past_key_values.permute([2, 0, 3, 1, 4]).split(
            self.prefix_tuning_config.num_transformer_submodules * 2
        )
        if TRANSFORMERS_MODELS_TO_PREFIX_TUNING_POSTPROCESS_MAPPING.get(self.config.model_type, None) is not None:
            post_process_fn = TRANSFORMERS_MODELS_TO_PREFIX_TUNING_POSTPROCESS_MAPPING[self.config.model_type]
            past_key_values = post_process_fn(past_key_values)
        return past_key_values, add_summary_length

    # 保存模型
    def save_pretrained(
            self,
            save_directory: str,
            safe_serialization: bool = True,
            selected_adapters: Optional[List[str]] = None,
            save_embedding_layers: Union[str, bool] = "auto",
            is_main_process: bool = True,
            **kwargs: Any,
    ) -> None:
        if os.path.isfile(save_directory):
            raise ValueError(f"Provided path ({save_directory}) should be a directory, not a file")

        if selected_adapters is None:
            selected_adapters = list(self.peft_config.keys())
        else:
            if any(
                    selected_adapter_name not in list(self.peft_config.keys())
                    for selected_adapter_name in selected_adapters
            ):
                raise ValueError(
                    f"You passed an invalid `selected_adapters` arguments, current supported adapter names are"
                    f" {list(self.peft_config.keys())} - got {selected_adapters}."
                )

        if is_main_process:
            os.makedirs(save_directory, exist_ok=True)
            self.create_or_update_model_card(save_directory)

        # 保存模型
        model = self
        state_dict = model.state_dict()
        for adapter_name in selected_adapters:
            peft_config = self.peft_config[adapter_name]
            if peft_config.peft_type == PeftType.LORA:
                # 筛选lora
                output_state_dict = {k: state_dict[k] for k in state_dict if "lora_" in k}
                # 筛选指定adapter_name的lora
                output_state_dict = {k: v for k, v in output_state_dict.items() if
                                     (("lora_" in k and adapter_name in k) or ("bias" in k))}
            elif peft_config.peft_type == PeftType.PREFIX_TUNING:
                # 筛选prefix tuning
                output_state_dict = {k: state_dict[k] for k in state_dict if adapter_name in k}
                # 兼容性考虑
                output_state_dict["prompt_embeddings"] = torch.ones((self.prefix_length, self.prefix_hidden_dim))
            else:
                raise NotImplementedError
            output_dir = os.path.join(save_directory, adapter_name) if adapter_name != "default" else save_directory
            os.makedirs(output_dir, exist_ok=True)

            if is_main_process and safe_serialization:
                ptrs = collections.defaultdict(list)
                for name, tensor in output_state_dict.items():
                    if isinstance(tensor, torch.Tensor):
                        ptrs[id_tensor_storage(tensor)].append(name)
                    else:
                        ptrs[id(tensor)].append(name)
                shared_ptrs = {ptr: names for ptr, names in ptrs.items() if len(names) > 1}

                for _, names in shared_ptrs.items():
                    for shared_tensor_name in names[1:]:
                        output_state_dict[shared_tensor_name] = output_state_dict[shared_tensor_name].clone()

                safe_save_file(
                    output_state_dict,
                    os.path.join(output_dir, SAFETENSORS_WEIGHTS_NAME),
                    metadata={"format": "pt"},
                )
            elif is_main_process:
                torch.save(output_state_dict, os.path.join(output_dir, WEIGHTS_NAME))

            if peft_config.base_model_name_or_path is None:
                peft_config.base_model_name_or_path = (
                    self.base_model.__dict__.get("name_or_path", None)
                    if peft_config.is_prompt_learning
                    else self.base_model.model.__dict__.get("name_or_path", None)
                )
            inference_mode = peft_config.inference_mode
            peft_config.inference_mode = True

            if peft_config.task_type is None:
                # deal with auto mapping
                base_model_class = self._get_base_model_class(
                    is_prompt_tuning=peft_config.is_prompt_learning,
                )
                parent_library = base_model_class.__module__

                auto_mapping_dict = {
                    "base_model_class": base_model_class.__name__,
                    "parent_library": parent_library,
                }
            else:
                auto_mapping_dict = None

            if is_main_process:
                peft_config.save_pretrained(output_dir, auto_mapping_dict=auto_mapping_dict)
            peft_config.inference_mode = inference_mode

    # 原始未修改
    def prepare_inputs_for_generation(self, *args, task_ids: torch.Tensor = None, **kwargs):
        peft_config = self.active_peft_config
        model_kwargs = self.base_model_prepare_inputs_for_generation(*args, **kwargs)
        uses_transformers_4_37 = packaging.version.parse(transformers.__version__) >= packaging.version.parse("4.37.0")
        uses_transformers_4_36 = packaging.version.parse(transformers.__version__) >= packaging.version.parse("4.36.0")
        transformers_new_cache_archs = ["llama", "mistral", "persimmon", "phi"]
        uses_cache = uses_transformers_4_37 or (
                uses_transformers_4_36 and self.base_model.config.model_type in transformers_new_cache_archs
        )

        if peft_config.is_prompt_learning:
            if model_kwargs.get("attention_mask", None) is not None:
                if uses_cache and (model_kwargs["past_key_values"] is not None):
                    # TODO figure out why this workaround is necessary, see #1252 for context
                    size = model_kwargs["input_ids"].shape[0], model_kwargs["past_key_values"][0][0].shape[-2]
                else:
                    size = model_kwargs["input_ids"].shape[0], peft_config.num_virtual_tokens

                prefix_attention_mask = torch.ones(size).to(model_kwargs["input_ids"].device)
                model_kwargs["attention_mask"] = torch.cat(
                    (prefix_attention_mask, model_kwargs["attention_mask"]), dim=1
                )

            if model_kwargs.get("position_ids", None) is not None:
                warnings.warn("Position ids are not supported for parameter efficient tuning. Ignoring position ids.")
                model_kwargs["position_ids"] = None

            if kwargs.get("token_type_ids", None) is not None:
                warnings.warn(
                    "Token type ids are not supported for parameter efficient tuning. Ignoring token type ids"
                )
                kwargs["token_type_ids"] = None

            if model_kwargs["past_key_values"] is None and peft_config.peft_type == PeftType.PREFIX_TUNING:
                past_key_values = self.get_prompt(batch_size=model_kwargs["input_ids"].shape[0])
                model_kwargs["past_key_values"] = past_key_values
            else:
                if model_kwargs["past_key_values"] is None:
                    inputs_embeds = self.word_embeddings(model_kwargs["input_ids"])
                    prompts = self.get_prompt(batch_size=model_kwargs["input_ids"].shape[0], task_ids=task_ids)
                    prompts = prompts.to(inputs_embeds.dtype)
                    model_kwargs["inputs_embeds"] = torch.cat((prompts, inputs_embeds), dim=1)
                    model_kwargs["input_ids"] = None
        return model_kwargs

def _get_batch_size(input_ids: Optional[torch.Tensor], inputs_embeds: Optional[torch.Tensor]) -> int:
    """Get the batch size based on either input_ids or input_embeds

    Raises an ValueError if both are None.

    """
    if (input_ids is None) and (inputs_embeds is None):
        raise ValueError("You have to provide either input_ids or inputs_embeds")

    if input_ids is not None:
        batch_size = input_ids.shape[0]
    else:
        batch_size = inputs_embeds.shape[0]
    return batch_size
