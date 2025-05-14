# @Author       : Li Haozhou
# @Time         : 2025/1/26 14:35
# @Description  :
# yanbao，Lora+prefix-tuning
import time
import os
import sys
import wandb
MODULE_PATH = os.path.abspath("..")
if MODULE_PATH not in sys.path:
    sys.path.append(MODULE_PATH)
import torch
import transformers
import datasets
from peft import (
    PrefixTuningConfig,
    LoraConfig,
    get_peft_model,
    prepare_model_for_int8_training, inject_adapter_in_model, PeftModel
)
import random
import peft
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, BertModel, BertTokenizer, AutoModelForSequenceClassification,ErnieModel
from dataset import data_collator
import util
from prompter import MultiPrompter
from merge_peft_model import MyPeftModelForCausalLM
# from utils import mail
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TOKENIZERS_PARALLELISM"] = "false" 

# 设置随机种子
def set_seed(seed):
    random.seed(seed)
    # np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        # torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
# 设置随机种子
seed = 2023
set_seed(seed)

class Train:
    def __init__(self, a_param):
        self.param = a_param
        self.print_param()
        self.prompter = MultiPrompter(self.param['template'])
        self.load_wandb()
        self.tokenizer, self.model, self.roberta_tokenizer, self.roberta_model = self.load_model()
        self.train_dataset, self.val_dataset = self.init_dataset(count=self.param["val_set_size"])
        self.train()

    def load_model(self):
        tokenizer = AutoTokenizer.from_pretrained(f"{self.param['base_model']}")
        tokenizer.pad_token_id = 0
        tokenizer.padding_side = "right"
        tokenizer.truncation_side = "right"
        peft.MODEL_TYPE_TO_PEFT_MODEL_MAPPING['CAUSAL_LM_MERGE'] = MyPeftModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained(
            self.param['base_model'],
            load_in_8bit=True,
            torch_dtype=torch.float16,
            device_map='auto',
        )
        model = prepare_model_for_int8_training(model)
        lora_config = LoraConfig(
            task_type="CAUSAL_LM_MERGE",
            r=self.param['lora_r'],
            lora_alpha=self.param['lora_alpha'],
            target_modules=self.param['lora_target_modules'],
            lora_dropout=self.param['lora_dropout'],
            bias="none",
        )
        prefix_tuning_config = PrefixTuningConfig(
            task_type="CAUSAL_LM_MERGE",
            num_virtual_tokens=self.param["prefix_length"],
            inference_mode=False,
            token_dim=4096,
            num_attention_heads=32,
        )
        peft_model = get_peft_model(model, prefix_tuning_config, adapter_name="prefix_model")
        peft_model = inject_adapter_in_model(lora_config, peft_model, adapter_name="lora_model")
        # 添加prefix_tuning和lora的梯度
        for n, p in peft_model.named_parameters():
            if 'prompt_encoder' in n or 'lora_' in n:##lora
                p.requires_grad = True
        for n, p in peft_model.named_parameters():
            if p.requires_grad:
                print(n)
        peft_model.print_trainable_parameters()
######=======================================Fin bert======================================================#############
        # 参数量较小，不需要量化
        roberta_tokenizer = BertTokenizer.from_pretrained(self.param["roberta_model"])
        # 添加gpt模型中的end_eos token和start_eos
        roberta_tokenizer.add_special_tokens({'additional_special_tokens': ["</e>"]})
        roberta_tokenizer.truncation_side = "left"  # 过长的话，裁剪左侧文本
        roberta_model = BertModel.from_pretrained(self.param["roberta_model"])
        return tokenizer, peft_model, roberta_tokenizer, roberta_model

    def init_dataset(self, ratio=0.1, count=0):
        data = datasets.load_dataset('json', data_files=self.param["data_path"])
        val_count = int(len(data["train"]) * ratio) if count == 0 else count
        train_val = data["train"].train_test_split(test_size=val_count, shuffle=True, seed=2023)

        def generate_and_tokenize_prompt(data_point):
            # gpt多轮次
            full_prompt = self.prompter.generate_prompt(
                report=data_point["report"],
                output=None,
            )
            tokenized_full_prompt = self.tokenize(full_prompt, data_point['output'])
            if not self.param["train_on_inputs"]:  # if False, masks out inputs in loss, input不会添加到训练指标中
                tokenized_output_prompt = self.tokenizer(data_point["output"], truncation=True,
                                                         max_length=self.param['max_len'], padding=False,
                                                         return_tensors=None)
                user_prompt_len = len(tokenized_full_prompt["input_ids"]) - len(
                    tokenized_output_prompt["input_ids"]) - 1
                tokenized_full_prompt["labels"] = [-100] * user_prompt_len + tokenized_full_prompt["labels"][
                                                                             user_prompt_len:]
            summary_tokenizer = self.roberta_tokenizer(data_point["summary"], truncation=True,
                                                      max_length=self.param["summary_length"], padding=False,
                                                      return_tensors=None)
            return {"full_prompt": tokenized_full_prompt, "summary": summary_tokenizer}

        train_data = (
            train_val["train"].shuffle(seed=42).map(generate_and_tokenize_prompt)
        )
        val_data = (
            train_val["test"].shuffle(seed=42).map(generate_and_tokenize_prompt)
        )
        return train_data, val_data

    def print_param(self):
        if int(os.environ.get("LOCAL_RANK", 0)) == 0:
            print(
                f"Training Alpaca-LoRA model with params:\n"
                f"base_model: {self.param['base_model']}\n"
                f"data_path: {self.param['data_path']}\n"
                f"output_dir: {self.param['output_dir']}\n"
                f"batch_size: {self.param['batch_size']}\n"
                f"micro_batch_size: {self.param['micro_batch_size']}\n"
                f"num_epochs: {self.param['num_epochs']}\n"
                f"learning_rate: {self.param['learning_rate']}\n"
                f"max_len: {self.param['max_len']}\n"
                f"val_set_size: {self.param['val_set_size']}\n"
                f"lora_r: {self.param['lora_r']}\n"
                f"lora_alpha: {self.param['lora_alpha']}\n"
                f"lora_dropout: {self.param['lora_dropout']}\n"
                f"lora_target_modules: {self.param['lora_target_modules']}\n"
                f"train_on_inputs: {self.param['train_on_inputs']}\n"
                f"group_by_length: {self.param['group_by_length']}\n"
                f"wandb_project: {self.param['wandb_project']}\n"
                f"wandb_watch: {self.param['wandb_watch']}\n"
                f"wandb_log_model: {self.param['wandb_log_model']}\n"
                f"resume_from_checkpoint: {self.param['resume_from_checkpoint'] or False}\n"
                f"template: {self.param['template']}\n"
            )

    def load_wandb(self):
        os.environ["WANDB_PROJECT"] = self.param["wandb_project"]
        # os.environ["WANDB_API_KEY"] = ''
        os.environ["WANDB_MODE"] = "offline"

    def tokenize(self, full_prompt, output, add_eos_token=True):
        result = self.tokenizer(full_prompt, output, truncation="only_first", max_length=self.param['max_len'] - 1, padding=False,
                                return_tensors=None)
        if (result["input_ids"][-1] != self.tokenizer.eos_token_id
                and len(result["input_ids"]) < self.param['max_len']
                and add_eos_token):
            result["input_ids"].append(self.tokenizer.eos_token_id)
            result["attention_mask"].append(1)

        result["labels"] = result["input_ids"].copy()
        return result

    def train(self):
        trainer = transformers.Trainer(
            model=self.model,
            train_dataset=self.train_dataset,
            eval_dataset=self.val_dataset,
            args=transformers.TrainingArguments(
                per_device_train_batch_size=self.param["micro_batch_size"],
                gradient_accumulation_steps=self.param['batch_size'] // self.param['micro_batch_size'],
                warmup_ratio=0.1,
                num_train_epochs=self.param["num_epochs"],
                learning_rate=self.param["learning_rate"],
                fp16=True,
                max_grad_norm=1.0,
                # bf16=True,####
                logging_steps=16,
                optim="adamw_torch",
                evaluation_strategy="steps",
                save_strategy="steps",
                eval_steps=200,
                save_steps=200, 
                output_dir=self.param["output_dir"],
                save_total_limit=12,
                # weight_decay=0.0001,
                load_best_model_at_end=True,
                ddp_find_unused_parameters=False,
                group_by_length=False,
                report_to="wandb",
                run_name=self.param["standard_name"],
            ),
            data_collator=data_collator.DataCollatorForSeq2Seq(
                self.tokenizer, self.roberta_tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
            ),

            callbacks=[util.SavePeftModelCallback],
        )

        if torch.__version__ >= "2" and sys.platform != "win32":
            self.model = torch.compile(self.model)

        with torch.autocast("cuda"):
            trainer.train(resume_from_checkpoint=self.param["resume_from_checkpoint"])
        self.model.save_pretrained(self.param["output_dir"])
        print("train over!")


if __name__ == "__main__":
    param = {
        "standard_name": "SEHLP",

        "base_model": "/media/wjz/新加卷/lhz/PLMs/llama_chinese/pycache/model/code/__pycache__/cache6/lora_first/FinLLaMA",  
        "roberta_model": "/media/wjz/新加卷/lhz/PLMs/llama_chinese/pycache/model/code/__pycache__/cache/FinBERT/",
        "data_path": "/media/wjz/新加卷/lhz/PLMs/llama_chinese/pycache/model/code/__pycache__/cache6/code_sum_enhanced/dataset/lcfr_train_add_sum.json",
        "output_dir": "./SEHLP_0504/",

        # training hyperparams
        "batch_size": 32,
        "micro_batch_size": 16,
        "num_epochs": 15,
        "learning_rate": 2e-4,
        "max_len": 1200,
        "val_set_size": 500,
        "prefix_length": 100,
        "summary_length": 100,

        # lora hyperparams
        "lora_r": 16,
        "lora_alpha": 16,
        "lora_dropout": 0.05,
        "lora_target_modules":  ["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj", "gate_proj"],
        # llm hyperparams
        "train_on_inputs": False,  # if False, masks out inpu ts in loss
        "group_by_length": False,  # faster, but produces an odd training loss curve
        # wandb params
        "wandb_project": "prefix_lora_add_sum",
        "wandb_watch": "",  # options: false | gradients | all
        "wandb_log_model": "",  # options: false | true
        "resume_from_checkpoint": None,  # either training checkpoint or final adapter
        # prompt
        "template": {
            "prompt_input": "作为一名金融分析师，你的任务是仔细地阅读所提供的财经研报以及其摘要，深入分析摘要与研报原文的内容，并基于这两部分内容做出判断，推荐一个适当的投资评级。请从'买入'、'谨慎增持'、'中性'、'卖出'四个选项中，选择最恰当的投资评级推荐。其中，'#摘要#'后内容为研报摘要，'#原文#'后内容为研报原文。"
                            "\n研报内容：{report}"
                            "\n结果：",
            "response_split": "结果："
        },
    }
    Train(param)