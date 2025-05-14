# @Author       : Li Haozhou
# @Time         : 2025/3/26 15:23
# @Description  :
import datasets
import peft
from peft import PeftModel, LoraConfig, PrefixTuningConfig, get_peft_model, inject_adapter_in_model
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig, BertTokenizer, BertModel,ErnieModel
import torch
import json
import random
import numpy as np
from torch import nn
from merge_peft_model import MyPeftModelForCausalLM
import Summary_Enhanced_Model
from prompter import Prompter, MultiPrompter
import util
import dataset1
import os
import csv
import re
import io
import sys
import json
from sklearn import metrics
import pandas as pd
import random

os.environ["CUDA_VISIBLE_DEVICES"] = "2"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
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

class Generate:
    def __init__(self, param):
        self.param = param
        self.prompter = MultiPrompter(self.param['template'])
        self.bert = BertModel.from_pretrained("/media/wjz/新加卷/lhz/PLMs/llama_chinese/pycache/model/code/__pycache__/cache/FinBERT/").requires_grad_(False).to("cuda")
        if self.param["doc_specific"]:
            self.tokenizer, self.roberta_tokenizer, self.peft_model, self.roberta_model, self.prefix_tuning_model = self.load_doctor_specific_model()
        else:
            self.tokenizer, self.model = self.load_model()

    def load_doctor_specific_model(self):
        tokenizer = AutoTokenizer.from_pretrained(f"{self.param['base_model']}")
        tokenizer.pad_token_id = 0
        tokenizer.padding_side = "right"
        tokenizer.truncation_side = "right"
        model = AutoModelForCausalLM.from_pretrained(
            self.param['base_model'],
            # torch_dtype=torch.float16,
            load_in_8bit=False,
            device_map="auto",
        )
        #  先加载  lora
        peft_model = PeftModel.from_pretrained(
            model,
            f'{self.param["save_model"]}lora_model',
            # torch_dtype=torch.float16,
            adapter_name="lora_model",
        )
######======================================= Fin BERT======================================================#############
        # 参数量较小，不需要量化
        roberta_tokenizer = BertTokenizer.from_pretrained(self.param["roberta_model"])
        # 添加gpt模型中的end_eos token
        roberta_tokenizer.add_special_tokens({'additional_special_tokens': ["</e>"]})
        roberta_tokenizer.truncation_side = "left"  # 过长的话，裁剪左侧文本
        roberta_model = BertModel.from_pretrained(self.param["roberta_model"]).to(device)
        # roberta_model = ErnieModel.from_pretrained(self.param["roberta_model"]).to(device)
        print(f"config:======{model.config}")
        prefix_tuning_model = Summary_Enhanced_Model.SummaryEnhanced(##  peft_config
                                                             num_layers=model.config.num_hidden_layers,
                                                             token_dim=model.config.hidden_size,  # prefix tuning中kv的维度 4096
                                                             bert_hidden_size=roberta_model.config.hidden_size).to(device)
        # 加载prefix tuning的模型权重
        prefix_tuning_model.load(f'{self.param["save_model"]}prefix_model', device="cuda")
        return tokenizer, roberta_tokenizer, peft_model, roberta_model, prefix_tuning_model

    def init_dataset(self):
        with open(self.param["test_path"], "r", encoding='utf-8') as file:
            test_data_json = [json.loads(single_data) for single_data in file.readlines()]
        test_dataset = []
        for data_point in test_data_json:
            full_prompt = self.prompter.generate_prompt(
                report=data_point["report"],
            )
            truth = data_point["output"]
            tokenized_full_prompt = self.tokenizer(full_prompt, padding=False, truncation=True,
                                                   max_length=self.param["max_len"], return_tensors="pt").to(device)
            if self.param["doc_specific"]:
                summary_tokenizer = self.roberta_tokenizer(data_point["summary"], truncation=True,
                                                           max_length=self.param["summary_length"], padding=False,
                                                           return_tensors="pt").to(device)
                test_dataset.append(
                    {"full_prompt": tokenized_full_prompt, "summary": summary_tokenizer, "truth": truth})
            else:
                test_dataset.append(tokenized_full_prompt)
        return dataset1.MyDataset(test_dataset)


    # 角色增强结果
    def generate_doc_specific(self, test_dataset):
        generation_config = GenerationConfig(
            temperature=0.3,
            do_sample=True,
            top_p=0.9,
            top_k=50,
            num_beams=1
        )
        index = 1
        predict =[]
        reference = []
        for batch in tqdm(test_dataset):
            with torch.no_grad():
                enhanced_summary= batch["summary"]
                reference_label = batch["truth"]
        # # ######======================================= Fin BERT======================================================#############
                s_input_ids, s_attention_mask, s_token_type_ids = enhanced_summary["input_ids"], enhanced_summary["attention_mask"], enhanced_summary[
                    "token_type_ids"]
                summary_bert_output = self.roberta_model(
                    input_ids=s_input_ids,
                    attention_mask=s_attention_mask,
                    token_type_ids=s_token_type_ids)
                summary_pooler_output = summary_bert_output.last_hidden_state.to(device)

                past_key_values = self.prefix_tuning_model(summary_pooler_output).to(device)
                base_config = self.peft_model.config
                past_key_values = past_key_values.view(
                    1,
                    self.param["prefix_length"] + summary_pooler_output.shape[1] + 1,
                    base_config.num_hidden_layers * 2,
                    base_config.num_attention_heads,
                    base_config.hidden_size // base_config.num_key_value_heads,
                ).to(device)
                past_key_values = past_key_values.expand(1,
                                                         self.param["prefix_length"] + summary_pooler_output.shape[1] + 1,
                                                         base_config.num_hidden_layers * 2,
                                                         base_config.num_attention_heads,
                                                         base_config.hidden_size // base_config.num_attention_heads, #4096/32/128
                                                        ).to(device)

                # 张量扩展
                past_key_values = past_key_values.permute([2, 0, 3, 1, 4]).split(2)
                # print(past_key_values.shape)
                batch_size = batch["full_prompt"]["input_ids"].shape[0]     # 为1
                prefix_padding = torch.ones(batch_size, self.param["prefix_length"]+summary_pooler_output.shape[1]+1, dtype=torch.int64).to(batch["full_prompt"]["input_ids"].device)
                input_ids = torch.cat((prefix_padding, batch["full_prompt"]["input_ids"]), dim=1).to(device)
                self.peft_model =  self.peft_model.to(device)
                generation_output = self.peft_model.generate(
                    input_ids=input_ids,
                    past_key_values=past_key_values,
                    generation_config=generation_config,
                    return_dict_in_generate=True,
                    output_scores=True,
                    max_new_tokens=50,
                    logits_processor=None,
                    max_length=1200,
                )
                s = generation_output.sequences[0]
                output = self.tokenizer.decode(s)
                print('inference')
                print(output)
                output = output.rsplit('</s>', 1)[0].split('\n')[-1]
                output = output.replace("\n", "").replace("结果：<s> ","").replace("</s>","").replace("结果：","").replace(" ", "").replace("<s>","")  # 删除\n
                print(f"{index}predict:\t{output}")
                predict.append(output)
                index += 1
                print(f"truth_label:\t{reference_label}")
                reference.append(reference_label)
                with open(f"{self.param['output_text']}", "a", encoding="utf-8") as file:
                    file.write(output + "\n")
                with open('/media/wjz/新加卷/lhz/PLMs/llama_chinese/pycache/model/code/__pycache__/cache6/code_sum_enhanced/__pycache__/finbert/SEHLP_0504/results_4400.csv', mode='w', newline='', encoding='utf-8') as file:
                 # 创建一个csv.writer对象，用于写入CSV文件
                    writer = csv.writer(file)
                  # 可选：写入头部信息，即列名
                    writer.writerow(['predict', 'reference'])
                    for row1, row2 in zip(predict, reference):
                        writer.writerow([row1, row2])

if __name__ == '__main__':
    tinyllama_doc_specific = {
        "standard_name": "SEHLP",
        "doc_specific": True,
        "save_model": "/media/wjz/新加卷/lhz/PLMs/llama_chinese/pycache/model/code/__pycache__/cache6/code_sum_enhanced/__pycache__/finbert/SEHLP_0504/checkpoint-4400/",
        "base_model": "/media/wjz/新加卷/lhz/PLMs/llama_chinese/pycache/model/code/__pycache__/cache6/lora_first/FinLLaMA",  # finllama backbone
        "roberta_model": "/media/wjz/新加卷/lhz/PLMs/llama_chinese/pycache/model/code/__pycache__/cache/FinBERT/", 
        "output_text": "/media/wjz/新加卷/lhz/PLMs/llama_chinese/pycache/model/code/__pycache__/cache6/code_sum_enhanced/__pycache__/finbert/SEHLP_0504/results_4400.txt",
        # hyperparams
        "max_len": 1200,
        # prefix_tuning hyperparams
        "prefix_length": 100,
        "summary_length": 100,
        # 测试集
        "test_path": "/media/wjz/新加卷/lhz/PLMs/llama_chinese/pycache/model/code/__pycache__/cache6/code_sum_enhanced/dataset/lcfr_test_add_sum.json",#yanbao_qwen14b_test.json
        "template": {
            "prompt_input": "作为一名金融分析师，你的任务是仔细地阅读所提供的财经研报以及其摘要，深入分析摘要与研报原文的内容，并基于这两部分内容做出判断，推荐一个适当的投资评级。请从'买入'、'谨慎增持'、'中性'、'卖出'四个选项中，选择最恰当的投资评级推荐。其中，'#摘要#'后内容为研报摘要，'#原文#'后内容为研报原文。"
                            "\n研报内容：{report}"
                            "\n结果：",
            "response_split": "结果："
        },
    }

    generate = Generate(tinyllama_doc_specific)
    # 生成测试集
    test_dataset = generate.init_dataset()
    print('begin')
    generate.generate_doc_specific(test_dataset=test_dataset)

    print('finished, start compute metrics.........')
    label_mapping = {
            "买入": "0",
            "谨慎增持": "1",
            "中性": "2",
            "卖出": "3"}
    def convert_labels_and_check_unknown(labels):
        converted_labels = []
        for label in labels:
            if label in label_mapping:
                converted_labels.append(int(label_mapping[label]))
            else:
                print(f"Warning: Unknown label '{label}'. Defaulting to '1' (其他).")
                # continue
                converted_labels.append(int(1))
        return converted_labels

    csv_file_path = '/media/wjz/新加卷/lhz/PLMs/llama_chinese/pycache/model/code/__pycache__/cache6/code_sum_enhanced/__pycache__/finbert/SEHLP_0504/results_4400.csv'
    df = pd.read_csv(csv_file_path)
    ##### 转换两个列表
    predict_num = convert_labels_and_check_unknown(df['predict'])
    reference_num = convert_labels_and_check_unknown(df['reference'])

    ###计算准确率
    accuracy = metrics.accuracy_score(reference_num, predict_num)
    ######计算精确度、召回率和F1得分
    precision = metrics.precision_score(reference_num, predict_num, average='macro',)
    recall = metrics.recall_score(reference_num, predict_num, average='macro',)
    f1 = metrics.f1_score(reference_num, predict_num, average='macro',)

    #######################打印结果
    print(f"Accuracy: {accuracy}")
    print(f"Macro F1-score: {f1}")
    print(f"Macro Recall: {recall}")
    print(f"Macro Precision: {precision}")
