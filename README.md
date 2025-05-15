# Code

## Environment

* python                        3.8
* accelerate                    0.21.0
* bitsandbytes                  0.42.0
* gradio                        4.19.2
* numpy                         1.26.3
* peft                          0.7.1
* scikit-learn                  1.4.2
* transformers                  4.35.2
* torch                         2.1.1+cu121

Install the dependencies using pip:
  ```
  pip install -r requirement.txt
  ```
## Usage
   ```
   CUDA_VISIBLE_DEVICES=0 python A_train_SEHLP.py 
   ```
## Evaluation Metrics
Accuracy, Macro-F1 (F1), Macro-Precision (P), Macro-Recall (R)

# Datasets
To our knowledge, publicly available datasets that leverage analyst reports and LLM-generated summaries for FSA remain absent. Therefore, we construct the LCFR-Instruct dataset from the previous benchmark that contains over 100,000 analyst reports sourced from major Chinese financial portals between 2015 and 2021. Each report includes an investment rating assigned by experienced analysts, reflecting their professional judgments and sentiments. After data augmentation and balancing, LCFR provides 16,912 reports spanning four sentiment categories: Buy, Accumulate, Neutral, and Sell. To prepare instruction-tuning samples for SEHLP, we initially prompt Qwen2.5-14B to summarize each financial report, then format each report-summary pair following the annotated templates specific to FSA: 
```
{"instruction": "[prompt]", "input": "[report and summary]", "output": "[label]"}
```
The raw texts and preprocessed data will be publicly available in the near future. If you want to use this dataset, please request my supervisor at qkpeng@mail.xjtu.edu.cn.


