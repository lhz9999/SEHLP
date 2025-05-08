# @Author       : Duhongkai
# @Time         : 2024/1/19 15:49
# @Description  : 各种指标计算

"""
Rouge
BLEU
Distinct
"""

from nltk.translate.bleu_score import sentence_bleu
from rouge_chinese import Rouge
import nltk

hypothesis = "###刚刚发声，A股这种情况十分罕见！大聪明逆市抄底330亿，一篇研报引爆全球，市场逻辑生变？"
hypothesis = ' '.join(list(hypothesis))

reference = "刚刚过去的这个月，美股总市值暴跌了将近6万亿美元（折合人民币超过40万亿），这背后的原因可能不仅仅是加息这么简单。最近瑞士信贷知名分析师Zoltan Polzsar撰写了一篇极其重要的文章，详细分析了现有世界秩序的崩坏本质以及美国和西方将要采取的应对策略。在该文中，Zoltan Polzsar直指美国通胀的本质和其长期性。同期，A股市场亦出现了大幅杀跌的情况。"
reference = ' '.join(list(reference))


# 单个 Dist-n = \frac{Count(unique ngram)}{Count(word)}
# 平均 \bar{Dist-n} = \sum{Dist-n}/count
def diversity_score(sentence_list, n):
    """
    :param sentence_list:[[1,2,3,4,5],[6,7,8,9,10]]
    :param n:
    :return:
    """
    diversity = 0
    for sentence in sentence_list:
        ngram_counts = nltk.ngrams(sentence, n)
        unique_ngrams = set(ngram_counts)
        diversity += len(unique_ngrams) / len(sentence)
    return diversity / len(sentence_list)


rouge = Rouge()
scores = rouge.get_scores(hypothesis, reference)
print(scores)
bleu_score = sentence_bleu(reference, hypothesis)
print(bleu_score)
diversity = diversity_score([[1, 2, 2, 2, 2]], 5)
print(diversity)
