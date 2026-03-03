import torch
import torch.nn as nn
import torch.nn.functional as F



# 定义负余弦相似度损失函数
class CosineSimilarityLoss(nn.Module):
    def __init__(self):
        super(CosineSimilarityLoss, self).__init__()

    def forward(self, softmax_out_un, score_near):
        # 计算每个批次的每一行的余弦相似度
        cosine_sim = F.cosine_similarity(softmax_out_un, score_near, dim=-1)  # 在最后一个维度计算
        # 计算负余弦相似度（损失）
        loss = 1 - cosine_sim
        return loss  # 返回均值损失

