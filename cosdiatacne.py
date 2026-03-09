import torch
import torch.nn as nn
import torch.nn.functional as F



class CosineSimilarityLoss(nn.Module):
    def __init__(self):
        super(CosineSimilarityLoss, self).__init__()

    def forward(self, softmax_out_un, score_near):

        cosine_sim = F.cosine_similarity(softmax_out_un, score_near, dim=-1)  # 在最后一个维度计算

        loss = 1 - cosine_sim
        return loss

