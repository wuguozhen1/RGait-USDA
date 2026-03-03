import torch
import torch.nn as nn

import numpy as np

def Entropy(input_):

    epsilon = 1e-10
    entropy = -input_ * torch.log(input_ + epsilon)
    entropy = torch.sum(entropy, dim=1)

    return entropy

def Eloss(entrop):

    entropy = 0.1 + torch.exp(-entrop)

    target_mask = torch.ones_like(entropy)
    target_weight = entropy * target_mask

    weight =target_weight / torch.sum(target_weight).detach().item()

    import os
    log_folder = 'output_logs/entryMIx_n'
    if not os.path.exists(log_folder):
        os.makedirs(log_folder)

        # 定义日志文件路径
    log_file_path_1 = os.path.join(log_folder, 'logMIx_n.txt')
    log_file_path_2 = os.path.join(log_folder, 'logMIx_n.txt')
    log_file_path_3 = os.path.join(log_folder, 'logMIx_n.txt')
    # 准备输出内容
    output_message_1 = (f'{entrop}\n')
    output_message_2 = (f'{entropy}\n')
    output_message_3 = (f'{weight*32}\n')

    # 将输出内容写入文件
    with open(log_file_path_1, 'a') as log_file:  # 使用 'a' 模式以追加内容到文件
        log_file.write(output_message_1)
    with open(log_file_path_2, 'a') as log_file:  # 使用 'a' 模式以追加内容到文件
        log_file.write(output_message_2)
    with open(log_file_path_3, 'a') as log_file:  # 使用 'a' 模式以追加内容到文件
        log_file.write(output_message_3)

    return weight*32


class LMMD_loss(nn.Module):
    def __init__(self, class_num=31, kernel_type='rbf', kernel_mul=2.0, kernel_num=5, fix_sigma=None):
        super(LMMD_loss, self).__init__()
        self.class_num = class_num
        self.kernel_num = kernel_num
        self.kernel_mul = kernel_mul
        self.fix_sigma = fix_sigma
        self.kernel_type = kernel_type

    def guassian_kernel(self, source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
        n_samples = int(source.size()[0]) + int(target.size()[0])
        total = torch.cat([source, target], dim=0)#合并数据(2bitch_size,class_num)
        total0 = total.unsqueeze(0).expand(
            int(total.size(0)), int(total.size(0)), int(total.size(1)))#扩展维度
        total1 = total.unsqueeze(1).expand(
            int(total.size(0)), int(total.size(0)), int(total.size(1)))
        L2_distance = ((total0-total1)**2).sum(2)
        if fix_sigma:
            bandwidth = fix_sigma
        else:
            bandwidth = torch.sum(L2_distance.data) / (n_samples**2-n_samples)#也称标准差 除去自己外的总对数(自己与自己距离为0)
        bandwidth /= kernel_mul ** (kernel_num // 2)
        bandwidth_list = [bandwidth * (kernel_mul**i)
                          for i in range(kernel_num)]#[bandwidth / 4, bandwidth / 2, bandwidth, bandwidth * 2, bandwidth * 4]

        kernel_val = [torch.exp(-L2_distance / bandwidth_temp)
                      for bandwidth_temp in bandwidth_list]#带宽对应的核值张量
        return sum(kernel_val)

    def get_loss(self, source, target, s_label, t_label):
        batch_size = source.size()[0]
        weight_ss, weight_tt, weight_st = self.cal_weight(
            s_label, t_label, batch_size=batch_size, class_num=self.class_num)
        weight_ss = torch.from_numpy(weight_ss).cuda()
        weight_tt = torch.from_numpy(weight_tt).cuda()
        weight_st = torch.from_numpy(weight_st).cuda()

        entropy = Entropy(t_label)
        wight_t = Eloss(entropy)
        wight_t = wight_t.view(-1, 1)

        kernels = self.guassian_kernel(source, target,
                                kernel_mul=self.kernel_mul, kernel_num=self.kernel_num, fix_sigma=self.fix_sigma)
        loss = torch.Tensor([0]).cuda()
        if torch.sum(torch.isnan(sum(kernels))):
            return loss
        SS = kernels[:batch_size, :batch_size]#为什么
        TT = kernels[batch_size:, batch_size:]
        ST = kernels[:batch_size, batch_size:]

        loss += torch.sum(wight_t.detach()*(weight_ss * SS + weight_tt * TT - 2 * weight_st * ST))

        return loss,torch.sum(entropy)/32

    def convert_to_onehot(self, sca_label, class_num=31):
        return np.eye(class_num)[sca_label]

    def cal_weight(self, s_label, t_label, batch_size=32, class_num=31):
        batch_size = s_label.size()[0]
        s_sca_label = s_label.cpu().data.numpy()
        s_vec_label = self.convert_to_onehot(s_sca_label, class_num=self.class_num)
        s_sum = np.sum(s_vec_label, axis=0).reshape(1, class_num)
        s_sum[s_sum == 0] = 100
        s_vec_label = s_vec_label / s_sum

        t_sca_label = t_label.cpu().data.max(1)[1].numpy()
        t_vec_label = t_label.cpu().data.numpy()
        t_sum = np.sum(t_vec_label, axis=0).reshape(1, class_num)
        t_sum[t_sum == 0] = 100
        t_vec_label = t_vec_label / t_sum

        index = list(set(s_sca_label) & set(t_sca_label))#将列表转换为集合的操作 集合无序不重复
        mask_arr = np.zeros((batch_size, class_num))
        mask_arr[:, index] = 1
        t_vec_label = t_vec_label * mask_arr
        s_vec_label = s_vec_label * mask_arr

        weight_ss = np.matmul(s_vec_label, s_vec_label.T)#点积 计算相似程度
        weight_tt = np.matmul(t_vec_label, t_vec_label.T)
        weight_st = np.matmul(s_vec_label, t_vec_label.T)

        length = len(index)
        if length != 0:
            weight_ss = weight_ss / length
            weight_tt = weight_tt / length
            weight_st = weight_st / length

        else:
            weight_ss = np.array([0])
            weight_tt = np.array([0])
            weight_st = np.array([0])

        return weight_ss.astype('float32'),weight_tt.astype('float32'), weight_st.astype('float32')