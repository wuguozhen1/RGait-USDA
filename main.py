import torch
import torch.nn.functional as F
import math
import argparse
import numpy as np
import os
import torch.nn as nn
from DSAN import DSAN
import data_loader
from cosdiatacne import  CosineSimilarityLoss

def load_data(root_path, src, tar,test, batch_size):
    kwargs = {'num_workers': 1, 'pin_memory': True}
    loader_src = data_loader.load_training(root_path, src, batch_size, kwargs)
    loader_tar = data_loader.load_training(root_path, test, batch_size, kwargs)
    loader_tarju = data_loader.load_training_t(root_path, tar, batch_size, kwargs)
    loader_tar_test = data_loader.load_testing(root_path, test, batch_size, kwargs)
    return loader_src, loader_tarju, loader_tar , loader_tar_test

def creat_bank( model, dataloaders):

    #特征银行创建
    source_loader, target_trainju_loader, target_train_loader , _ = dataloaders
    iter_targetju = iter(target_trainju_loader)
    fea_bank=torch.randn(17965,256)
    score_bank = torch.randn(17965, 24).cuda()
    model.eval()
    with torch.no_grad():

        for i in range(len(target_trainju_loader)):

            data = next(iter_targetju)
            inputs = data[0]
            indx=data[-1]
            #labels = data[1]
            inputs = inputs.cuda()
            output = model.feature(inputs)

            output_norm=F.normalize(output)
            outputs = model.classer(output)

            outputs=nn.Softmax(dim=1)(outputs)


            fea_bank[indx] = output_norm.detach().clone().cpu()
            score_bank[indx] = outputs.detach().clone()  #.cpu()

    return fea_bank,score_bank
def train_epoch(epoch, model, dataloaders,loss_fn, optimizer,fea_bank,score_bank,current_task):

    #特征银行创建
    source_loader, target_trainju_loader, target_train_loader , _ = dataloaders

    iter_source = iter(source_loader)
    iter_target = iter(target_train_loader)
    num_iter = len(target_train_loader)

    model.train()
    iter_targetju = iter(target_trainju_loader)

    for i in range(1, num_iter):
        data_source, label_source = next(iter_source)
        data_target, _ = next(iter_target)
        if i % len(source_loader) == 0:  # 检查是否达到了一个完整的训练周期（epoch）
            iter_source = iter(source_loader)
        inputs_test, _, tar_idx = next(iter_targetju)

        data_source, label_source = data_source.cuda(), label_source.cuda()
        inputs_test, tar_idx = inputs_test.cuda(), tar_idx
        data_target =data_target.cuda()

        optimizer.zero_grad()
        loss= 0

        features_test = model.feature(inputs_test)
        outputs_test = model.classer(features_test)
        softmax_out = nn.Softmax(dim=1)(outputs_test)

        with torch.no_grad():
            output_f_norm = F.normalize(features_test)
            output_f_ = output_f_norm.cpu().detach().clone()

            # 更新特征库和得分库
            fea_bank[tar_idx] = output_f_.detach().clone().cpu()
            score_bank[tar_idx] = softmax_out.detach().clone()

            # 计算一阶近邻
            distance = output_f_ @ fea_bank.T  # [batch, N]
            values_near, idx_near = torch.topk(distance, dim=-1, largest=True, k=args.K + 1)
            idx_near = idx_near[:, 1:]  # [batch, K]
            values_near = values_near[:, 1:]  # [batch, K] 对应相似度

            score_near = score_bank[idx_near]  # [batch, K, C]
            fea_near = fea_bank[idx_near]  # [batch, K, dim]

            # 计算二阶近邻
            fea_bank_re = fea_bank.unsqueeze(0).expand(fea_near.shape[0], -1, -1)  # [batch, N, dim]
            distance_ = torch.bmm(fea_near, fea_bank_re.permute(0, 2, 1))  # [batch, K, N]
            values_near_near, idx_near_near = torch.topk(distance_, dim=-1, largest=True, k=args.KK + 1)
            idx_near_near = idx_near_near[:, :, 1:]  # [batch, K, KK]
            values_near_near = values_near_near[:, :, 1:]  # [batch, K, KK] 对应相似度

            # 计算权重
            tar_idx_ = tar_idx.unsqueeze(-1).unsqueeze(-1)
            match = (idx_near_near == tar_idx_).sum(-1).float()  # [batch, K]
            weight = torch.where(match > 0., match, torch.ones_like(match).fill_(0.1))  # [batch, K]

            weight_kk = weight.unsqueeze(-1).expand(-1, -1, args.KK).fill_(0.1)  # [batch, K, KK]
            weight_kk = weight_kk.contiguous().view(weight_kk.shape[0], -1)  # [batch, K*KK]

            # 处理二阶近邻得分
            score_near_kk = score_bank[idx_near_near].view(score_bank[idx_near_near].shape[0], -1, 24)
            # 或更简洁的写法：

        # ---------------------- 修改后的损失计算部分 ----------------------
        epsilon = 1e-6  # 防止除以零

        # 二阶近邻损失 (nn of nn)
        output_re = softmax_out.unsqueeze(1).expand(-1, args.K * args.KK, -1)  # [batch, K*KK, C]

        # 计算损失并除以相似度
        loss_term_kk = loss_fn(output_re, score_near_kk)  # 假设输出形状为 [batch, K*KK, C]
        loss_term_kk = loss_term_kk.sum(dim=-1)  # sum over classes [batch, K*KK]
        values_near_near_flat = values_near_near.reshape(values_near_near.shape[0], -1)  # [batch, K*KK]
        print("loss_term_kk shape:", loss_term_kk.shape)  # 假设输出 [batch_size, 32]
        print("values_near_near_flat shape:", values_near_near_flat.shape)  # 假设输出 [batch_size, 35]
        loss_term_kk /= (values_near_near_flat + epsilon)  # 除以相似度

        const = torch.mean((loss_term_kk ).sum(1))
        loss = const  # 初始化为二阶损失

        # 一阶近邻损失 (nn)
        softmax_out_un = softmax_out.unsqueeze(1).expand(-1, args.K, -1)  # [batch, K, C]

        # 计算损失并除以相似度
        loss_term = loss_fn(softmax_out_un, score_near)  # 假设输出形状为 [batch, K, C]
        loss_term = loss_term.sum(dim=-1)  # sum over classes [batch, K]
        loss_term /= (values_near + epsilon)  # 除以相似度

        loss += torch.mean((loss_term ).sum(1))  # 累加损失
        # -----------------------------------------------------------

        # 最终损失是两部分之和

        # msoftmax = softmax_out.mean(dim=0)
        # gentropy_loss = torch.sum(msoftmax *
        #                             torch.log(msoftmax + args.epsilon))
        # loss += gentropy_loss

        # self, if not explicitly removing the self feature in expanded neighbor then no need for this
        # loss += -torch.mean((softmax_out * score_self).sum(-1))
    #

        label_source_pred, loss_lmmd ,loss_entry= model(data_source, data_target, label_source)
        loss_cls = F.nll_loss(F.log_softmax(
            label_source_pred, dim=1), label_source)
        lambd = 2 / (1 + math.exp(-10 * (epoch) / args.nepoch)) - 1

        if current_task=='A':
            loss1 = 0.3*lambd* loss
            loss1.backward()
            if i % args.log_interval == 0:
                # print(
                #     f'Epoch: [{epoch:2d}], loss: {loss.item():.4f}')
                print(
                    f'Epoch: [{epoch:2d}], Loss: {loss1.item():.4f}, cls_Loss: {loss_cls.item():.4f}, loss_lmmd: {loss_lmmd.item():.4f}, loss_emtry: {loss_entry.item():.4f}, loss: {loss:.4f}')
        else:
            loss2 = loss_cls +0.5* lambd * loss_lmmd+0.3* loss_entry
            if i % args.log_interval == 0:
                # print(
                #     f'Epoch: [{epoch:2d}], loss: {loss.item():.4f}')
                print(
                    f'Epoch: [{epoch:2d}], Loss: {loss2.item():.4f}, cls_Loss: {loss_cls.item():.4f}, loss_lmmd: {loss_lmmd.item():.4f}, loss_emtry: {loss_entry.item():.4f}, loss: {loss:.4f}')

            loss2.backward()
        optimizer.step()


    return fea_bank,score_bank


def test(model, dataloader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in dataloader:
            data, target = data.cuda(), target.cuda()
            pred = model.predict(data)
            # sum up batch loss
            test_loss += F.nll_loss(F.log_softmax(pred, dim=1), target).item()
            pred = pred.data.max(1)[1]
            correct += pred.eq(target.data.view_as(pred)).cpu().sum()

        test_loss /= len(dataloader)#bitch 对正确率的影响
        print(
            f'Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(dataloader.dataset)} ({100. * correct / len(dataloader.dataset):.2f}%)')
    return correct


def get_args():
    def str2bool(v):
        if v.lower() in ('yes', 'true', 't', 'y', '1'):#转换小写再检查是否再表内并返回
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Unsupported value encountered.')

    parser = argparse.ArgumentParser()
    parser.add_argument('--root_path', type=str, help='Root path for dataset',
                        default='D:/Tools/PyCharm/dataset/DA_dataset24')
    parser.add_argument('--src', type=str,
                        help='Source domain', default='Normal/train')
    parser.add_argument('--tar', type=str,
                        help='Target domain', default='24_text/Mix_train.txt')
    parser.add_argument('--test', type=str,
                        help='Target domain', default='Mix/train_1')
    parser.add_argument('--nclass', type=int,
                        help='Number of classes', default=24)
    parser.add_argument('--batch_size', type=float,
                        help='batch size', default=32)
    parser.add_argument('--nepoch', type=int,
                        help='Total epoch num', default=200)
    parser.add_argument('--lr', type=list, help='Learning rate', default=[0.001,0.01, 0.01])
    parser.add_argument('--early_stop', type=int,
                        help='Early stoping number', default=50)
    parser.add_argument('--K', type=int, default=5)
    parser.add_argument('--KK', type=int, default=7)
    parser.add_argument('--epsilon', type=float, default=1e-5)
    parser.add_argument('--seed', type=int,
                        help='Seed', default=20241001)
    parser.add_argument('--weight', type=float,
                        help='Weight for adaptation loss', default=0.5)
    parser.add_argument('--momentum', type=float, help='Momentum', default=0.9)
    parser.add_argument('--decay', type=float,
                        help='L2 weight decay', default=5e-4)
    parser.add_argument('--bottleneck', type=str2bool,
                        nargs='?', const=True, default=True)
    parser.add_argument('--log_interval', type=int,
                        help='Log interval', default=10)
    parser.add_argument('--gpu', type=str,
                        help='GPU ID', default='0')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = get_args()
    print(vars(args))
    SEED = args.seed
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False#不理解
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    dataloaders = load_data(args.root_path, args.src,
                            args.tar, args.test,args.batch_size)
    loss_fn = CosineSimilarityLoss()
    model = DSAN(num_classes=args.nclass).cuda()
    # 加载权重（假设.pth文件中保存了整个模型的参数）
    # model_path = 'model_pre.pth'
    # loaded_state_dict = torch.load(model_path)
    # for param in model.feature_layers.parameters():
    #     param.requires_grad = False
    # 选择性地加载部分层的参数
    # 这里我们假设只想加载layer1和layer3的参数
    # model_dict = model.state_dict()

    # pretrained_dict = {k: v for k, v in loaded_state_dict.items() if
    #                    k in ['feature_layers']}
    # model_dict.update(pretrained_dict)
    # model.load_state_dict(loaded_state_dict)
    model_path = 'model_pre.pth'
    loaded_state_dict = torch.load(model_path)
    model.load_state_dict(loaded_state_dict)
    alternate_freq = 1  # 交替频率（每alternate_freq个epoch切换一次任务）
    current_task = 'A'  # 初始任务
    correct = 0
    stop = 0

    if args.bottleneck:
        optimizer = torch.optim.SGD([
            {'params': model.feature_layers.parameters(), 'lr': args.lr[0]},
            {'params': model.bottle.parameters(), 'lr': args.lr[1]},
            {'params': model.cls_fc.parameters(), 'lr': args.lr[2]},
        ], lr=args.lr[0], momentum=args.momentum, weight_decay=args.decay)
    else:
        optimizer = torch.optim.SGD([
            {'params': model.feature_layers.parameters()},
            {'params': model.cls_fc.parameters(), 'lr': args.lr[1]},
        ], lr=args.lr[0], momentum=args.momentum, weight_decay=args.decay)

    fea_bank,score_bank = creat_bank(model, dataloaders)
    for epoch in range(1, args.nepoch + 1):

        stop += 1
        for index, param_group in enumerate(optimizer.param_groups):#用于将一个可遍历的数据对象（如列表、元组或字符串）组合为一个索引序列，同时列出数据和数据下标
            param_group['lr'] = args.lr[index] / math.pow((1 + 10 * (epoch - 1) / args.nepoch), 0.75)

        fea_bank,score_bank = train_epoch(epoch, model, dataloaders, loss_fn,optimizer,fea_bank,score_bank,current_task)
        t_correct = test(model, dataloaders[-1])
        if (epoch + 1) % alternate_freq == 0:
            current_task = 'B' if current_task == 'A' else 'A'
            print(f"Switching to Task {current_task} for next epoch")
        if t_correct > correct:
            correct = t_correct
            stop = 0
            torch.save(model.state_dict(), '24_model/Nm-Mixpre_e416dis.pth')
        print(
            f'{args.src}-{args.tar}: max correct: {correct} max accuracy: {100. * correct / len(dataloaders[-1].dataset):.2f}%\n')
        import os
        log_folder = '24_log'
        if not os.path.exists(log_folder):
            os.makedirs(log_folder)

            # 定义日志文件路径
        log_file_path = os.path.join(log_folder, 'Nm-Mixpre_e416dis.txt')

        # 准备输出内容
        output_message = (
            f'{args.src}-{args.tar}: max correct: {correct} max accuracy: {100. * correct / len(dataloaders[-1].dataset):.2f}%  now accuracy:({100. * t_correct / len(dataloaders[-1].dataset):.2f}%)\n'
        )

        # 将输出内容写入文件
        with open(log_file_path, 'a') as log_file:  # 使用 'a' 模式以追加内容到文件
            log_file.write(output_message)

        if stop >= args.early_stop:
            print(
                f'Final test acc: {100. * correct / len(dataloaders[-1].dataset):.2f}%')
            break
