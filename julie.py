def train_target(args):
    dset_loaders = data_load(args)
    ## set base network
    netF = network.ResBase(res_name=args.net).cuda()

    netB = network.feat_bootleneck(type=args.classifier,
                                   feature_dim=netF.in_features,
                                   bottleneck_dim=args.bottleneck).cuda()
    netC = network.feat_classifier(type=args.layer,
                                   class_num=args.class_num,
                                   bottleneck_dim=args.bottleneck).cuda()

    modelpath = args.output_dir_src + '/source_F.pt'
    netF.load_state_dict(torch.load(modelpath))
    modelpath = args.output_dir_src + '/source_B.pt'
    netB.load_state_dict(torch.load(modelpath))
    modelpath = args.output_dir_src + '/source_C.pt'
    netC.load_state_dict(torch.load(modelpath))

    param_group = []
    param_group_c=[]
    for k, v in netF.named_parameters():
        #if k.find('bn')!=-1:
        if True:
            param_group += [{'params': v, 'lr': args.lr * 0.1}]

    for k, v in netB.named_parameters():
        if True:
            param_group += [{'params': v, 'lr': args.lr * 1}]
    for k, v in netC.named_parameters():
        param_group_c += [{'params': v, 'lr': args.lr * 1}]

    optimizer = optim.SGD(param_group)
    optimizer = op_copy(optimizer)

    optimizer_c = optim.SGD(param_group_c)
    optimizer_c = op_copy(optimizer_c)


    #building feature bank and score bank
    source_loader, target_train_loader, _ = dataloaders
    num_sample=len(target_train_loader)

    fea_bank=torch.randn(num_sample,256)
    score_bank = torch.randn(num_sample, 12).cuda()

    model.feature.eval()
    model.classer.eval()

    with torch.no_grad():
        iter_target = iter(target_train_loader)
        for i in range(len(target_train_loader)):
            data = next(iter_target)
            inputs = data[0]
            indx=data[-1]
            #labels = data[1]
            inputs = inputs.cuda()
            output = model.feature(inputs)
            output_norm=F.normalize(output)
            outputs = model.classer(output)
            outputs=nn.Softmax(-1)(outputs)
            fea_bank[indx] = output_norm.detach().clone().cpu()
            score_bank[indx] = outputs.detach().clone()  #.cpu()

    max_iter = args.max_epoch * len(target_train_loader)
    interval_iter = max_iter // args.interval
    iter_num = 0

    netF.train()
    netB.train()
    netC.train()
    acc_log=0
    while iter_num < max_iter:
        try:
            inputs_test, _, tar_idx = iter_target.next()
        except:
            iter_target = iter(target_train_loader)
            inputs_test, _, tar_idx = iter_target.next()

        if inputs_test.size(0) == 1:
            continue

        inputs_test = inputs_test.cuda()

        iter_num += 1
        lr_scheduler(optimizer, iter_num=iter_num, max_iter=max_iter)
        lr_scheduler(optimizer_c, iter_num=iter_num, max_iter=max_iter)

        features_test = model.feature(inputs_test)
        outputs_test = model.classer(features_test)
        softmax_out = nn.Softmax(dim=1)(outputs_test)
        #output_re = softmax_out.unsqueeze(1)

        with torch.no_grad():
            output_f_norm = F.normalize(features_test)
            output_f_ = output_f_norm.cpu().detach().clone()

            fea_bank[tar_idx] = output_f_.detach().clone().cpu()
            score_bank[tar_idx] = softmax_out.detach().clone()

            distance = output_f_@fea_bank.T
            _, idx_near = torch.topk(distance,
                                    dim=-1,
                                    largest=True,
                                    k=args.K+1)
            idx_near = idx_near[:, 1:]  #batch x K
            score_near = score_bank[idx_near]    #batch x K x C

            fea_near = fea_bank[idx_near]  #batch x K x num_dim
            fea_bank_re = fea_bank.unsqueeze(0).expand(fea_near.shape[0],-1,-1) # batch x n x dim
            distance_ = torch.bmm(fea_near, fea_bank_re.permute(0,2,1))  # batch x K x n
            _,idx_near_near=torch.topk(distance_,dim=-1,largest=True,k=args.KK+1)  # M near neighbors for each of above K ones
            idx_near_near = idx_near_near[:,:,1:] # batch x K x M
            tar_idx_ = tar_idx.unsqueeze(-1).unsqueeze(-1)
            match = (
                idx_near_near == tar_idx_).sum(-1).float()  # batch x K
            weight = torch.where(
                match > 0., match,
                torch.ones_like(match).fill_(0.1))  # batch x K

            weight_kk = weight.unsqueeze(-1).expand(-1, -1,
                                                    args.KK)  # batch x K x M
            weight_kk = weight_kk.fill_(0.1)

            # removing the self in expanded neighbors, or otherwise you can keep it and not use extra self regularization
            #weight_kk[idx_near_near == tar_idx_]=0

            score_near_kk = score_bank[idx_near_near]  # batch x K x M x C
            #print(weight_kk.shape)
            weight_kk = weight_kk.contiguous().view(weight_kk.shape[0],
                                                    -1)  # batch x KM

            score_near_kk = score_near_kk.contiguous().view(score_near_kk.shape[0], -1,
                                               12)  # batch x KM x C

            score_self = score_bank[tar_idx]


        # nn of nn
        output_re = softmax_out.unsqueeze(1).expand(-1, args.K * args.KK,
                                                    -1)  # batch x C x 1
        const = torch.mean(
            (F.kl_div(output_re, score_near_kk, reduction='none').sum(-1) *
             weight_kk.cuda()).sum(1)) # kl_div here equals to dot product since we do not use log for score_near_kk
        loss = torch.mean(const)


        # nn
        softmax_out_un = softmax_out.unsqueeze(1).expand(-1, args.K,
                                                         -1)  # batch x K x C

        loss += torch.mean((
            F.kl_div(softmax_out_un, score_near, reduction='none').sum(-1) *
            weight.cuda()).sum(1))

        # self, if not explicitly removing the self feature in expanded neighbor then no need for this
        #loss += -torch.mean((softmax_out * score_self).sum(-1))


        msoftmax = softmax_out.mean(dim=0)
        gentropy_loss = torch.sum(msoftmax *
                                    torch.log(msoftmax + args.epsilon))
        loss += gentropy_loss

        optimizer.zero_grad()
        optimizer_c.zero_grad()
        loss.backward()
        optimizer.step()
        optimizer_c.step()


        if iter_num % interval_iter == 0 or iter_num == max_iter:
            netF.eval()
            netB.eval()
            netC.eval()
            if args.dset == 'visda-2017':
                acc_s_te, acc_list = cal_acc(target_train_loader, netF, netB,
                                             netC,flag= True)
                log_str = 'Task: {}, Iter:{}/{}; Accuracy on target = {:.2f}%'.format(
                    args.name, iter_num, max_iter, acc_s_te
                ) + '\n' + 'T: ' + acc_list

            args.out_file.write(log_str + '\n')
            args.out_file.flush()
            print(log_str + '\n')
            netF.train()
            netB.train()
            netC.train()

            if acc_s_te>acc_log:
                acc_log=acc_s_te
                torch.save(
                    netF.state_dict(),
                    osp.join(args.output_dir, "target_F_" + '2021_'+str(args.K) + ".pt"))
                torch.save(
                    netB.state_dict(),
                    osp.join(args.output_dir,
                                "target_B_" + '2021_' + str(args.K) + ".pt"))
                torch.save(
                    netC.state_dict(),
                    osp.join(args.output_dir,
                                "target_C_" + '2021_' + str(args.K) + ".pt"))


    return netF, netB, netC