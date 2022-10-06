import sys, os
from albumentations.core.composition import OneOf
base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(base_path)
print(os.getcwd())


import torch
from torch import nn, optim
import torch.nn.functional as F
import time
import copy
import argparse
import numpy as np
import torchvision
import torchvision.transforms as transforms
from nets import vgg_centroids
from torch.autograd import Variable
import random 
import matplotlib
matplotlib.use('agg') 
import matplotlib.pyplot as plt
import json
import warnings
warnings.filterwarnings('ignore')

from utils import init_dict, save_dict, curve_save, time_mark, print_cz, update_lr_multistep, remove_oldfile
from dataset import dataset_4client_10class
# from local_train_cleaning_v2_noaug import train_noise_cleaning_nc_v2, train_noise_cleaning_nc_onlyGNN, test 
from server import communication
import config # import config_kvasir as config
import server_gnn

def check_classifier_norm(
    args, 
    model
):
    for key in model.state_dict().keys():
        if "centroids_param" in key:
            param_centroids = model.state_dict()[key]
            norm_centroids = torch.mean(torch.abs(param_centroids), dim=-1, keepdim=False)
            # print_cz("* centroid norm:\t {}".format(norm_centroids), f=args.logfile)
            norm_centroids_sum = torch.sum(norm_centroids, dim=-1, keepdim=False)
            # print_cz("* centroid norm sum:\t {}".format(norm_centroids_sum), f=args.logfile)
    return norm_centroids_sum.item()

# 

def knn_adjacency(
    args,
    whole_adjacency_w,
    **kwargs
):
    knn_list = []
    new_whole_adjacency_w = torch.zeros(whole_adjacency_w.shape)
    for widx in range(whole_adjacency_w.shape[0]):
        values, indices = torch.topk(input=whole_adjacency_w[widx], k=args.topK)
        new_whole_adjacency_w[indices, :] = whole_adjacency_w[indices, :]
        new_whole_adjacency_w[:, indices] = whole_adjacency_w[:, indices]
        knn_list.append(indices)
    #
    return new_whole_adjacency_w, knn_list

def complementary_labels_v2(
    args,
    widx_tensor, # from dataset index
    batch_knn_adjacency,
    pseudo_labels,
    logits_whole_tensor,
    **kwargs
):
    """
    根据邻居的预测，取分数最低的
    """
    # bz*C 
    # tmp_onehot = torch.zeros(batch_knn_adjacency.shape[0], args.class_num)
    # 
    tmp_indices_list = []
    for idx, widx in enumerate(widx_tensor):
        neigbor_logits = torch.mean(
            logits_whole_tensor[batch_knn_adjacency[idx]],
            dim=0,
            keepdim=False
            )
        _, indices = torch.topk(neigbor_logits, k=args.class_num-2, dim=-1, largest=False) # 返回分数最低的类别
        tmp_indices_list.append(indices)
    batch_complementary_labels = torch.stack(tmp_indices_list, dim=0)
    return batch_complementary_labels

def complementary_labels_v3(
    args,
    widx_tensor, # from dataset index
    batch_knn_adjacency,
    pseudo_labels,
    logits_whole_tensor,
    **kwargs
):
    """
    v1和v2的折中，与v1没有本质区别
    排除pseudo label和邻居们的高分预测
    取分数最低的C-2
    """
    # bz*C 
    # tmp_onehot = torch.zeros(batch_knn_adjacency.shape[0], args.class_num)
    # 
    tmp_indices_list = []
    for idx, widx in enumerate(widx_tensor):
        neigbor_logits = torch.mean(
            logits_whole_tensor[batch_knn_adjacency[idx]],
            dim=0,
            keepdim=False
            )
        # 将pseudo label的维度置为最大值，这样后续会被排除
        neigbor_logits[pseudo_labels[widx]] = 5
        #
        _, indices = torch.topk(neigbor_logits, k=args.class_num-2, dim=-1, largest=False) # 返回分数最低的类别
        tmp_indices_list.append(indices)
    #
    batch_complementary_labels = torch.stack(tmp_indices_list, dim=0)
    return batch_complementary_labels


class Ensemble_NL_Loss(nn.Module):

    def __init__(
        self, 
        args, 
        **kwargs
        ):
        super(Ensemble_NL_Loss, self).__init__()
        self.class_num = args.class_num
        self.eps = args.eps
        # self.eps = 1.0e-6
    
    def forward(
        self, 
        batch_logits, 
        batch_complementary_labels
        ):
        """
        batch_logits: bz*C
        batch_complementary_labels: bz*2
        """
        # bz*2*C
        batch_complementary_labels_onehot = F.one_hot(
            batch_complementary_labels,
            self.class_num
        )
        # bz*C
        batch_complementary_labels_multihot = torch.sum(
            batch_complementary_labels_onehot,
            dim=1,
            keepdim=False
        )
        # 网络的预测概率, 进行了softmax
        sigma = F.softmax(batch_logits, dim=-1)
        # 预测概率按1取反, multihot标签考虑多个NL标签
        loss = (- batch_complementary_labels_multihot * torch.log(1.0 - sigma + self.eps)).sum(dim=-1, keepdim=False)
        loss = torch.mean(loss, dim=-1, keepdim=False)
        return loss




from sklearn import metrics
from sklearn.metrics import roc_auc_score # 
import math
import scipy


def inference_sample_info(
    args,
    model, 
    image_npy,
    noisy_label_npy,
    loss_fun_noreduce, 
    **kwargs
    ):
    model.eval()
    model.to(args.device)
    # device = next(model.parameters()).device

    noisy_class_widx_list = [[] for i in range(args.class_num)]
    loss_list = []
    feature_list = []
    logits_list = []

    model.eval()
    inference_train_loader = dataset_4client_10class.single_inference_trainloader(
        args,
        image_npy=image_npy, 
        label_npy=noisy_label_npy,
    )
    
    with torch.no_grad():
        for i, (data, target, widxs, _) in enumerate(inference_train_loader):
            batch_input = data.to(args.device).float()
            batch_noisy_label = target.to(args.device).long()
            # print("batch_input shape:\t", batch_input.shape)
            
            # 注意这里output是来自param or fc2
            batch_output, _, batch_feature = model(batch_input)
            batch_loss = loss_fun_noreduce(batch_output, batch_noisy_label) # N tensor
            batch_feature, batch_loss = batch_feature.cpu(), batch_loss.cpu()
            # output -> logits
            batch_logits = F.softmax(batch_output, dim=-1) 
            # 遍历batch内每个样本，进行分配
            for j in range(batch_input.shape[0]):
                noisy_class_widx_list[batch_noisy_label[j]].append(j+i*args.batch_size) # widx            
                loss_list.append(batch_loss[j])            
                feature_list.append(batch_feature[j])
                logits_list.append(batch_logits[j])
    #
    for class_idx in range(args.class_num):
        noisy_class_widx_list[class_idx] = torch.FloatTensor(noisy_class_widx_list[class_idx]).long()
    #
    loss_tensor = torch.stack(loss_list, dim=0).view(-1)
    feature_tensor = torch.stack(feature_list, dim=0)
    logits_tensor = torch.stack(logits_list, dim=0)
    return noisy_class_widx_list, loss_tensor, feature_tensor, logits_tensor


def gold_evaluation(
    args,
    predict_npy,
    gold_label_npy,
):
    mean_acc = 100*metrics.accuracy_score(gold_label_npy, predict_npy)
    f1_macro = 100*metrics.f1_score(y_true=gold_label_npy, y_pred=predict_npy, average='macro')
    recall = 100*metrics.recall_score(y_true=gold_label_npy, y_pred=predict_npy, average='macro')
    precision = 100*metrics.precision_score(y_true=gold_label_npy, y_pred=predict_npy, average='macro')
    kappa = 100*metrics.cohen_kappa_score(y1=gold_label_npy, y2=predict_npy)
    mcc = 100*metrics.matthews_corrcoef(y_true=gold_label_npy, y_pred=predict_npy)
    return mean_acc, f1_macro, recall, precision, kappa, mcc

def train_noise_cleaning_nc_onlyGNN_NL(
    args,
    client_idx,
    model, # 本地模型
    # fc_param_list, # 本地模型的FC参数 # 暂时不需要
    optimizer, 
    gnn_server,
    optimizer_gnn,
    loss_fun, 
    loss_fun_noreduce,
    loss_fun_nl,
    
    image_npy, # local_images,
    noisy_label_npy, # original_noisy_labels,
    gold_label_npy, 
    server_class_centroids,
    ema_client_class_centroids_list,
    **kwargs
):

    # 所有样本inference的特征, 最新的; 对所有样本进行分类
    noisy_class_widx_list, loss_tensor, feature_tensor, logits_tensor = inference_sample_info(
        args,
        model, 
        image_npy,
        noisy_label_npy,
        loss_fun_noreduce, 
    )
    # 当前client每个类别的centroid
    tmp_client_centroids_list = []
    for class_idx in range(args.class_num):
        centroid = torch.mean(feature_tensor[noisy_class_widx_list[class_idx]], dim=0, keepdim=False)
        tmp_client_centroids_list.append(centroid) # 遍历每个class
    new_client_class_centroids = torch.stack(tmp_client_centroids_list, dim=0) # C*D
    # 当前client最新centroid, 更新list中对应的记录
    # 
    if ema_client_class_centroids_list[client_idx] is None: # 初始化
        ema_client_class_centroids_list[client_idx] = new_client_class_centroids
    else:
        ema_client_class_centroids_list[client_idx] = (1.0-args.centroid_update_ema)*ema_client_class_centroids_list[client_idx] + args.centroid_update_ema*new_client_class_centroids # # K - C*D
    
    if ema_client_class_centroids_list[1] is None:
        for _client_idx_ in range(len(ema_client_class_centroids_list)):
            ema_client_class_centroids_list[_client_idx_] = new_client_class_centroids
    # 得到数据集的class centroids, 将作为server GNN的输入 
    
    # 在early stop之前, 所有noisy样本参与FL本地训练
    train_loader = dataset_4client_10class.single_inference_trainloader(
        args,
        image_npy=image_npy, 
        label_npy=noisy_label_npy,
        # state='train'
    )
    ####################################################
    # all sample for class-wise graph
    class_adjacency_list = []
    for class_idx in range(args.class_num):
        # torch tensor
        class_adjacency_w = construct_whole_graph(
            args=args,
            widx_list=noisy_class_widx_list[class_idx],
            feature_tensor=feature_tensor
        )
        class_adjacency_list.append(class_adjacency_w)
    # 叠加所有类别的邻接矩阵
    whole_adjacency = torch.sum(
            torch.stack(class_adjacency_list, dim=0),
            dim=0,
            keepdim=False
        )
    # 利用新的伪标签，对新的rest sample进行改进的ensemble negative learning
    _, knn_list = knn_adjacency(
        args=args,
        whole_adjacency_w=whole_adjacency)
    knn_tensor = torch.stack(knn_list, dim=0) ################
    
    server_class_centroids, loss_avg, mean_acc, f1_macro, auc, recall, precision, kappa, mcc, predict_npy = train_w_gnn_NL(
        args=args,
        train_loader=train_loader, 
        client_idx=client_idx,
        model=model,
        client_class_centroids_list=ema_client_class_centroids_list,
        server_class_centroids=server_class_centroids,
        optimizer=optimizer, 
        gnn_server=gnn_server,
        optimizer_gnn=optimizer_gnn,
        loss_fun=loss_fun, 
        loss_fun_nl=loss_fun_nl, #
        pseudo_labels=noisy_label_npy, #
        knn_tensor=knn_tensor, #
        logits_tensor=logits_tensor
    )
    
    return server_class_centroids, (loss_avg, mean_acc, f1_macro, auc, recall, precision, kappa, mcc), ema_client_class_centroids_list[client_idx]


def train_noise_cleaning_nc_v2(
    args,
    client_idx,
    model, # 本地模型
    # fc_param_list, # 本地模型的FC参数 # 暂时不需要
    optimizer, 
    gnn_server,
    optimizer_gnn,
    loss_fun, 
    loss_fun_noreduce,
    loss_fun_nl,
    image_npy, # local_images,
    noisy_label_npy, # original_noisy_labels,
    gold_label_npy,
    pseudo_labels, # 最新的pseudo_labels
    clean_widx,
    clean_adjacency,
    clean_class_adjacency_list,
    server_class_centroids,
    ema_client_class_centroids_list,
    **kwargs
):
    ####################################################
    # 所有样本inference的特征, 最新的; 对所有样本进行分类
    noisy_class_widx_list, loss_tensor, feature_tensor, logits_tensor = inference_sample_info(
        args,
        model, 
        image_npy,
        noisy_label_npy,
        loss_fun_noreduce, 
    )
    # 拿到所有样本inference特征及centroids后，进行基于graph的cleaning
    new_pseudo_labels, new_clean_widx, delta_widx, \
            (new_clean_adjacency, new_clean_class_adjacency_list), \
                new_client_class_centroids = noise_cleaning_iter_v3(
        args=args,
        noisy_class_widx_list=noisy_class_widx_list, 
        loss_tensor=loss_tensor, 
        feature_tensor=feature_tensor, 
        logits_tensor=logits_tensor,        
        original_noisy_labels=torch.tensor(noisy_label_npy).long(),
        pseudo_labels=pseudo_labels,
        clean_widx=clean_widx,
        clean_adjacency=clean_adjacency,
        clean_class_adjacency_list=clean_class_adjacency_list,
        server_class_centroids=server_class_centroids,
        client_class_centroids=ema_client_class_centroids_list[client_idx],
    )
    print_cz("delta_widx:\t {}".format(delta_widx.shape), f=args.logfile)
    pseudo_gold_acc, _, _, _, _, _ = gold_evaluation(args=args, predict_npy=new_pseudo_labels[new_clean_widx], gold_label_npy=gold_label_npy[new_clean_widx])

    # 当前client最新clean centroid, 更新list中对应的记录
    ema_client_class_centroids_list[client_idx] = (1.0-args.centroid_update_ema) * ema_client_class_centroids_list[client_idx] + \
                                            args.centroid_update_ema * new_client_class_centroids
    if ema_client_class_centroids_list[1] is None:
        for _client_idx_ in range(len(ema_client_class_centroids_list)):
            ema_client_class_centroids_list[_client_idx_] = new_client_class_centroids
    # 得到数据集的class centroids, 将作为server GNN的输入 
    
    ####################################################
    # 利用新的(clean/所有)样本对模型进行正常的训练
    print_cz("new clean_widx:\t {}".format(new_clean_widx.shape), f=args.logfile)
    # if clean_widx is not None:
    if new_clean_widx is not None:
        # print("train_noise_cleaning/clean_widx:\t")
        # print(clean_widx)
        train_loader = dataset_4client_10class.single_purified_trainloader(
            args,
            widx_purified=new_clean_widx,
            image_npy=image_npy, 
            label_npy=noisy_label_npy,
            # state='train'
        )
    else:
        train_loader = dataset_4client_10class.single_inference_trainloader(
                args,
                image_npy=image_npy, 
                label_npy=noisy_label_npy,
                # state='train'
            ) 
    server_class_centroids, loss_avg, mean_acc, f1_macro, auc, recall, precision, kappa, mcc, predict_npy = train_w_gnn(
        args=args,
        train_loader=train_loader, 
        client_idx=client_idx,
        model=model, 
        # fc_param_list=fc_param_list,
        client_class_centroids_list=ema_client_class_centroids_list,
        server_class_centroids=server_class_centroids,
        optimizer=optimizer, 
        gnn_server=gnn_server,
        optimizer_gnn=optimizer_gnn,
        loss_fun=loss_fun
    )
     ####################################################
    # 利用新的伪标签，对新的rest sample进行改进的ensemble negative learning
    _, knn_list = knn_adjacency(
        args=args,
        whole_adjacency_w=new_clean_adjacency)
    knn_tensor = torch.stack(knn_list, dim=0) #
    print("knn_tensor:\t", knn_tensor.shape)
    # print("whole_adjacency_w:\t")
    # print(new_clean_adjacency)
    # print("new_whole_adjacency_w:\t")
    # print(new_whole_adjacency_w)
    # print("knn_list:\t")
    # print(knn_list)
    rest_widx = torch.FloatTensor([widx for widx in list(range(feature_tensor.shape[0])) 
        if widx not in new_clean_widx.tolist()]
    ).long()
    print("rest_widx:\t", rest_widx.shape)

    rest_train_loader = dataset_4client_10class.single_purified_trainloader(
        args,
        widx_purified=rest_widx,
        image_npy=image_npy, 
        label_npy=noisy_label_npy,
        # state='train'
    )
    loss_nl_avg = train_negative(
        args=args,
        model=model, 
        rest_train_loader=rest_train_loader, 
        optimizer=optimizer, 
        loss_fun_nl=loss_fun_nl, 
        pseudo_labels=new_pseudo_labels,
        knn_tensor=knn_tensor,
        logits_tensor=logits_tensor,
    )

    print_cz("loss_nl_avg:\t{}".format(loss_nl_avg), f=args.logfile)
    
    return server_class_centroids, \
        (loss_avg, mean_acc, f1_macro, auc, recall, precision, kappa, mcc), \
            (new_pseudo_labels, new_clean_widx, (new_clean_adjacency, new_clean_class_adjacency_list), ema_client_class_centroids_list[client_idx]), pseudo_gold_acc


def train_w_gnn(
    args,
    train_loader, 
    client_idx,
    model,
    # fc_param_list,
    client_class_centroids_list,
    server_class_centroids,
    optimizer, 
    gnn_server,
    optimizer_gnn,
    loss_fun, 
    **kwargs
):
    model.train()
    model.to(args.device) 
    gnn_server.train()
    gnn_server.to(args.device) 
    loss_all = 0

    label_list_cz = [] 
    pred_list_cz = [] 
    output_list_cz = []

    input_client_class_centroids = torch.stack(
            client_class_centroids_list,
            dim=0
        ).view(len(client_class_centroids_list)*args.class_num, -1) # KC*D
    # print("client_class_centroids shape:\t", input_client_class_centroids.shape)
    centroid_convey_count = 0
    # dict record
    norm_server_list, norm_client_list, norm_gnn_input_list = [], [], []

    # for batch_idx, (data, target, sub_idxs, widxs_clean) in enumerate(train_loader):
    for batch_idx, (data, target, _, _) in enumerate(train_loader):
        # 每个batch
        if data.shape[0] == 1:
            continue
        optimizer.zero_grad() # cz mark
        model.zero_grad() 
        optimizer_gnn.zero_grad()
        gnn_server.zero_grad()
        # 
        data = data.to(args.device).float()
        target = target.to(args.device).long()

        server_class_centroids = gnn_server(input_client_class_centroids.to(args.device))
        output, _, feature = model(data)
        # 各类别的预测概率
        output_list_cz.append(torch.nn.functional.softmax(output, dim=-1).data.cpu().numpy())
        # 预测类别
        _, pred_cz = output.topk(1, 1, True, True)#取maxk个预测值lr_current = update_lr(lr=args.lr, epoch=a_iter, lr_step=args.lr_step, lr_gamma=args.lr_gamma)
        pred_list_cz.extend(
            ((pred_cz.cpu()).numpy()).tolist())
        label_list_cz.extend(
            ((target.cpu()).numpy()).tolist())
        #
        loss = loss_fun(output, target)

        # norm regularizer
        norm_centroids = torch.mean(torch.norm(model.centroids_param), dim=-1, keepdim=False)
        norm_server = torch.mean(torch.norm(server_class_centroids), dim=-1, keepdim=False)
        loss += args.norm_regularizer * norm_centroids.mean()
        loss += args.norm_regularizer * norm_server.mean()

        if batch_idx % args.centroid_interval == 0:
            output_aux = F.linear(input=feature, weight=server_class_centroids)
            loss_aux = loss_fun(output_aux, target)
            ((1.00 - args.param_ratio)*loss + args.param_ratio*loss_aux).backward()
            optimizer.step()
            optimizer_gnn.step()
            #
            norm_client = torch.sum(torch.mean(
                    torch.abs(model.state_dict()["centroids_param"].clone().detach()), dim=-1, keepdim=False
                    ), dim=-1, keepdim=False)
            # print_cz(' client param norm： {}'.format(norm_client.item()), f=args.logfile)
            norm_server = torch.sum(torch.mean(
                    torch.abs(server_class_centroids), dim=-1, keepdim=False
                    ), dim=-1, keepdim=False)
            # print_cz(' server param norm： {}'.format(norm_server.item()), f=args.logfile)
            norm_gnn_input = torch.sum(torch.mean(
                    torch.abs(input_client_class_centroids), dim=-1, keepdim=False
                    ), dim=-1, keepdim=False)
            # print_cz(' input gnn norm： {}'.format(norm_gnn_input.item()), f=args.logfile)
            # update centroid_parm from server to client
            model.update_centroids_param(
                nn.Parameter(
                    (1.0-args.ema_ratio)*model.state_dict()["centroids_param"].clone().detach() + args.ema_ratio*server_class_centroids.clone().detach()
                )
            )
            # model.update_centroids_param(
            #     nn.Parameter(
            #         (1.0-args.ema_ratio)*model.state_dict()["centroids_param"].clone().detach() + args.ema_ratio*server_class_centroids
            #     )
            # )
            centroid_convey_count += 1
            # 
            norm_client_list.append(norm_client)
            norm_server_list.append(norm_server)
            norm_gnn_input_list.append(norm_gnn_input)
            
        else:
            loss.backward()
            optimizer.step()
        #
        loss_all += loss.item()
        #
        optimizer.zero_grad() 
        model.zero_grad()
        optimizer_gnn.zero_grad()
        gnn_server.zero_grad()
    #
    print_cz("convy centroid grad in {:d} / {:d} batch".format(centroid_convey_count, len(train_loader)), f=args.logfile)
    #
    mean_acc = 100*metrics.accuracy_score(label_list_cz, pred_list_cz)
    f1_macro = 100*metrics.f1_score(y_true=label_list_cz, y_pred=pred_list_cz, average='macro')
    auc = 100.0*roc_auc_score(
        y_true=np.array(label_list_cz), 
        y_score=np.concatenate(output_list_cz, axis=0), 
        multi_class='ovr')
    recall = 100*metrics.recall_score(y_true=label_list_cz, y_pred=pred_list_cz, average='macro')
    precision = 100*metrics.precision_score(y_true=label_list_cz, y_pred=pred_list_cz, average='macro')
    kappa = 100*metrics.cohen_kappa_score(y1=label_list_cz, y2=pred_list_cz)
    mcc = 100*metrics.matthews_corrcoef(y_true=label_list_cz, y_pred=pred_list_cz)

    # update dict
    if client_idx == 0:
        args.info_dicts_norm['One']['norm_client_centroids'].append(torch.mean(torch.stack(norm_client_list)).view(-1).item())
        args.info_dicts_norm['One']['norm_server_centroids'].append(torch.mean(torch.stack(norm_server_list)).view(-1).item())
        args.info_dicts_norm['One']['norm_gnn_input'].append(torch.mean(torch.stack(norm_gnn_input_list)).view(-1).item())

    return server_class_centroids, loss_all/len(train_loader), mean_acc, f1_macro, auc, recall, precision, kappa, mcc, np.array(pred_list_cz).reshape(-1)


def train_w_gnn_NL(
    args,
    train_loader, 
    client_idx,
    model,
    # fc_param_list,
    client_class_centroids_list,
    server_class_centroids,
    optimizer, 
    gnn_server,
    optimizer_gnn,
    loss_fun, 
    loss_fun_nl, #
    pseudo_labels, #
    knn_tensor, #
    logits_tensor, #
    **kwargs
):
    model.train()
    model.to(args.device) 
    gnn_server.train()
    gnn_server.to(args.device) 
    loss_all = 0

    label_list_cz = [] 
    pred_list_cz = [] 
    output_list_cz = []

    input_client_class_centroids = torch.stack(
            client_class_centroids_list,
            dim=0
        ).view(len(client_class_centroids_list)*args.class_num, -1) # KC*D
    # print("client_class_centroids shape:\t", input_client_class_centroids.shape)
    centroid_convey_count = 0
    # dict record
    norm_server_list, norm_client_list, norm_gnn_input_list = [], [], []

    # for batch_idx, (data, target, sub_idxs, widxs_clean) in enumerate(train_loader):
    for batch_idx, (data, target, _, widxs_rest) in enumerate(train_loader):
        # 每个batch
        if data.shape[0] == 1:
            continue
        optimizer.zero_grad() # cz mark
        model.zero_grad() 
        optimizer_gnn.zero_grad()
        gnn_server.zero_grad()
        # 
        data = data.to(args.device).float()
        target = target.to(args.device).long()

        server_class_centroids = gnn_server(input_client_class_centroids.to(args.device))
        output, _, feature = model(data)
        # 各类别的预测概率
        output_list_cz.append(torch.nn.functional.softmax(output, dim=-1).data.cpu().numpy())
        # 预测类别
        _, pred_cz = output.topk(1, 1, True, True)#取maxk个预测值lr_current = update_lr(lr=args.lr, epoch=a_iter, lr_step=args.lr_step, lr_gamma=args.lr_gamma)
        pred_list_cz.extend(
            ((pred_cz.cpu()).numpy()).tolist())
        label_list_cz.extend(
            ((target.cpu()).numpy()).tolist())
        #
        loss = loss_fun(output, target)

        # norm regularizer
        norm_centroids = torch.mean(torch.norm(model.centroids_param), dim=-1, keepdim=False)
        norm_server = torch.mean(torch.norm(server_class_centroids), dim=-1, keepdim=False)
        loss += args.norm_regularizer * norm_centroids.mean()
        loss += args.norm_regularizer * norm_server.mean()

        # negative learning
        batch_complementary_labels = complementary_labels_v3(
            args=args,
            widx_tensor=widxs_rest, # from dataset index
            batch_knn_adjacency=knn_tensor[widxs_rest],
            pseudo_labels=pseudo_labels,
            logits_whole_tensor=logits_tensor
        )
        loss_nl = loss_fun_nl(
            batch_logits=output, 
            batch_complementary_labels=batch_complementary_labels
        )
        loss += args.nl_ratio * loss_nl

        if batch_idx % args.centroid_interval == 0:
            output_aux = F.linear(input=feature, weight=server_class_centroids)
            loss_aux = loss_fun(output_aux, target)
            ((1.00 - args.param_ratio)*loss + args.param_ratio*loss_aux).backward()
            optimizer.step()
            optimizer_gnn.step()
            #
            norm_client = torch.sum(torch.mean(
                    torch.abs(model.state_dict()["centroids_param"].clone().detach()), dim=-1, keepdim=False
                    ), dim=-1, keepdim=False)
            # print_cz(' client param norm： {}'.format(norm_client.item()), f=args.logfile)
            norm_server = torch.sum(torch.mean(
                    torch.abs(server_class_centroids), dim=-1, keepdim=False
                    ), dim=-1, keepdim=False)
            # print_cz(' server param norm： {}'.format(norm_server.item()), f=args.logfile)
            norm_gnn_input = torch.sum(torch.mean(
                    torch.abs(input_client_class_centroids), dim=-1, keepdim=False
                    ), dim=-1, keepdim=False)
            # print_cz(' input gnn norm： {}'.format(norm_gnn_input.item()), f=args.logfile)
            # update centroid_parm from server to client
            model.update_centroids_param(
                nn.Parameter(
                    (1.0-args.ema_ratio)*model.state_dict()["centroids_param"].clone().detach() + args.ema_ratio*server_class_centroids.clone().detach()
                )
            )
            # model.update_centroids_param(
            #     nn.Parameter(
            #         (1.0-args.ema_ratio)*model.state_dict()["centroids_param"].clone().detach() + args.ema_ratio*server_class_centroids
            #     )
            # )
            centroid_convey_count += 1
            # 
            norm_client_list.append(norm_client)
            norm_server_list.append(norm_server)
            norm_gnn_input_list.append(norm_gnn_input)
            
        else:
            loss.backward()
            optimizer.step()
        #
        loss_all += loss.item()
        #
        optimizer.zero_grad() 
        model.zero_grad()
        optimizer_gnn.zero_grad()
        gnn_server.zero_grad()
    #
    print_cz("convy centroid grad in {:d} / {:d} batch".format(centroid_convey_count, len(train_loader)), f=args.logfile)
    #
    mean_acc = 100*metrics.accuracy_score(label_list_cz, pred_list_cz)
    f1_macro = 100*metrics.f1_score(y_true=label_list_cz, y_pred=pred_list_cz, average='macro')
    auc = 100.0*roc_auc_score(
        y_true=np.array(label_list_cz), 
        y_score=np.concatenate(output_list_cz, axis=0), 
        multi_class='ovr')
    recall = 100*metrics.recall_score(y_true=label_list_cz, y_pred=pred_list_cz, average='macro')
    precision = 100*metrics.precision_score(y_true=label_list_cz, y_pred=pred_list_cz, average='macro')
    kappa = 100*metrics.cohen_kappa_score(y1=label_list_cz, y2=pred_list_cz)
    mcc = 100*metrics.matthews_corrcoef(y_true=label_list_cz, y_pred=pred_list_cz)

    # update dict
    if client_idx == 0:
        args.info_dicts_norm['One']['norm_client_centroids'].append(torch.mean(torch.stack(norm_client_list)).view(-1).item())
        args.info_dicts_norm['One']['norm_server_centroids'].append(torch.mean(torch.stack(norm_server_list)).view(-1).item())
        args.info_dicts_norm['One']['norm_gnn_input'].append(torch.mean(torch.stack(norm_gnn_input_list)).view(-1).item())

    return server_class_centroids, loss_all/len(train_loader), mean_acc, f1_macro, auc, recall, precision, kappa, mcc, np.array(pred_list_cz).reshape(-1)


def train_negative(
    args,
    model, 
    rest_train_loader, 
    optimizer, 
    loss_fun_nl, 
    pseudo_labels,
    knn_tensor,
    logits_tensor,
    **kwargs
    ):
    model.train()
    model.to(args.device) 
    loss_all = 0
    for data, target, sub_idxs, widxs_rest in rest_train_loader:
        if data.shape[0] == 1:
            continue
        optimizer.zero_grad() 
        model.zero_grad() 
        #
        data = data.to(args.device).float()
        target = target.to(args.device).long()
        output, _, feature = model(data)
        batch_complementary_labels = complementary_labels_v3(
            args=args,
            widx_tensor=widxs_rest, # from dataset index
            batch_knn_adjacency=knn_tensor[widxs_rest],
            pseudo_labels=pseudo_labels,
            logits_whole_tensor=logits_tensor
        )
        loss = args.nl_ratio * loss_fun_nl(
            batch_logits=output, 
            batch_complementary_labels=batch_complementary_labels
        )
        loss.backward()
        optimizer.step()
        optimizer.zero_grad() 
        model.zero_grad() 
        loss_all += loss.item()
    return loss_all/len(rest_train_loader)


def test(
    args,
    model, 
    test_loader, 
    loss_fun, 
    # device,
    verbose=False
):
    model.eval()
    model.to(args.device) # 
    test_loss = 0
    # correct = 0
    targets = []

    label_list_cz = [] # cz
    pred_list_cz = [] # cz
    output_list_cz = []

    with torch.no_grad():
        for data, target, idxs, _ in test_loader:
            data = data.to(args.device).float()
            target = target.to(args.device).long()
            targets.append(target.detach().cpu().numpy())

            output, _, _ = model(data)
            # cz 
            _, pred_cz = output.topk(1, 1, True, True)#
            pred_list_cz.extend(
                ((pred_cz.cpu()).numpy()).tolist())
            label_list_cz.extend(
                ((target.cpu()).numpy()).tolist())
            
            test_loss += loss_fun(output, target).item()
            # pred = output.data.max(1)[1]
            output_list_cz.append(torch.nn.functional.softmax(output, dim=-1).cpu().detach().numpy())
            # correct += pred.eq(target.view(-1)).sum().item()
    # cz
    # test_pred = np.concatenate(output_list_cz, axis=0)
    # test_label = np.array(label_list_cz)
    mean_acc = 100*metrics.accuracy_score(label_list_cz, pred_list_cz)
    f1_macro = 100*metrics.f1_score(y_true=label_list_cz, y_pred=pred_list_cz, average='macro')
    auc = 100.0*roc_auc_score(
        y_true=np.array(label_list_cz), 
        y_score=np.concatenate(output_list_cz, axis=0), 
        multi_class='ovr'
        )
    
    recall = 100*metrics.recall_score(y_true=label_list_cz, y_pred=pred_list_cz, average='macro')
    precision = 100*metrics.precision_score(y_true=label_list_cz, y_pred=pred_list_cz, average='macro')
    
    kappa = 100*metrics.cohen_kappa_score(y1=label_list_cz, y2=pred_list_cz)
    mcc = 100*metrics.matthews_corrcoef(y_true=label_list_cz, y_pred=pred_list_cz)

    # return test_loss/len(test_loader), mean_acc, f1_macro, auc
    if verbose:
        classification_report = metrics.classification_report(y_true=label_list_cz, y_pred=pred_list_cz, digits=5)
        confusion_matrix=metrics.confusion_matrix(y_true=label_list_cz, y_pred=pred_list_cz)
        return test_loss/len(test_loader), mean_acc, f1_macro, auc, recall, precision, kappa, mcc, classification_report, confusion_matrix 
    else:
        return test_loss/len(test_loader), mean_acc, f1_macro, auc, recall, precision, kappa, mcc 
    

def collect_features(
    args,
    model, 
    test_loader, 
):
    model.eval()
    model.to(args.device) # 
    feature_list, target_list = [], []
    with torch.no_grad():
        for data, target, _, _ in test_loader:
            data = data.to(args.device).float()
            _, _, features = model(data)
            for j in range(data.shape[0]):
                feature_list.append(features[j])
                target_list.append(target[j])
            del data
    feature_tensor = torch.stack(feature_list, dim=0)
    target_tensor = torch.stack(target_list, dim=0)
    return feature_tensor, target_tensor
    

"""
local_images:
noisy_labels:
cidx: class-wise idx for each class
widx: whole_idx, 0 to N-1 for each sample in dataset
whole_graph_adjacency: N*N
whole_graph_pseudo_labels: N
class_centroids: C*D, e.g., 4*512
class_subgraphs_adjacency: C*(N/C)*(N/C), e.g., 4*(N/4)*(N/4)
class_subgraphs_idx: [(subgraph_idx, whole_idx)]
class_subgraphs_pseudo_labels:

"""
def noisy_class_widx(
    args,
    noisy_labels,
    **kwargs
):  
    """
    using noisy/pseudo labels
    N dataset widx -> 4*(N/4) class widx
    """
    noisy_class_widx_list = [[] for i in range(args.class_num)]
    N = noisy_labels.shape[0]
    for i in range(N):
        noisy_class_widx_list[noisy_labels[i]].append(i)
    # 确认转换无误，不同noisy类别的样本之和==原本数据集的样本数
    assert sum([len(noisy_class_widx_list[i] for i in range(args.class_num))]) == N, "not equal samples: {:d} vs. {:d}".format(
        sum([len(noisy_class_widx_list[i] for i in range(args.class_num))]), 
        N
    )
    # 每个类别打包成tensor
    for class_idx in range(args.class_num):
        noisy_class_widx_list[class_idx] = torch.FloatTensor(noisy_class_widx_list[class_idx]).long()
    #
    return noisy_class_widx_list


def build_clean_core_graph_v3(
    args,
    noisy_class_widx_list,
    loss_tensor,
    feature_tensor,
    **kwargs
):
   
    loss_class_list = [loss_tensor[noisy_class_widx_list[class_idx]] for class_idx in range(args.class_num)]
    clean_widx_list = []
    clean_class_adjacency_list = []
    tmp_centroids_list = []
    for class_idx in range(args.class_num):
        # 每个class中clean core样本数量
        core_num = int(args.core_ratio*len(loss_class_list[class_idx])) 
        loss_idx_tensor = torch.FloatTensor(loss_class_list[class_idx]) # N_class
        # 找出阈值
        # print("loss_idx_tensor:\t", type(loss_idx_tensor))
        # print(loss_idx_tensor.shape)
        sorted_loss, _ = torch.sort(loss_idx_tensor, dim=0, descending=False)
        # print("sorted_loss:\t", type(sorted_loss))
        # print(sorted_loss.shape)
        core_thresh = sorted_loss[core_num]
        ########################################
        # 获得该类别core graph的cidx
        ########################################
        core_cidx = torch.where(loss_idx_tensor<=core_thresh)[0]
        # clean_core_cidx_list[class_idx].extend(core_cidx) 
        ########################################
        # cidx转换到widx
        ########################################
        core_widx = noisy_class_widx_list[class_idx][core_cidx] # 得到clean core的widx
        clean_widx_list.extend(core_widx) #单一列表
        print_cz("class_idx:\t {}".format(class_idx), f=args.logfile)
        print_cz("core_widx:\t {}".format(len(core_widx)), f=args.logfile)
        print_cz("clean_widx_list:\t {}".format(len(clean_widx_list)), f=args.logfile)

        ########################################
        # 直接计算whole_adjacency_w
        ########################################
        # torch tensor
        class_adjacency_w = construct_whole_graph(
            args=args,
            widx_list=core_widx,
            feature_tensor=feature_tensor
        )
        clean_class_adjacency_list.append(class_adjacency_w)
        #
        ########################################
        # 得到每个类别的centroid
        ########################################
        class_centroid = graph_class_centroid(
            args=args,
            class_adjacency=class_adjacency_w,
            feature_tensor=feature_tensor 
            # check, 理论上应与feature_whole_tensor_select结果相同
        )
        tmp_centroids_list.append(class_centroid)
    class_centroids = torch.stack(tmp_centroids_list, dim=0)
    # 叠加所有类别的邻接矩阵
    clean_adjacency = torch.sum(
            torch.stack(clean_class_adjacency_list, dim=0),
            dim=0,
            keepdim=False
        )
    # return clean_core_cidx_list
    clean_widx = torch.FloatTensor(clean_widx_list).long()
    #
    return class_centroids, clean_widx, (clean_adjacency, clean_class_adjacency_list)
    

def construct_whole_graph(
    args,
    widx_list,
    feature_tensor,
    **kwargs
):
    if widx_list is None:
        # dense adj
        feature_tensor_masked = feature_tensor
    else:
        # print("construct_whole_graph feature_tensor.device:\t", feature_tensor.device)
        feature_tensor_masked = torch.zeros(feature_tensor.shape).to(feature_tensor.device)
        feature_tensor_masked[widx_list] = feature_tensor[widx_list]
        # print("construct whole graph with masked-widx tensor...")
    #
    graph_similarity_list = []
    for widx in range(feature_tensor_masked.shape[0]):
        similarity = compute_similarity_ova(feature_k=feature_tensor_masked[widx], feature_all=feature_tensor_masked)
        graph_similarity_list.append(similarity)
    #
    whole_graph_adjacency = torch.stack(graph_similarity_list, dim=0)
    assert whole_graph_adjacency.shape[0] == whole_graph_adjacency.shape[1], "non-square adj, shape: {}".format(
        whole_graph_adjacency.shape
    )
    
    return whole_graph_adjacency # torch tensor


def construct_whole_inter_graph(
    args,
    widx_1,
    widx_2,
    feature_tensor,
    **kwargs
):
    if widx_1 is None or widx_2 is None:
        # dense adj
        print("Error with None widx_list")
    else:
        # print("construct_whole_inter_graph feature_tensor.device:\t", feature_tensor.device)
        feature_tensor_masked_1 = torch.zeros(feature_tensor.shape).to(feature_tensor.device)
        feature_tensor_masked_1[widx_1] = feature_tensor[widx_1]
        feature_tensor_masked_2 = torch.zeros(feature_tensor.shape).to(feature_tensor.device)
        feature_tensor_masked_2[widx_2] = feature_tensor[widx_2]
        print("construct inter graph with masked-widx tensor...")
    graph_similarity_list = []
    for widx in range(feature_tensor_masked_1.shape[0]):
        similarity = compute_similarity_ova(feature_k=feature_tensor_masked_1[widx], feature_all=feature_tensor_masked_2)
        graph_similarity_list.append(similarity)
    #
    inter_graph_adjacency = torch.stack(graph_similarity_list, dim=0)
    assert inter_graph_adjacency.shape[0] == inter_graph_adjacency.shape[1], "non-square adj, shape: {}".format(
        inter_graph_adjacency.shape
    )
    return inter_graph_adjacency
 


def check_graph(
    args,
    graph_adjacency,
    **kwargs
):
    """
    纠正<0, nan, inf的情况，并report
    """
    graph_adjacency[graph_adjacency<0.0] = 0.0
    return graph_adjacency

def compute_similarity_ova(
    feature_k,
    feature_all
):  
    """
    feature_k: D
    feature_all: N*D
    one vs. all
    """
    similarity = F.cosine_similarity(
        x1=feature_all, 
        x2=feature_k.view(1, -1), 
        dim=1, 
        eps=1e-08
    )
    return similarity

def graph_class_centroid(
    args,
    class_adjacency,
    feature_tensor,
    **kwargs 
):  
    assert class_adjacency.shape[0] == feature_tensor.shape[0], "unmatched size of adj vs tensor, {:d}/{:d}".format(class_adjacency.shape[0], feature_tensor.shape[0])
    centroid = torch.zeros(feature_tensor[0].shape)
    E_sum = class_adjacency.sum(dim=-1,keepdim=False).sum(dim=-1,keepdim=False) # 标量
    for idx in range(class_adjacency.shape[0]):
        # 利用第idx个节点的edge权重之和，对第idx节点的特征进行加权
        E_node = class_adjacency[idx].sum(dim=-1,keepdim=False) # 标量
        centroid += (E_node/E_sum) * feature_tensor[idx]
    #
    return centroid # torch tensor

def classification_with_centroid(
    args,
    feature_all,
    class_centroids,
    **kwargs
):
    similarity_all = F.cosine_similarity(
        x2=feature_all.view(feature_all.shape[0], -1, 1), 
        x1=torch.transpose(
            class_centroids.clone().detach(), 
            dim0=0,
            dim1=1
            ).view(1, -1, args.class_num), 
        dim=1, 
        eps=1e-08
    ).view(feature_all.shape[0], args.class_num) # N*C

    pred_class_all = similarity_all.max(dim=-1)[1]
    # print_cz('pred_class_all shape:\t {}'.format(pred_class_all.shape), f=args.logfile)
    return pred_class_all


def extend_clean_graph(
    args,
    feature_tensor,
    pseudo_labels,
    model_logits,
    server_class_centroids,
    client_class_centroids,
    clean_widx,
    clean_adjacency,
    **kwargs
):

    delta_widx_list = []
    delta_class_widx_list = [[] for class_idx in range(args.class_num)]
    # delta_adjacency = torch.zeros(clean_adjacency.shape)
    delta_class_adjacency_list = []
    # delta_class_adjacency_list = [torch.zeros(clean_adjacency.shape) for class_idx in range(args.class_num)]
    server_centroid_preds = classification_with_centroid(
        args,
        feature_all=feature_tensor,
        class_centroids=server_class_centroids
    )
    # print("extend_clean_graph/classification_with_centroid/client_class_centroids:\n", client_class_centroids)
    client_centroid_preds = classification_with_centroid(
        args,
        feature_all=feature_tensor,
        class_centroids=client_class_centroids
    )

    # 转化成list进行元素判断
    clean_widx_list = clean_widx.tolist()
    for widx in range(feature_tensor.shape[0]):
        if widx in clean_widx_list: # 不考虑已经加入graph的样本
            assert torch.sum(clean_adjacency[widx], dim=-1, keepdim=False) > 0, "single node {:d}: {}".format(
                widx, clean_adjacency
            )
            continue 
        else: # 考察目前graph之外的样本，按照step 2
            # print("widx:\t", widx)
            # print("server_centroid_preds[widx]:\t", server_centroid_preds[widx])
            # print("model_logits[widx, server_centroid_preds[widx]]:\t", model_logits[widx, server_centroid_preds[widx]])
            if pseudo_labels[widx] == server_centroid_preds[widx] \
                    and server_centroid_preds[widx] == client_centroid_preds[widx] \
                    and model_logits[widx, server_centroid_preds[widx]] > args.confid_thresh:
                delta_widx_list.append(widx)
                delta_class_widx_list[pseudo_labels[widx]].append(widx)
        #
    #
    delta_adjacency = construct_whole_graph(
        args,
        widx_list=delta_widx_list,
        feature_tensor=feature_tensor
    )
    # class-wise delta matrix
    # 不等价, 排除了类别之间的连接
    for class_idx in range(args.class_num):
        # delta_adjacency_tmp = delta_adjacency.clone().detach()
        # print("extend_clean_graph delta_adjacency.device:\t", delta_adjacency.device)
        delta_adjacency_tmp = torch.zeros(delta_adjacency.shape).to(delta_adjacency.device)
        delta_adjacency_tmp[delta_class_widx_list[class_idx], :] = delta_adjacency[delta_class_widx_list[class_idx], :]
        delta_adjacency_tmp[:, delta_class_widx_list[class_idx]] = delta_adjacency[:, delta_class_widx_list[class_idx]]
        #
        delta_class_adjacency_list.append(delta_adjacency_tmp)
    #
    delta_widx = torch.FloatTensor(delta_widx_list).long()
    # 每个类别打包成tensor
    # for class_idx in range(args.class_num):
    #     delta_class_widx_list[class_idx] = torch.FloatTensor(delta_class_widx_list[class_idx]).long()
    #
    return delta_widx, (delta_adjacency, delta_class_adjacency_list), (client_centroid_preds, server_centroid_preds)
    # return (delta_widx, delta_class_widx_list), (delta_adjacency, delta_class_adjacency_list), (client_centroid_preds, server_centroid_preds)
    # return (delta_widx_list, delta_class_widx_list), (delta_adjacency, delta_class_adjacency_list), (client_centroid_preds, server_centroid_preds)

def construct_inter_graph(
    args,
    rest_widx,
    clean_widx,
    feature_tensor,
    **kwargs
):
    
    tmp_inter_adjancey_1 = construct_whole_inter_graph(
        args=args,
        widx_1=rest_widx,
        widx_2=clean_widx,
        feature_tensor=feature_tensor
    )
    tmp_inter_adjancey_2 = construct_whole_inter_graph(
        args=args,
        widx_1=clean_widx,
        widx_2=rest_widx,
        feature_tensor=feature_tensor
    )
    inter_adjancey = (tmp_inter_adjancey_1 + tmp_inter_adjancey_2)/2
    return inter_adjancey

def optimize_pseudo_labels(
    args,
    adjacency_matrix,
    clean_widx,
    rest_widx,
    last_pseudo_labels,
    client_pred_labels,
    server_pred_labels,
    original_noisy_labels,
    **kwargs,
):
    """
    pseudo label for other to-optimize samples (not clean core, not delta)
    label propagation in Step 3
    """
    # normalize graph
    W = adjacency_matrix.clone().detach().numpy()
    W = W + np.eye(W.shape[0])
    print("***"*5)
    print("W:\t")
    # print(W)
    print(W.max(), W.min())
    S = W.sum(axis=1)
    # S[S==0] = 1
    # print("***"*5)
    # print("S info:")
    # print(S)
    # print(S.max(), S.min())
    De = np.array(1.0/np.sqrt(S)) # 遇到分母为0
    De = np.diag(De)
    # 需要矩阵乘法
    # 大致确定老版本的scipy的*表示矩阵乘法，新版本为element-wise，矩阵乘法要用@
    # Wn = De*W*De 
    Wn = np.dot(
        np.dot(De, W),
        De
    )
    #####################################
    whole_widx = np.array(list(range(adjacency_matrix.shape[0])))

    Z = np.zeros((Wn.shape[0], args.class_num))
    # A = np.eye(Wn.shape[0]) - args.alpha*Wn #######
    # # # 使用注释的这套A和y，F1=87% （Line 828， Line 836-838, Line 848）
    # remove previous pseudo label constraints
    A = 2 * (np.eye(Wn.shape[0]) - args.alpha*Wn)
    start_time = time.time()
    for class_idx in range(args.class_num):
        
        ##############
        # cur_pseudo_widx = whole_widx[last_pseudo_labels[whole_widx] == class_idx]
        # y_pseudo = np.zeros((Wn.shape[0],))
        # y_pseudo[cur_pseudo_widx] = 1.0
        #
        cur_client_widx = whole_widx[client_pred_labels[whole_widx] == class_idx]
        y_client = np.zeros((Wn.shape[0],))
        y_client[cur_client_widx] = 1.0
        #
        cur_server_widx = whole_widx[server_pred_labels[whole_widx] == class_idx]
        y_server = np.zeros((Wn.shape[0],))
        y_server[cur_server_widx] = 1.0
        ##############
        # y = y_pseudo + y_client + y_server
        y = (1.0 - args.alpha) * (y_client + y_server)

        f, _ = scipy.sparse.linalg.cg(
            A, 
            y, 
            tol=1e-7
        )
        Z[:,class_idx] = f

    Z[Z<0] = 0
    probs_l1 = F.normalize(torch.tensor(Z),1).numpy()
    probs_l1[probs_l1 <0] = 0
    p_labels = np.argmax(probs_l1,1)

    # 对于clean_widx, pseudo label与clean label一致
    p_labels[clean_widx] = last_pseudo_labels[clean_widx]
    ########## 此处存疑，应该是之前的pseudo label吧，而非original_noisy_labels
    # p_labels[clean_widx] = original_noisy_labels[clean_widx]

    print("pseudo time:\t{:.2f}".format(time.time()-start_time))
    return p_labels


def combine_lists(
    list_1,
    list_2
):
    list_combine = []
    list_combine.extend(list_1)
    list_combine.extend(list_2)
    return list_combine

def update_clean_graph(
    args,
    clean_widx,
    clean_adjacency,
    clean_class_adjacency_list,
    delta_widx,
    # delta_class_widx_list,
    delta_adjacency,
    delta_class_adjacency_list,
    # class_centroids,
    feature_tensor,
    # clean_labels,
    **kwargs,
):  
    # clean_labels: N, -1表示非clean
    
    ########################################
    # 更新widx_list和class_widx_list
    ########################################
    clean_widx_list = clean_widx.tolist()
    delta_widx_list = delta_widx.tolist()
    if len(delta_widx_list) > 0: # 若delta非空
        new_clean_widx_list = combine_lists(clean_widx_list, delta_widx_list)
        new_clean_widx, _ = torch.sort(
            torch.FloatTensor(new_clean_widx_list).long(),
            dim=-1, 
            descending=False
        )
    else:
        new_clean_widx = clean_widx
    #

    ########################################
    # 更新adjacency_w
    ########################################
    new_clean_adjacency = clean_adjacency + delta_adjacency
    new_clean_class_adjacency_list = [adj1+adj2 for (adj1, adj2) in  zip(clean_class_adjacency_list, delta_class_adjacency_list)]

    ########################################
    # 更新class centroids
    ########################################
    tmp_new_centroids_list = []
    # print("update_clean_graph:\n")
    # print(new_clean_class_adjacency_list[0].shape)
    # print(new_clean_class_adjacency_list[0])
    for class_idx in range(args.class_num):
        new_centroid = graph_class_centroid(
            args,
            new_clean_class_adjacency_list[class_idx],
            feature_tensor
        )
        tmp_new_centroids_list.append(new_centroid)
    new_class_centroids = torch.stack(tmp_new_centroids_list, dim=0)

    return new_clean_widx, (new_clean_adjacency, new_clean_class_adjacency_list), new_class_centroids


def noise_cleaning_iter_v3(
    args,
    noisy_class_widx_list, 
    loss_tensor, 
    feature_tensor, 
    logits_tensor,
    original_noisy_labels,
    pseudo_labels,
    clean_widx,
    clean_adjacency,
    clean_class_adjacency_list,
    server_class_centroids,
    client_class_centroids,
    **kwargs
):

    if clean_adjacency is None:
        ########################################
        # step 1
        # 初始化clean graph
        ########################################
        print_cz("* 初始化clean core graph", f=args.logfile)
        client_class_centroids, clean_widx, \
                (clean_adjacency, clean_class_adjacency_list) = build_clean_core_graph_v3(
            args=args,
            noisy_class_widx_list=noisy_class_widx_list,
            loss_tensor=loss_tensor,
            feature_tensor=feature_tensor
        )
    ########################################
    # step 2
    # 利用一致性，扩大clean graph, 产生delta graph
    ########################################
    if pseudo_labels is None:
        pseudo_labels = original_noisy_labels
    #
    # (delta_widx, delta_class_widx_list), \
    delta_widx, \
        (delta_adjacency, delta_class_adjacency_list), \
            (client_pred_labels, server_pred_labels) = extend_clean_graph(
        args=args,
        feature_tensor=feature_tensor,
        pseudo_labels=pseudo_labels,
        model_logits=logits_tensor,
        server_class_centroids=server_class_centroids,
        client_class_centroids=client_class_centroids,
        clean_widx=clean_widx,
        clean_adjacency=clean_adjacency,
    )

    ########################################
    # step 3
    # 利用label propagation, 优化剩余样本的pseudo labels
    ########################################
    combine_widx_list =clean_widx.tolist() + delta_widx.tolist()
    rest_widx_list = [
        widx for widx in list(range(feature_tensor.shape[0])) 
        if widx not in combine_widx_list
    ]
    rest_widx = torch.FloatTensor(rest_widx_list).long()
    combine_widx = torch.FloatTensor(combine_widx_list).long()

    inter_adjacency = construct_inter_graph(
        args=args,
        rest_widx=rest_widx, # list -> tensor
        clean_widx=clean_widx,
        feature_tensor=feature_tensor
    )
    new_pseudo_labels = optimize_pseudo_labels(
        args=args,
        adjacency_matrix=inter_adjacency,
        clean_widx=combine_widx, # list -> tensor
        rest_widx=rest_widx,
        last_pseudo_labels=pseudo_labels,
        client_pred_labels=client_pred_labels,
        server_pred_labels=server_pred_labels,
        original_noisy_labels=original_noisy_labels
    )

    ########################################
    # step 4
    # 合并delta graph, 更新centroids
    ########################################
    new_clean_widx, \
        (new_clean_adjacency, new_clean_class_adjacency_list), \
            new_client_class_centroids = update_clean_graph(
        args=args,
        clean_widx=clean_widx, # list -> tensor
        clean_adjacency=clean_adjacency,
        clean_class_adjacency_list=clean_class_adjacency_list,
        delta_widx=delta_widx,
        # delta_class_widx_list=delta_class_widx_list,
        delta_adjacency=delta_adjacency,
        delta_class_adjacency_list=delta_class_adjacency_list,
        feature_tensor=feature_tensor,
    )

    return new_pseudo_labels, new_clean_widx, delta_widx, \
            (new_clean_adjacency, new_clean_class_adjacency_list), \
                new_client_class_centroids



if __name__ == '__main__':
    """
    数据集: K个noisy train set当作client side, 1个clean valid set和1个clean test set用于挑选模型和性能评估
    最终希望server端的模型server model能在clean test set上取得优异性能 -> 考虑更频繁的通信
    也可以将client端local model在valid/test set测试进行辅助训练关注, 但需要注意server aggregation和local training对性能的影响
    """
    ########################################
    # hyper-parameters and new dir    
    ########################################
    args = config.get_args()
    args.main_file = __file__
    #
    log_path = args.save_path + time_mark() \
        + '_{}'.format(args.theme) \
        + '_{}'.format(args.optim) \
        + '_lr{}'.format(args.lr) \
        + '_steps{}'.format(args.lr_multistep) \
        + '_gamma{}'.format(args.lr_gamma) \
        + '_seed{:d}'.format(args.seed) \
        + '_iters{:d}'.format(args.iters) \
        + '_wk{}'.format(args.wk_iters) 
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    logfile = open(os.path.join(log_path,'log.txt'), 'a')
    with open(log_path + '/setting.json', 'w') as f:
        f.write(json.dumps(args.__dict__, indent=4))
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if args.device.lower() == 'cuda' and torch.cuda.is_available():
        args.device = torch.device('cuda')
    else:
        args.device = torch.device('cpu')
    # specific seed
    seed= args.seed
    np.random.seed(seed)
    torch.manual_seed(seed)     
    torch.cuda.manual_seed_all(seed) 
    #
    random.seed(seed) # new
    torch.backends.cudnn.deterministic = True # new
    #
    # print args info
    config.args_info(args, logfile=logfile)
    
    args.logfile = logfile
    SAVE_PTH_NAME = 'model'

    # import shutil
    # save_file_list =[
    #     __file__,
    #     'server_gnn.py'
    # ]
    # for filename in save_file_list:
    #     print_cz("log_path: {},  filename: {}".format(log_path, filename), f=args.logfile)
    #     filename = filename.split('/')[-1]
    #     print_cz("source: {}".format(os.path.join('/home/zchen72/code/noiseFL-v2/', filename)), f=args.logfile)
    #     print_cz("target: {}".format(os.path.join(log_path, filename)), f=args.logfile)
    #     shutil.copyfile(
    #         os.path.join('/home/zchen72/code/noiseFL-v2/', filename),
    #         os.path.join(log_path, filename)
    #     )

           
    ########################################
    # prepare the data    
    ########################################
    noise_rate_list = [float(noise_rate) for noise_rate in args.noise_rates.split('_')]
    print_cz("* noisy rates: {} / {}".format(noise_rate_list, args.noise_rates), f=args.logfile)
    (train_loaders, valid_loader, test_loader), \
        (train_image_npy_list, train_noisy_label_npy_list, train_clean_label_npy_list), \
            (valid_image_npy, valid_label_npy), \
                (test_image_npy, test_label_npy) = dataset_4client_10class.prepare_data_kvasir_4clients_noisy(
                        args=args,
                        data_dir=config.kvasir_data_dir_10class,
                        csv_dir=config.noisy_kvasir_csv_dir_10class,
                        noisy_type=args.noisy_type,
                        noise_rate=noise_rate_list
                    )

    train_len = [len(loader) for loader in train_loaders]
    valid_len  = [len(valid_loader)]
    test_len  = [len(test_loader)]
    print_cz('Train loader len:  {}'.format(train_len), f=logfile)
    print_cz('Valid loader len:  {}'.format(valid_len), f=logfile)
    print_cz('Test  loader len:  {}'.format(test_len), f=logfile)

    # name of each client dataset
    datasets = ['A', 'B', 'C', 'D']

    ############ record curve #####################
    info_keys = [ 
        'train_epochs','test_epochs', 
        'train_loss', 'train_f1', 'train_mcc', 
        'train_clean_loss', 'train_clean_f1', 'train_clean_mcc',
        'test_loss', 'test_f1', 'test_mcc',
        'lr',
        'param_ratio',
        'pseudo_gold_acc',
        'clean_ratio',
        'server_test_f1'
    ]
    info_dicts = {
        'Server': init_dict(keys=info_keys),  
        'A': init_dict(keys=info_keys), 
        'B': init_dict(keys=info_keys), 
        'C': init_dict(keys=info_keys), 
        'D': init_dict(keys=info_keys), 
    }
    # out of training
    info_keys_norm = [ 
        'norm_epochs',
        'norm_centroids',
    ]
    info_dicts_norm = {
        'Server': init_dict(keys=info_keys_norm),  
        'A': init_dict(keys=info_keys_norm), 
        'B': init_dict(keys=info_keys_norm), 
        'C': init_dict(keys=info_keys_norm), 
        'D': init_dict(keys=info_keys_norm), 
    }
    # within training
    info_keys_norm_within = ['norm_epochs', 'norm_client_centroids','norm_server_centroids', 'norm_gnn_input']
    args.info_dicts_norm = {
        'One': init_dict(keys=info_keys_norm_within),  
    }


    ########################################
    # model initialization at server and clients
    ########################################
    if args.network == 'vgg11_nb_small': # no_bias, default
        server_model = vgg_centroids.VGG11_Slim2_Centroids_SingleHead(n_classes=args.class_num)#.to(device)
        print_cz("* backbone vgg11_nb_small with {}".format(args.network), f=args.logfile)
    else:
        print_cz("* Error backbone {}".format(args.network), f=args.logfile)

    # 包含LayerNorm
    gnn_server = server_gnn.Server_Centroids_GNN(
        args=args,
        in_channel=64, 
        inter_channel=64,
        out_channel=64, 
        res_connect=True
    )

    # client_num = len(datasets)
    client_num = len(train_loaders)
    args.client_num = client_num #########
    client_weights = [1.0/client_num for _client_idx_ in range(client_num)] # client importance
    client_models = [copy.deepcopy(server_model) for _client_idx_ in range(client_num)] # personalized model list 
    # loss fun
    loss_fun = nn.CrossEntropyLoss() ## loss
    loss_fun_noreduce = nn.CrossEntropyLoss(reduce=False) ## loss
    loss_fun_nl = Ensemble_NL_Loss(args=args)

    start_time = time.time()
    # initial for save
    # valid_best_f1 = 0
    # valid_best_iter = 0
    test_best_f1 = 0
    test_best_iter = 0 
    ########################################
    # 以下6个变量，保持更新
    # 包含多个client，需要用一个list进行封装
    pseudo_labels_list = [None for _client_idx_ in range(client_num)] # 开始时为None, 对train_noise_cleaning合法
    clean_widx_list = [None for _client_idx_ in range(client_num)] #[0,1,2,3] for each client
    clean_adjacency_list = [None for _client_idx_ in range(client_num)] # 开始时为None, 对train_noise_cleaning合法
    clean_class_adjacency_list2 = [None for _client_idx_ in range(client_num)] # 开始时为None, 对train_noise_cleaning合法

    # 初始化server class centroids及list
    fc_param_list = []
    for _client_idx_ in range(client_num):
        fc_param_list.append(client_models[_client_idx_].state_dict()["centroids_param"].clone().detach().to('cpu'))
    server_class_centroids = torch.mean(
        torch.stack(fc_param_list, dim=0),
        dim=0,
        keepdim=False
    )
    print_cz("初始化server class centroids", f=args.logfile)
    server_class_centroids_list = [server_class_centroids.clone().detach() for _client_idx_ in range(client_num)] # 为了client之间并行化
    # 将在train函数内部收集sample信息进行初始化
    ema_client_class_centroids_list = [None for _client_idx_ in range(client_num)] # K-element list, K - C*D

    ########################################
    # start training
    ########################################
    lr_current = args.lr
    lr_steps = [int(float(lr_ratio_step)*args.iters) for lr_ratio_step in args.lr_multistep.split('_')]
    print_cz("* lr_multistep {} / iters {}".format(lr_steps, args.iters), f=args.logfile)
    param_ratio_slope = (args.param_ratio_end - args.param_ratio_begin)/args.iters
    for a_iter in range(args.iters): #
        iter_start_time = time.time()
        print_cz("============== Iteration {}/{} ==============".format(a_iter, args.iters-1), f=logfile)
        ########################################
        # update learning strategy
        ########################################
        args.param_ratio = args.param_ratio_begin + param_ratio_slope*a_iter
        # update lr
        lr_current = update_lr_multistep(
            args=args, 
            lr_current=lr_current, 
            step=a_iter, 
            lr_steps=lr_steps, 
            lr_gamma=args.lr_gamma
        )
        # update optimizer
        if (args.optim).lower() == 'sgd':
            optimizers = [
                optim.SGD(
                    params=client_models[idx].parameters(), 
                    lr=lr_current, 
                    weight_decay=args.wd
                    ) 
                    for idx in range(client_num)
                ]
            optimizer_gnn = optim.SGD(
                        params=gnn_server.parameters(), 
                        lr=lr_current, 
                        weight_decay=args.wd
                    ) 
        elif (args.optim).lower() == 'adam':
            optimizers = [
                optim.Adam(
                    params=client_models[idx].parameters(), 
                    lr=lr_current, 
                    weight_decay=args.wd
                    ) 
                    for idx in range(client_num)
                ]
            optimizer_gnn = optim.Adam(
                        params=gnn_server.parameters(), 
                        lr=lr_current, 
                        weight_decay=args.wd
                    ) 
        ########################################
        # local training in a_iter-th round
        ########################################
        for wi in range(args.wk_iters):
            epoch = wi + a_iter * args.wk_iters
            print_cz("#####"*7 + " new epoch {:d} ".format(epoch) + "#####"*7, f=args.logfile)
            print_cz("============ Local Train epoch {} ============".format(epoch), f=logfile)
            print_cz("=== lr_current:  {:.4e} ===".format(lr_current), f=logfile)
            ########################################
            # local traininig for all clients
            ########################################
            for client_idx in range(client_num):
                print_cz("*****"*5 + " new client {:d} ".format(client_idx) + "*****"*5, f=args.logfile)
                ########################################
                # local train for each client
                ########################################
                # client_models[client_idx].to(args.device)
                # print_cz("client_idx:\t {}".format(client_idx))
                #
                if a_iter >= args.warm_iter:
                    print_cz("* Noise clean at {}-client + server GNN ...".format(client_idx), f=args.logfile)
                    # 得到更新后的server class centroids
                    server_class_centroids, (client_train_loss, client_train_acc, client_train_f1,client_train_auc, client_train_recall, client_train_precision, client_train_kappa, client_train_mcc), \
                        (pseudo_labels_list[client_idx], clean_widx_list[client_idx], 
                            (clean_adjacency_list[client_idx], clean_class_adjacency_list2[client_idx]), ema_client_class_centroids_list[client_idx]
                            ), pseudo_gold_acc = train_noise_cleaning_nc_v2(
                                args=args,  
                                client_idx=client_idx,
                                model=client_models[client_idx], 
                                # fc_param_list=fc_param_list,  # 暂时不需要
                                optimizer=optimizers[client_idx], 
                                gnn_server=gnn_server,
                                optimizer_gnn=optimizer_gnn,
                                loss_fun=loss_fun,  
                                loss_fun_noreduce=loss_fun_noreduce,
                                loss_fun_nl=loss_fun_nl,
                                image_npy=train_image_npy_list[client_idx], #
                                noisy_label_npy=train_noisy_label_npy_list[client_idx], #original_noisy_labels,
                                gold_label_npy=train_clean_label_npy_list[client_idx],
                                pseudo_labels=pseudo_labels_list[client_idx], # 开始时为None, 对train_noise_cleaning合法
                                clean_widx=clean_widx_list[client_idx],
                                clean_adjacency=clean_adjacency_list[client_idx], # 开始时为None, 对train_noise_cleaning合法
                                clean_class_adjacency_list=clean_class_adjacency_list2[client_idx], # 开始时为None, 对train_noise_cleaning合法
                                server_class_centroids=server_class_centroids_list[client_idx], # 将server centroids优化，由clients之间的串行改为并行
                                ema_client_class_centroids_list=ema_client_class_centroids_list
                    )
                    print_cz("* pseudo_gold_acc: {:.2f}".format(pseudo_gold_acc), f=args.logfile)
                    info_dicts[datasets[client_idx]]['pseudo_gold_acc'].append(pseudo_gold_acc)
                    clean_ratio = 100.0*len(clean_widx_list[client_idx])/train_noisy_label_npy_list[client_idx].shape[0]
                    print_cz("* clean_ratio: {:.2f}".format(clean_ratio), f=args.logfile)
                    info_dicts[datasets[client_idx]]['clean_ratio'].append(clean_ratio)
                    # 更新server class centroids
                    server_class_centroids_list[client_idx] = server_class_centroids
                else:
                    print_cz("* Warm up with only server GNN, also update server centroids", f=args.logfile)
                    server_class_centroids, (client_train_loss, client_train_acc, client_train_f1, client_train_auc, client_train_recall, client_train_precision, client_train_kappa, client_train_mcc), ema_client_class_centroids_list[client_idx] = train_noise_cleaning_nc_onlyGNN_NL(
                                args=args,  
                                client_idx=client_idx,
                                model=client_models[client_idx], 
                                # fc_param_list=fc_param_list,  # 暂时不需要 
                                optimizer=optimizers[client_idx], 
                                gnn_server=gnn_server,
                                optimizer_gnn=optimizer_gnn,
                                loss_fun=loss_fun,  
                                loss_fun_noreduce=loss_fun_noreduce,
                                loss_fun_nl=loss_fun_nl, #
                                image_npy=train_image_npy_list[client_idx], #
                                noisy_label_npy=train_noisy_label_npy_list[client_idx], #original_noisy_labels,
                                gold_label_npy=train_clean_label_npy_list[client_idx],
                                server_class_centroids=server_class_centroids_list[client_idx], # 将server centroids优化，由clients之间的串行改为并行
                                ema_client_class_centroids_list=ema_client_class_centroids_list
                    )               
                    # 更新server class centroids
                    server_class_centroids_list[client_idx] = server_class_centroids
                #
                print_cz(' Client {:<5s}| Local Train_Loss: {:.4f} | Acc: {:.2f}  F1: {:.2f}  AUC: {:.2f}  Recall: {:.2f}  Preci: {:.2f}  Kappa: {:.2f}  MCC: {:.2f}'.format(
                        datasets[client_idx], 
                        client_train_loss, 
                        client_train_acc, 
                        client_train_f1, 
                        client_train_auc,
                        client_train_recall, 
                        client_train_precision, 
                        client_train_kappa, 
                        client_train_mcc
                    ), 
                    f=logfile
                )
                # all training records before aggregation
                info_dicts[datasets[client_idx]]['train_epochs'].append(epoch)
                info_dicts[datasets[client_idx]]['train_loss'].append(client_train_loss)
                info_dicts[datasets[client_idx]]['train_f1'].append(client_train_f1)
                info_dicts[datasets[client_idx]]['train_mcc'].append(client_train_mcc)
                #
                info_dicts[datasets[client_idx]]['lr'].append(lr_current)
                #
                info_dicts[datasets[client_idx]]['param_ratio'].append(args.param_ratio)

                # test server centroid
                test_feature_tensor, test_label_tesnor = collect_features(
                    args=args,
                    model=client_models[client_idx], 
                    test_loader=test_loader, 
                )
                server_centroid_preds = classification_with_centroid(
                    args,
                    feature_all=test_feature_tensor,
                    class_centroids=server_class_centroids
                )
                test_server_f1_macro = 100*metrics.f1_score(y_true=test_label_tesnor.cpu().numpy(), y_pred=server_centroid_preds.cpu().numpy(), average='macro')
                print_cz(' Client {:<5s} Server centroid | Test F1: {:.2f}'.format(
                        datasets[client_idx], 
                        test_server_f1_macro
                    ), 
                    f=logfile
                )
                info_dicts[datasets[client_idx]]['server_test_f1'].append(test_server_f1_macro)
                
            # train all client models for 1-epoch
            ########################################
            # end of one epoch local training
            ########################################
            print_cz('***', f=logfile)
            
            ########################################
            # test client model after local train
            ########################################
            client_test_list = []
            for client_idx in range(client_num):
                client_test_loss, client_test_acc, client_test_f1, client_test_auc, \
                    client_test_recall, client_test_precision, client_test_kappa, client_test_mcc \
                    = test(
                            args,
                            client_models[client_idx], 
                            test_loader, 
                            loss_fun, 
                        )
                client_test_list.append(
                    [client_test_loss, 
                    client_test_acc, 
                    client_test_f1, 
                    client_test_auc,
                    client_test_recall, 
                    client_test_precision, 
                    client_test_kappa, 
                    client_test_mcc]
                    )
                print_cz(' Client {:<5s}| Local Test  Loss: {:.4f} | Acc: {:.2f}  F1: {:.2f}  AUC: {:.2f}  Recall: {:.2f}  Preci: {:.2f}  Kappa: {:.2f}  MCC: {:.2f}'.format(
                        datasets[client_idx], 
                        client_test_loss, 
                        client_test_acc, 
                        client_test_f1, 
                        client_test_auc,
                        client_test_recall, 
                        client_test_precision, 
                        client_test_kappa, 
                        client_test_mcc
                    ), 
                    f=logfile
                )
                ############ record curve #####################
                if wi < args.wk_iters-1:
                    # client model before aggregation
                    info_dicts[datasets[client_idx]]['test_epochs'].append(epoch)
                    info_dicts[datasets[client_idx]]['test_loss'].append(client_test_loss)
                    info_dicts[datasets[client_idx]]['test_f1'].append(client_test_f1)
                    info_dicts[datasets[client_idx]]['test_mcc'].append(client_test_mcc)
            client_test_macro = np.mean(np.array(client_test_list), axis=0)
            print_cz(' Client {}| Local Test  Loss: {:.4f} | Acc: {:.2f}  F1: {:.2f}  AUC: {:.2f}  Recall: {:.2f}  Preci: {:.2f}  Kappa: {:.2f}  MCC: {:.2f}'.format(
                    'Average', 
                    client_test_macro[0], 
                    client_test_macro[1], 
                    client_test_macro[2], 
                    client_test_macro[3],
                    client_test_macro[4], 
                    client_test_macro[5], 
                    client_test_macro[6], 
                    client_test_macro[7]
                ), 
                f=logfile
            )
            print_cz('***'*2, f=logfile)
        ##########################
        # print(client_weights)
        
        ########################################
        # aggregation
        ########################################
        print_cz(' Aggregation ', f=logfile)
        server_model, client_models = communication(
            args=args, 
            server_model=server_model, 
            models=client_models, 
            client_weights=client_weights, 
            # a_iter, 
            logfile=logfile
        )
        # 此时client和server模型具有相同的backbone和FC2，server模型具有自己的centroids_param

        # client centroids_param也进行fedavg，并更新
        # 目前client的centroids_param参与了更新
        fc_param_list = []
        for model in client_models:
            fc_param_list.append(model.state_dict()["centroids_param"].clone().detach().to('cpu'))
        avg_client_class_param = torch.mean(
                torch.stack(fc_param_list, dim=0),
                dim=0,
                keepdim=False
            )
        for model in client_models:
            model.update_centroids_param(nn.Parameter(avg_client_class_param.clone().detach()))
        # 更新server分类器参数
        # 使用client平均的centroids_param作为server的新参数，也是fedavg的思想
        server_model.update_centroids_param(
                nn.Parameter(avg_client_class_param.clone().detach())
            )
        server_class_centroids_list = [avg_client_class_param.clone().detach() for _client_idx_ in range(client_num)]
        print_cz("* 新的averaged server class centroids，嵌入server model", f=args.logfile)
        
        print_cz('* check centroid norm', f=logfile)
        for _client_idx_ in range(len(client_models)):
            print_cz("updated client {}".format(datasets[_client_idx_]), f=args.logfile)
            norm_centroids_sum = check_classifier_norm(args, client_models[_client_idx_])
            info_dicts_norm[datasets[_client_idx_]]['norm_epochs'].append(epoch) 
            info_dicts_norm[datasets[_client_idx_]]['norm_centroids'].append(norm_centroids_sum)
        #
        print_cz("updated server", f=args.logfile)
        norm_centroids_sum = check_classifier_norm(args, server_model)
        info_dicts_norm['Server']['norm_epochs'].append(epoch)
        info_dicts_norm['Server']['norm_centroids'].append(norm_centroids_sum)
        

        # ########################################
        # # server valid
        # ########################################
        valid_loss, valid_acc, valid_f1, valid_auc, \
            valid_recall, valid_precision, valid_kappa, valid_mcc = test(
                args,
                server_model, 
                valid_loader, 
                loss_fun, 
            )
        print_cz(' # Server| Valid  Loss: {:.4f} | Acc: {:.2f}  F1: {:.2f}  AUC: {:.2f}'.format(
                valid_loss, 
                valid_acc, 
                valid_f1, 
                valid_auc
                ), 
                f=logfile
            )

        ########################################
        # server test
        ########################################
        test_loss, test_acc, test_f1, test_auc, \
            test_recall, test_precision, test_kappa, test_mcc = test(
                args,
                server_model, 
                test_loader, 
                loss_fun, 
            )
        print_cz(' # Server| Test  Loss: {:.4f} | Acc: {:.2f}  F1: {:.2f}  AUC: {:.2f}  Recall: {:.2f}  Preci: {:.2f}  Kappa: {:.2f}  MCC: {:.2f}'.format(
                test_loss, 
                test_acc, 
                test_f1, 
                test_auc,
                test_recall, 
                test_precision, 
                test_kappa, 
                test_mcc
                ), 
                f=logfile
        )
        # server test performance for record
        for i in range(args.wk_iters):
            info_dicts['Server']['test_loss'].append(test_loss)
            info_dicts['Server']['test_f1'].append(test_f1)
            info_dicts['Server']['test_mcc'].append(test_mcc)

        ########################################
        # save for test
        ########################################
        if test_f1 > test_best_f1:
            # server save
            remove_oldfile(dirname=log_path, file_keyword='-server-best-')
            torch.save(
                server_model.state_dict(), 
                os.path.join(
                    log_path, 
                    SAVE_PTH_NAME+'-server-best-F1-{:.2f}-recall-{:.2f}-prec-{:.2f}-iters-{:d}.pth'.format(
                        test_f1,
                        test_recall,
                        test_precision, 
                        a_iter
                        )
                )
            )
            test_best_f1 = test_f1
            test_best_iter = a_iter
            print_cz(str="Saving new test model to {}".format(
                SAVE_PTH_NAME+'-server-best-F1-{:.2f}-recall-{:.2f}-prec-{:.2f}-iters-{:d}.pth'.format(
                    test_f1, 
                    test_recall,
                    test_precision,  
                    a_iter
                    )
                ), f=logfile)
        # save regular
        # if a_iter % args.save_interval == 9:
        #     torch.save(
        #         server_model.state_dict(), 
        #         os.path.join(
        #             log_path, 
        #             SAVE_PTH_NAME+'-server-iter-{}-F1-{:.2f}-mcc-{:.2f}-recall-{:.2f}-prec-{:.2f}-iters-{:d}.pth'.format(
        #                 a_iter,
        #                 test_f1,
        #                 test_mcc,
        #                 test_recall,
        #                 test_precision, 
        #                 a_iter
        #                 )
        #         )
        #     )

        
        ########################################
        # client test after communication
        ########################################
        client_test_list = []
        for client_idx in range(client_num):
            client_test_loss, client_test_acc, client_test_f1, client_test_auc, \
                client_test_recall, client_test_precision, client_test_kappa, client_test_mcc = test(
                    args,
                    client_models[client_idx], 
                    test_loader, 
                    loss_fun, 
                    # device
                )
            client_test_list.append(
                    [client_test_loss, 
                    client_test_acc, 
                    client_test_f1, 
                    client_test_auc,
                    client_test_recall, 
                    client_test_precision, 
                    client_test_kappa, 
                    client_test_mcc]
                    )
            print_cz(' Client {:<5s}| Local Test  Loss: {:.4f} | Acc: {:.2f}  F1: {:.2f}  AUC: {:.2f}  Recall: {:.2f}  Preci: {:.2f}  Kappa: {:.2f}  MCC: {:.2f}'.format(
                        datasets[client_idx], 
                        client_test_loss, 
                        client_test_acc, 
                        client_test_f1, 
                        client_test_auc,
                        client_test_recall, 
                        client_test_precision, 
                        client_test_kappa, 
                        client_test_mcc
                    ), 
                    f=logfile
                )
            ############ record curve #####################
            # client model after aggregation
            info_dicts[datasets[client_idx]]['test_epochs'].append(wi+a_iter*args.wk_iters)
            info_dicts[datasets[client_idx]]['test_loss'].append(client_test_loss)
            info_dicts[datasets[client_idx]]['test_f1'].append(client_test_f1)
            info_dicts[datasets[client_idx]]['test_mcc'].append(client_test_mcc)
        client_test_macro = np.mean(np.array(client_test_list), axis=0)
        print_cz(' Client {}| Local Test  Loss: {:.4f} | Acc: {:.2f}  F1: {:.2f}  AUC: {:.2f}  Recall: {:.2f}  Preci: {:.2f}  Kappa: {:.2f}  MCC: {:.2f}'.format(
            'Average', 
            client_test_macro[0], 
            client_test_macro[1], 
            client_test_macro[2], 
            client_test_macro[3],
            client_test_macro[4], 
            client_test_macro[5], 
            client_test_macro[6], 
            client_test_macro[7],
            ), 
            f=logfile
        )
        ########################################
        ############### Curves ##################
        ########################################
        curve_save(
            x=info_dicts[datasets[0]]['train_epochs'][len(info_dicts[datasets[0]]['train_epochs'])-len(info_dicts[datasets[0]]['clean_ratio']):], 
            y=[
                info_dicts[datasets[0]]['clean_ratio'], 
                info_dicts[datasets[1]]['clean_ratio'], 
                info_dicts[datasets[2]]['clean_ratio'], 
                info_dicts[datasets[3]]['clean_ratio'], 
                ], 
            tag=['client_A', 'client_B', 'client_C', 'client_D'],
            yaxis='clean ratio', 
            theme='clean ratio acc-all-client', 
            save_dir=log_path
        )
        curve_save(
            x=info_dicts[datasets[0]]['train_epochs'][len(info_dicts[datasets[0]]['train_epochs'])-len(info_dicts[datasets[0]]['pseudo_gold_acc']):], 
            y=[
                info_dicts[datasets[0]]['pseudo_gold_acc'], 
                info_dicts[datasets[1]]['pseudo_gold_acc'], 
                info_dicts[datasets[2]]['pseudo_gold_acc'], 
                info_dicts[datasets[3]]['pseudo_gold_acc'], 
                ], 
            tag=['client_A', 'client_B', 'client_C', 'client_D'],
            yaxis='pseudo gold acc', 
            theme='pseudo gold acc-all-client', 
            save_dir=log_path
        )
        #
        curve_save(
            x=info_dicts[datasets[0]]['train_epochs'], 
            y=[
                info_dicts[datasets[0]]['param_ratio'], 
                info_dicts[datasets[1]]['param_ratio'], 
                info_dicts[datasets[2]]['param_ratio'], 
                info_dicts[datasets[3]]['param_ratio'], 
                ], 
            tag=['client_A', 'client_B', 'client_C', 'client_D'],
            yaxis='param_ratio', 
            theme='Param ratio-all-client', 
            save_dir=log_path
        )
        # norm scalar
        curve_save(
            x=info_dicts_norm[datasets[0]]['norm_epochs'], 
            y=[
                info_dicts_norm['Server']['norm_centroids'], 
                info_dicts_norm[datasets[0]]['norm_centroids'], 
                info_dicts_norm[datasets[1]]['norm_centroids'], 
                info_dicts_norm[datasets[2]]['norm_centroids'], 
                info_dicts_norm[datasets[3]]['norm_centroids'], 
                ], 
            tag=['Server', 'client_A', 'client_B', 'client_C', 'client_D'],
            yaxis='Norm centroids', 
            theme='Norm-centroids-all-client', 
            save_dir=log_path
        )
        curve_save(
            x=info_dicts_norm[datasets[0]]['norm_epochs'], 
            y=[
                args.info_dicts_norm['One']['norm_client_centroids'],
                args.info_dicts_norm['One']['norm_server_centroids'],
                args.info_dicts_norm['One']['norm_gnn_input'],
                ], 
            tag=['client_centroids', 'server_centroids', 'gnn_input'],
            yaxis='Norm', 
            theme='Norm-comparison-all-client', 
            save_dir=log_path
        )
        # lr
        curve_save(
            x=info_dicts[datasets[0]]['train_epochs'], 
            y=[
                info_dicts[datasets[0]]['lr'], 
                info_dicts[datasets[1]]['lr'], 
                info_dicts[datasets[2]]['lr'], 
                info_dicts[datasets[3]]['lr'], 
                ], 
            tag=['client_A', 'client_B', 'client_C', 'client_D'], 
            yaxis='Learning rate', 
            theme='LR-all-client', 
            save_dir=log_path
        )
        curve_save(
            x=info_dicts[datasets[0]]['train_epochs'], 
            y=[
                info_dicts[datasets[0]]['train_mcc'], 
                info_dicts[datasets[1]]['train_mcc'], 
                info_dicts[datasets[2]]['train_mcc'], 
                info_dicts[datasets[3]]['train_mcc'], 
                ], 
            tag=['client_A', 'client_B', 'client_C', 'client_D'], 
            yaxis='Performance', 
            theme='Train-mcc-all-client', 
            save_dir=log_path
        )
        curve_save(
            x=info_dicts[datasets[0]]['train_epochs'], 
            y=[
                info_dicts[datasets[0]]['train_f1'], 
                info_dicts[datasets[1]]['train_f1'], 
                info_dicts[datasets[2]]['train_f1'], 
                info_dicts[datasets[3]]['train_f1'], 
            ], 
            tag=['client_A', 'client_B', 'client_C', 'client_D'], 
            yaxis='Performance', 
            theme='Train-F1-all-client', 
            save_dir=log_path
        )
        curve_save(
            x=info_dicts[datasets[0]]['train_epochs'], 
            y=[
                info_dicts[datasets[0]]['train_loss'], 
                info_dicts[datasets[1]]['train_loss'], 
                info_dicts[datasets[2]]['train_loss'], 
                info_dicts[datasets[3]]['train_loss'], 
            ], 
            tag=['client_A', 'client_B', 'client_C', 'client_D'], 
            yaxis='Loss', 
            theme='Train-Loss-all-client', 
            save_dir=log_path
        )
        # # test curves
        curve_save(
            x=info_dicts[datasets[0]]['test_epochs'], 
            y=[
                info_dicts['Server']['test_mcc'], 
                info_dicts[datasets[0]]['test_mcc'], 
                info_dicts[datasets[1]]['test_mcc'], 
                info_dicts[datasets[2]]['test_mcc'], 
                info_dicts[datasets[3]]['test_mcc'], 
                ],  
            tag=['Server', 'client_A', 'client_B', 'client_C', 'client_D'],
            yaxis='Performance', 
            theme='Test-mcc-all-client', 
            save_dir=log_path
        )
        curve_save(
            x=info_dicts[datasets[0]]['test_epochs'], 
            y=[ 
                info_dicts['Server']['test_f1'],
                info_dicts[datasets[0]]['test_f1'], 
                info_dicts[datasets[1]]['test_f1'], 
                info_dicts[datasets[2]]['test_f1'], 
                info_dicts[datasets[3]]['test_f1'], 
            ], 
            tag=['Server', 'client_A', 'client_B', 'client_C', 'client_D'],
            yaxis='Performance', 
            theme='Test-F1-all-client', 
            save_dir=log_path
        )
        curve_save(
            x=info_dicts[datasets[0]]['test_epochs'], 
            y=[ 
                info_dicts['Server']['test_loss'],
                info_dicts[datasets[0]]['test_loss'], 
                info_dicts[datasets[1]]['test_loss'], 
                info_dicts[datasets[2]]['test_loss'], 
                info_dicts[datasets[3]]['test_loss'], 
            ], 
            tag=['Server', 'client_A', 'client_B', 'client_C', 'client_D'],
            yaxis='Loss', 
            theme='Test-Loss-all-client', 
            save_dir=log_path
        )
        #################
        curve_save(
            x=info_dicts[datasets[0]]['train_epochs'], 
            y=[
                info_dicts[datasets[0]]['server_test_f1'], 
                info_dicts[datasets[1]]['server_test_f1'], 
                info_dicts[datasets[2]]['server_test_f1'], 
                info_dicts[datasets[3]]['server_test_f1'], 
                info_dicts['Server']['test_f1'],
            ], 
            tag=['client_A', 'client_B', 'client_C', 'client_D', 'aggregated'], 
            yaxis='Performance', 
            theme='Test-F1 server centroid-all-client', 
            save_dir=log_path
        )
        print_cz(' Iter time:  {:.1f} min'.format((time.time()-iter_start_time)/60.0), f=logfile)
        ########################################
        # end of one iteration
        ########################################

    ########################################
    # end of FL
    ########################################
    # final save - server model
    torch.save(
        server_model.state_dict(), 
        os.path.join(
            log_path, 
            SAVE_PTH_NAME+'-server-end-F1-{:.2f}-recall-{:.2f}-prec-{:.2f}.pth'.format(
                test_f1, 
                test_recall,
                test_precision,  
                )
        )
    )
    # 
    print_cz('***'*4, f=logfile)
    print_cz(' Total time:  {:.2f} h'.format((time.time()-start_time)/3600.0), f=logfile)
    # summary
    print_cz(' Saving the checkpoint to {}'.format(log_path), f=logfile)
    # print_cz(' valid best iter {}'.format(str(valid_best_iter)), f=logfile)
    print_cz(' test  best iter {}'.format(str(test_best_iter)), f=logfile)
    #        
    logfile.flush()
    logfile.close()

