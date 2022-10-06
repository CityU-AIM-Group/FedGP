from pickletools import optimize
import torch
from torch import nn, optim
import torch.nn.functional as F
import time
import copy
import math
import numpy as np
from torch.autograd import Variable
from sklearn import metrics
from sklearn.metrics import roc_auc_score
# from local_noise_cleaning import adjacency_c2w # 

from utils import print_cz

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

def compute_centroids_graph(
    args,
    x,
    # adj_ratio,
):
    """
    x: C*K, D
    """
    graph_similarity_list = []
    for cent_idx in range(x.shape[0]):
        similarity = compute_similarity_ova(
            feature_k=x[cent_idx],
            feature_all=x
        )
        graph_similarity_list.append(similarity)
    #
    graph_similarity_tensor = torch.stack(graph_similarity_list, dim=0)
    thresholds, _ = torch.topk(
        graph_similarity_tensor.view(-1), 
        k=int(graph_similarity_tensor.shape[0] * graph_similarity_tensor.shape[1] * args.adj_ratio),
        largest=True, 
        sorted=True
    )
    # 取稀疏化
    adjacency_matrix = torch.where(
        graph_similarity_tensor>thresholds[-1]-args.eps,
        graph_similarity_tensor,
        torch.zeros(graph_similarity_tensor.shape).to(args.device)
        # 0.0
    )
    return adjacency_matrix

def A_normalize(
    matrix, 
    # self_loop, 
    self_loop_flag=True):
    # normalization for each adjacency matrix [K, K]
    matrix = F.relu(matrix, inplace=False)
    if self_loop_flag:
        self_loop = torch.eye(matrix.shape[0]).to(matrix.device)
        matrix = matrix + self_loop # self.diagonal_i
    with torch.no_grad():
        degree = torch.diag(torch.pow(torch.sum(matrix, 1), -0.5))

    return torch.mm(degree, torch.mm(matrix, degree))

class GCLayer_Centroids(nn.Module):
    def __init__(
        self, 
        in_channel, 
        out_channel,  
        res_connect=True
        ):
        super(GCLayer_Centroids, self).__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.weight = nn.Parameter(torch.FloatTensor(in_channel, out_channel), requires_grad=True)
        self.res_connect = res_connect
        # if res_connct is true, in_channel should be equal to out_channel
        if self.res_connect:
            assert self.in_channel == self.out_channel, "res_connect with unmatched channels: {:d}/{:d}".format(
                self.in_channel, 
                self.out_channel
            )   
        self.reset_parameters()
        
    def reset_parameters(self):
        self.weight.data.normal_(0, 0.01)

    def forward(self, x, adjacency_matrix):
        # x [C*K D]
        # adjacency_matrix [C*K, C*K]
        # AXW
        output = torch.mm(
                    adjacency_matrix,
                    torch.mm(x, self.weight)
                )
        # output = torch.mm(
        #             self.weight, 
        #             torch.mm(x, adjacency_matrix)
        #         )
        if self.res_connect:
            # 跳接
            output = output + x
        output = F.relu(output, inplace=True)

        return output



class GCLayer_Centroids_noReLU(nn.Module):
    def __init__(
        self, 
        in_channel, 
        out_channel,  
        res_connect=True
        ):
        super(GCLayer_Centroids_noReLU, self).__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.weight = nn.Parameter(torch.FloatTensor(in_channel, out_channel), requires_grad=True)
        self.res_connect = res_connect
        # if res_connct is true, in_channel should be equal to out_channel
        if self.res_connect:
            assert self.in_channel == self.out_channel, "res_connect with unmatched channels: {:d}/{:d}".format(
                self.in_channel, 
                self.out_channel
            )   
        self.reset_parameters()
        
    def reset_parameters(self):
        self.weight.data.normal_(0, 0.01)

    def forward(self, x, adjacency_matrix):
        # x [C*K D]
        # adjacency_matrix [C*K, C*K]
        # AXW
        output = torch.mm(
                    adjacency_matrix,
                    torch.mm(x, self.weight)
                )
        # output = torch.mm(
        #             self.weight, 
        #             torch.mm(x, adjacency_matrix)
        #         )
        if self.res_connect:
            # 跳接
            output = output + x
        # output = F.relu(output, inplace=True)

        return output


def graph_class_pooling(
    args,
    x,
    # centroids_class_list,
):
    """
    x: [KC, D], 每个client的C类centroids放在一起，再将不同client的合起来; KC: 先K维后C维
    y: [C, D]
    args.server_pool: avg/max
    """
    pooled_list = []
    if args.server_pool.lower() == 'avg':
        for class_idx in range(args.class_num):
            # 全0初始化
            pool_mask = torch.zeros(x.shape[0]).to(args.device)
            for client_idx in range(args.client_num):
                pool_mask[class_idx + client_idx*args.class_num] = 1
            #
            pooled_centroid = torch.sum(
                x*pool_mask.view(-1, 1),
                dim=0,
                keepdim=False
            )/torch.sum(pool_mask, dim=0, keepdim=False)
            pooled_list.append(pooled_centroid)
        #
        y = torch.stack(pooled_list, dim=0)

    elif args.server_pool.lower() == 'max':
        for class_idx in range(args.class_num):
            # 全0初始化
            pool_mask = torch.zeros(x.shape[0]).to(args.device)
            for client_idx in range(args.client_num):
                pool_mask[class_idx + client_idx*args.class_num] = 1
            #
            pooled_centroid, _ = torch.max(
                x*pool_mask.view(-1, 1),
                dim=0
            )
            pooled_list.append(pooled_centroid)
        #
        y = torch.stack(pooled_list, dim=0)
    else:
        print_cz("invalid type of args.server_pool: {}".format(
            args.server_pool
            ), f=args.logfile)

    return y 



class Server_Centroids_GNN(nn.Module):
    """
    add layer norm
    decouple relu
    """
    def __init__(
        self,
        args, 
        in_channel,
        inter_channel,
        out_channel,  
        res_connect=True
        ):
        super(Server_Centroids_GNN, self).__init__()
        self.norm_input = torch.nn.LayerNorm(in_channel)
        self.gnn_layer1 = GCLayer_Centroids_noReLU(
            in_channel=in_channel, 
            out_channel=inter_channel,
            res_connect=res_connect
            )
        self.norm_layer1 = torch.nn.LayerNorm(inter_channel)
        self.gnn_layer2 = GCLayer_Centroids_noReLU(
            in_channel=inter_channel, 
            out_channel=out_channel,
            res_connect=res_connect
            )
        self.args = args
    
    def forward(
        self,
        x 
        ):
        """
        class-based pooling after layer2
        x: KC*D
        """
        x = self.norm_input(x) # input: layer norm
        adjacency_matrix1 = A_normalize(
                matrix=compute_centroids_graph(
                    args=self.args,
                    x=x
                ), 
                self_loop_flag=True
            )
        layer1_output = F.relu(
            self.norm_layer1(
                self.gnn_layer1(
                    x=x, 
                    adjacency_matrix=adjacency_matrix1
                )
            )
        )
        #
        adjacency_matrix2 = A_normalize(
                matrix=compute_centroids_graph(
                    args=self.args,
                    x=layer1_output
                ), 
                self_loop_flag=True
            )
        layer2_output = self.gnn_layer2(
            x=layer1_output, 
            adjacency_matrix=adjacency_matrix2
        )
        # 
        pooled_layer2_output = graph_class_pooling(
            args=self.args,
            x=layer2_output,
        )

        return pooled_layer2_output

"""
norm_input.weight
norm_input.bias
gnn_layer1.weight
norm_layer1.weight
norm_layer1.bias
gnn_layer2.weight
"""

if __name__ == "__main__":

    import config
    args = config.get_args()
    args.logfile = None
    args.client_num = 2

    # gclayer = GCLayer_Centroids(in_channel=64, out_channel=64, res_connect=True)
    # print(gclayer.parameters())

    model = Server_Centroids_GNN_v2(
        args=args,
        in_channel=64, 
        inter_channel=64,
        out_channel=64, 
        res_connect=True
    )

     
    for param in model.parameters():
        param_begin = param.clone().detach()
        print(param.shape)

    optimizer = torch.optim.Adam(
        params=model.parameters(), lr=1e-3
    )

    # print(dir(optimizer))
    # print(len(optimizer.param_groups))
    # print(optimizer.param_groups[0])
    # print(type(optimizer.param_groups[0]['params']))
    # print(optimizer.param_groups[0]['params'].shape)


    # input_tensor = torch.rand(4, 64) 
    input_tensor = torch.rand(8, 64) 
    print("input_tensor:\t", input_tensor.shape)
    input_tensor.requires_grad_()
    # input_tensor.grad = torch.zeros(input_tensor.shape)

    # adj = compute_centroids_graph(
    #     args=args,
    #     x=input_tensor,
    # )
    # print("adj:\t", adj.shape)

    output_tensor = model(input_tensor)
    # output_tensor = gclayer(input_tensor, adj)
    print("output_tensor:\t", output_tensor.shape)
    output_tensor.grad = torch.rand(output_tensor.shape)
    print("output_tensor grad:\t", output_tensor.grad.shape)

    output_tensor.backward(torch.ones(output_tensor.shape))
    print("input_tensor grad:\t", input_tensor.grad.shape)    

    optimizer.step()

    for param in model.parameters():
        print("param:\t", type(param), param.grad.shape)
        print(param.grad)
        param_end = param.clone().detach()
    
    param_diff = torch.sum(
        torch.abs(param_end - param_begin).view(-1), 
        dim=-1, 
        keepdim=False
    )
    print("param_diff:\t", param_diff)
