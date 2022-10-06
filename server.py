import torch 
from torch import nn, optim
import torch.nn.functional as F
import time
import copy
import numpy as np
import torch
from torch import nn, optim
import torch.nn.functional as F
import time
import copy
import numpy as np

from utils import init_dict, save_dict, curve_save, time_mark, print_cz, update_lr


################# Key Function ########################
def communication(
    args, 
    server_model, 
    models,  
    client_weights, 
    logfile=None,
    **kwargs
    ):
    client_num = len(client_weights) # 
    with torch.no_grad():
        # aggregate params
        if 'fedbn' in args.mode.lower():
            print_cz('Server aggregate: # fedbn #, args.mode: {}'.format(args.mode.lower()), f=logfile)
            for key in server_model.state_dict().keys():
                ############################# add at 0504 for centroid model
                if "centroids_param" in key:
                    print_cz("server agg skip {}, kept at local".format(key), f=logfile)
                    continue
                #############################
                if 'bn' not in key: # 排除BN层，其余的做fedavg
                    temp = torch.zeros_like(server_model.state_dict()[key], dtype=torch.float32).to(args.device)
                    for client_idx in range(client_num):
                        temp += client_weights[client_idx] * models[client_idx].state_dict()[key]
                    server_model.state_dict()[key].data.copy_(temp) # 对非bn层，平均后重新赋值
                    for client_idx in range(client_num): # 对非bn层，从server端复制参数给每个client
                        models[client_idx].state_dict()[key].data.copy_(server_model.state_dict()[key])
        #
        elif 'fedavg' in args.mode.lower():
            print_cz('Server aggregate: # fedavg #, args.mode: {}'.format(args.mode.lower()), f=logfile)
            for key in server_model.state_dict().keys():
                # num_batches_tracked is a non trainable LongTensor and
                # num_batches_tracked are the same for all clients for the given datasets
                ############################# add at 0504 for centroid model
                if "centroids_param" in key:
                    print_cz("server agg skip {}, kept at local".format(key), f=logfile)
                    continue
                #############################
                if 'num_batches_tracked' in key:
                    server_model.state_dict()[key].data.copy_(models[0].state_dict()[key])
                else:
                    temp = torch.zeros_like(server_model.state_dict()[key]).to(args.device)
                    for client_idx in range(len(client_weights)):
                        temp += client_weights[client_idx] * models[client_idx].state_dict()[key]
                    server_model.state_dict()[key].data.copy_(temp)
                    for client_idx in range(len(client_weights)):
                        models[client_idx].state_dict()[key].data.copy_(server_model.state_dict()[key])
        #
        else:
            print_cz('Error: invalid args.mode!', f=logfile)
        #
    return server_model, models