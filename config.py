import os 
import time 
import torch 
import argparse

from utils import print_cz

data_dir = './data/'
csv_folder = './csv/'

kvasir_data_dir_10class='/home/zchen72/Dataset/Kvasir-Capsule/fl_img_10class/'
kvasir_csv_dir_10class='/home/zchen72/code/noiseFL-v2-public/csv/Kvasir-Capsule-10class/'
noisy_kvasir_csv_dir_10class='/home/zchen72/code/noiseFL-v2-public/csv/Kvasir-Capsule-10class/noisy/'


def get_args():

    parser = argparse.ArgumentParser()

    parser.add_argument('--network', type = str, default='vgg11_nb_small', help = 'classification model')
    # training settings
    parser.add_argument('--optim', type=str, default='sgd')
    parser.add_argument('--lr', type=float, default=1e-2, help='learning rate')
    parser.add_argument("--lr_step", type=int, default=-1, help="5")
    parser.add_argument("--lr_multistep", type=str, default=None, help='0.5_0.75')
    parser.add_argument("--lr_gamma", type=float, default=-1, help='0.1 | 0.5')
    parser.add_argument("--wd", type=float, default=1e-6)
    parser.add_argument('--batch_size', type = int, default= 128, help ='batch size')
    parser.add_argument('--iters', type = int, default=100, help = 'iterations for communication')
    parser.add_argument('--wk_iters', type = int, default=1, help = 'optimization iters in local worker between communication')
    #
    # parser.add_argument('--weight', type = bool, default=True, help='class imbalance weight')
    #    
    parser.add_argument('--seed', type = int, default=1, help = 'seed')
    parser.add_argument('--theme', type = str, default='', help='comments for training')
    parser.add_argument('--save_path', type = str, default='./experiment/', help='path to save the checkpoint')

    ##################
    parser.add_argument('--mode', type=str, default='fedavg', help='fedbn')
    parser.add_argument('--device', type=str, default='cuda', help='cuda | cpu')
    ##################
    parser.add_argument('--noisy_type', type=str, default=None, help='symmetric | pairflip')
    parser.add_argument('--noise_types', type=str, default=None, help='symm_pair_symm_pair')
    parser.add_argument('--noise_rate', type=float, default=0.0)
    parser.add_argument('--noise_rates', type=str, default=None, help='0.2_0.2_0.2_0.2')

    ##################
    parser.add_argument('--resolution', type=int, default=128, help="original resolution is 336")
    parser.add_argument('--pretrained', type=int, default=0)
    ##################
    parser.add_argument('--class_num', type=int, default=10)
    parser.add_argument('--core_ratio', type=float, default=0.2)
    parser.add_argument('--confid_thresh', type=float, default=0.3)
    parser.add_argument('--alpha', type=float, default=0.5, help="label propagation")
    parser.add_argument('--eps', type=float, default=1e-8)
    parser.add_argument('--topK', type=int, default=50, help="sparse graph, including negative loss")

    parser.add_argument('--adj_ratio', type=float, default=0.5, help="server centroid graph adj ratio")
    parser.add_argument('--server_pool', type=str, default='avg', help="server centroid graph pooling: avg/max")

    parser.add_argument('--warm_iter', type=int, default=0)
    parser.add_argument('--param_ratio', type=float, default=0.01, help="local classifier ratio")
    parser.add_argument('--param_ratio_begin', type=float, default=0.5, help="local classifier ratio begin")
    parser.add_argument('--param_ratio_end', type=float, default=0.5, help="local classifier ratio end")
    parser.add_argument('--ema_ratio', type=float, default=0.5, help="")
    parser.add_argument('--centroid_update_ema', type=float, default=0.5, help="centroid update ema")
    parser.add_argument('--norm_regularizer', type=float, default=0.05, help="regularizer to reduce classifier norm")
    parser.add_argument('--centroid_interval', type=int, default=1, help="centroid interval")
    
    parser.add_argument('--cs_interval', type=int, default=5, help="client-server EMA interval")
    parser.add_argument('--save_interval', type=int, default=1000, help="save model interval")
    parser.add_argument('--nl_ratio', type=float, default=0.1, help="negative learning loss ratio")

    args = parser.parse_args()
    return args 

def args_info(args, logfile=None):
    
    print_cz(os.getcwd(), f=logfile)
    print_cz('Device: {}'.format(args.device), f=logfile)
    print_cz('=== {} ==='.format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())), f=logfile)
    print_cz('=== Setting ===', f=logfile)
    print_cz('    network: {}'.format(args.network), f=logfile)
    print_cz('    class_num: {}'.format(args.class_num), f=logfile)
    print_cz('    optim: {}'.format(args.optim), f=logfile)
    print_cz('    lr: {}'.format(args.lr), f=logfile)
    print_cz('    lr_step: {}'.format(args.lr_step), f=logfile)
    print_cz('    lr_multistep: {}'.format(args.lr_multistep), f=logfile)
    print_cz('    lr_gamma: {}'.format(args.lr_gamma), f=logfile)
    print_cz('    mode: {}'.format(args.mode), f=logfile)
    print_cz('    iters: {}'.format(args.iters), f=logfile)
    print_cz('    wk_iters: {}'.format(args.wk_iters), f=logfile)
    print_cz('    batch_size: {}'.format(args.batch_size), f=logfile)
    print_cz('    resolution: {}'.format(args.resolution), f=logfile)
    
    print_cz('=== Noisy Setting ===', f=logfile)
    print_cz('    noisy_type: {}'.format(args.noisy_type), f=logfile)
    print_cz('    noise_types: {}'.format(args.noise_types), f=logfile)
    print_cz('    noise_rate: {}'.format(args.noise_rate), f=logfile)
    print_cz('    noise_rates: {}'.format(args.noise_rates), f=logfile)
    
    print_cz('=== Ours Setting ===', f=logfile)
    print_cz('    core_ratio: {}'.format(args.core_ratio), f=logfile)
    print_cz('    confid_thresh: {}'.format(args.confid_thresh), f=logfile)
    print_cz('    alpha: {}'.format(args.alpha), f=logfile)
    print_cz('    topK: {}'.format(args.topK), f=logfile)
    print_cz('    adj_ratio: {}'.format(args.adj_ratio), f=logfile)
    print_cz('    server_pool: {}'.format(args.server_pool), f=logfile)
    print_cz('    eps: {}'.format(args.eps), f=logfile)
    print_cz('    param_ratio: {}'.format(args.param_ratio), f=logfile)
    print_cz('    param_ratio_begin: {}'.format(args.param_ratio_begin), f=logfile)
    print_cz('    param_ratio_end: {}'.format(args.param_ratio_end), f=logfile)
    print_cz('    ema_ratio: {}'.format(args.ema_ratio), f=logfile)
    print_cz('    centroid_update_ema: {}'.format(args.centroid_update_ema), f=logfile)
    print_cz('    norm_regularizer: {}'.format(args.norm_regularizer), f=logfile)
    print_cz('    centroid_interval: {}'.format(args.centroid_interval), f=logfile)
    print_cz('    warm_iter: {}'.format(args.warm_iter), f=logfile)
    print_cz('    nl_ratio: {}'.format(args.nl_ratio), f=logfile)
    print_cz('    theme: {}'.format(args.theme), f=logfile)

    print_cz('=====================', f=logfile)
 