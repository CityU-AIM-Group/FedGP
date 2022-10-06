import sys, os
base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(base_path)

import os 
import pandas as pd 
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import cv2
import numpy as np
import albumentations
import time
from utils import print_cz

########### dataset

def get_transforms(image_size):
    
    transforms_train = albumentations.Compose([
        albumentations.Resize(image_size, image_size),
        albumentations.VerticalFlip(p=0.5),
        albumentations.HorizontalFlip(p=0.5),
        # albumentations.RandomBrightness(limit=0.2, p=0.75),
        # albumentations.OneOf([
        #     albumentations.MedianBlur(blur_limit=5),
        #     albumentations.GaussianBlur(blur_limit=5),
        #     albumentations.GaussNoise(var_limit=(5.0, 30.0)),
        # ], p=0.7),
        # albumentations.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, border_mode=0, p=0.85,
        #     ),
        albumentations.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    transforms_test = albumentations.Compose([
        albumentations.Resize(image_size, image_size),
        albumentations.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    return transforms_train, transforms_test


class Kvasir_Dataset(Dataset):
    def __init__(
        self, 
        args,
        data_dir='/home/zchen72/Dataset/Kvasir-Capsule/fl_img_10class/', 
        csv_dir='/home/zchen72/code/noiseFL/csv/Kvasir-Capsule-10class/noisy',
        csv_filename=None, 
        transform=None,
    ):
        super(Kvasir_Dataset, self).__init__()       
        self.img_dir = data_dir # train/valid/test samples are in one folder
        filename = os.path.join(
            csv_dir,
            csv_filename
        )
        self.csv_file = pd.read_csv(
            filepath_or_buffer=filename, 
            sep=','
        )
        #
        self.imgnames = self.csv_file['imgname']
        self.labels = self.csv_file['clean_label']
        self.transform = transform
        start_time = time.time()
        self.image_npy, self.label_npy = self.__pre_load__()
        # print("self.image_npy:\t", self.image_npy.shape)
        print_cz("{} pre load time: {:.2f} min".format(
                filename,
                # csv_filename,
                (time.time()-start_time)/60.0
            ), 
            f=args.logfile
        )

    def __pre_load__(self):

        image_list = []
        label_list = []
        for idx in range(len(self.imgnames)):
            image = cv2.imread(
                os.path.join(self.img_dir, self.imgnames[idx])
            )
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            label = self.labels[idx]
            image_list.append(image)
            label_list.append(label)
        # N*H*W*C
        image_npy = np.stack(image_list, axis=0) 
        # print("image_npy:\t", image_npy.shape)
        # N
        label_npy = np.stack(label_list, axis=0)
        # (4021, 512, 512, 3) (4021,) for train
        # (2139, 512, 512, 3) (2139,) for test
        return image_npy, label_npy

    def __getitem__(self, idx):

        label = self.label_npy[idx]
        # H*W*C
        image = self.image_npy[idx] # (512, 512, 3)
        # print('getitem begin: ', image.shape)
        if self.transform is not None:
            res = self.transform(image=image)
            image = res['image'].astype(np.float32)
        else:
            image = image.astype(np.float32)
        # C*H*W
        image = image.transpose(2, 0, 1) # (3, 512, 512)
        # print('getitem end: ', image.shape)
        
        return image, label, idx, idx
    
    def __len__(self):
        return len(self.labels)



class Noisy_Kvasir_Dataset(Dataset):
    def __init__(
        self, 
        args,
        data_dir='/home/zchen72/Dataset/Kvasir-Capsule/fl_img_10class/', 
        csv_dir='/home/zchen72/code/noiseFL/csv/Kvasir-Capsule-10class/noisy',
        csv_filename=None, 
        transform=None,
    ):

        super(Noisy_Kvasir_Dataset, self).__init__()       
        self.img_dir = data_dir # train/valid/test samples are in one folder

        filename = os.path.join(
            csv_dir,
            csv_filename
        )
        self.csv_file = pd.read_csv(
            filepath_or_buffer=filename, 
            sep=','
        )

        #
        self.imgnames = self.csv_file['imgname']
        self.clean_labels = self.csv_file['clean_label']
        self.noisy_labels = self.csv_file['noisy_label']

        self.transform = transform
        
        start_time = time.time()
        self.image_npy, self.noisy_label_npy, self.clean_label_npy = self.__pre_load__()
        # print("self.image_npy:\t", self.image_npy.shape)
        print_cz("{} pre load time: {:.2f} min".format(
                csv_filename,
                (time.time()-start_time)/60.0
            ), 
            f=args.logfile
        )

    def __pre_load__(self):

        image_list = []
        noisy_label_list = []
        clean_label_list = []
        for idx in range(len(self.imgnames)):
            image = cv2.imread(
                os.path.join(self.img_dir, self.imgnames[idx])
            )
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            clean_label = self.clean_labels[idx]
            noisy_label = self.noisy_labels[idx]
            #
            image_list.append(image)
            clean_label_list.append(clean_label)
            noisy_label_list.append(noisy_label)
        # N*H*W*C
        image_npy = np.stack(image_list, axis=0) 
        # print("image_npy:\t", image_npy.shape)
        # N
        clean_label_npy = np.stack(clean_label_list, axis=0)
        noisy_label_npy = np.stack(noisy_label_list, axis=0)
        # (4021, 512, 512, 3) (4021,) for train
        # (2139, 512, 512, 3) (2139,) for test
        return image_npy, noisy_label_npy, clean_label_npy

    def __getitem__(self, idx):

        noisy_label = self.noisy_label_npy[idx]
        clean_label = self.clean_label_npy[idx]
        # H*W*C
        image = self.image_npy[idx] # (512, 512, 3)
        # print('getitem begin: ', image.shape)
        if self.transform is not None:
            res = self.transform(image=image)
            image = res['image'].astype(np.float32)
        else:
            image = image.astype(np.float32)
        # C*H*W
        image = image.transpose(2, 0, 1) # (3, 512, 512)
        # print('getitem end: ', image.shape)
        
        return image, noisy_label, clean_label, idx, idx
    
    def __len__(self):
        return len(self.clean_labels)


############### applied #################
def prepare_data_kvasir_4clients_noisy(
    args,
    data_dir='/home/zchen72/Dataset/Kvasir-Capsule/fl_img_10class/', 
    csv_dir='/home/zchen72/code/noiseFL/csv/Kvasir-Capsule-10class/noisy/',
    noisy_type=None,
    noise_rate=None, # int or list
    ):
    """
    return train, valid and test dataloader list for 4 clients 
    intensity to [0,1]
    """
    # Prepare data
    transform_train, transform_test = get_transforms(
        args.resolution
    )

    if isinstance(noise_rate, float):
        csv_filename_A = 'noisy-train-4clients-A-shuffle-{}-{}.csv'.format(noisy_type[:4], noise_rate)
        csv_filename_B = 'noisy-train-4clients-B-shuffle-{}-{}.csv'.format(noisy_type[:4], noise_rate)
        csv_filename_C = 'noisy-train-4clients-C-shuffle-{}-{}.csv'.format(noisy_type[:4], noise_rate)
        csv_filename_D = 'noisy-train-4clients-D-shuffle-{}-{}.csv'.format(noisy_type[:4], noise_rate)
    elif isinstance(noise_rate, list) or isinstance(noise_rate, tuple):
        csv_filename_A = 'noisy-train-4clients-A-shuffle-{}-{}.csv'.format(noisy_type[:4], noise_rate[0])
        csv_filename_B = 'noisy-train-4clients-B-shuffle-{}-{}.csv'.format(noisy_type[:4], noise_rate[1])
        csv_filename_C = 'noisy-train-4clients-C-shuffle-{}-{}.csv'.format(noisy_type[:4], noise_rate[2])
        csv_filename_D = 'noisy-train-4clients-D-shuffle-{}-{}.csv'.format(noisy_type[:4], noise_rate[3])

    ## dataset
    train_set_A = Noisy_Kvasir_Dataset(
        args=args,
        data_dir=data_dir, 
        csv_dir=csv_dir,
        csv_filename=csv_filename_A,
        transform=transform_train
    )
    train_set_B = Noisy_Kvasir_Dataset(
        args=args,
        data_dir=data_dir, 
        csv_dir=csv_dir,
        csv_filename=csv_filename_B,
        transform=transform_train
    )
    train_set_C = Noisy_Kvasir_Dataset(
        args=args,
        data_dir=data_dir, 
        csv_dir=csv_dir,
        csv_filename=csv_filename_C, 
        transform=transform_train
    )
    train_set_D = Noisy_Kvasir_Dataset(
        args=args,
        data_dir=data_dir, 
        csv_dir=csv_dir,
        csv_filename=csv_filename_D, 
        transform=transform_train
    )

    valid_set = Kvasir_Dataset(
        args=args,
        data_dir=data_dir, 
        csv_dir=csv_dir,
        csv_filename='valid-shuffle.csv', 
        transform=transform_test
    )
    test_set = Kvasir_Dataset(
        args=args,
        data_dir=data_dir, 
        csv_dir=csv_dir,
        csv_filename='test-shuffle.csv',
        transform=transform_test
    )


    ## dataloader
    train_loader_A = torch.utils.data.DataLoader(
        train_set_A, 
        batch_size=args.batch_size, 
        shuffle=True
    )
    train_loader_B = torch.utils.data.DataLoader(
        train_set_B, 
        batch_size=args.batch_size, 
        shuffle=True
    )
    train_loader_C = torch.utils.data.DataLoader(
        train_set_C, 
        batch_size=args.batch_size, 
        shuffle=True
    )
    train_loader_D = torch.utils.data.DataLoader(
        train_set_D, 
        batch_size=args.batch_size, 
        shuffle=True
    )

    # print("train_set_A.image_npy:\t", train_set_A.image_npy.shape)
    train_loaders = [train_loader_A, train_loader_B, train_loader_C, train_loader_D]
    train_image_npy_list = [train_set_A.image_npy, train_set_B.image_npy, train_set_C.image_npy, train_set_D.image_npy]
    train_noisy_label_npy_list = [train_set_A.noisy_label_npy, train_set_B.noisy_label_npy, train_set_C.noisy_label_npy, train_set_D.noisy_label_npy]
    train_clean_label_npy_list = [train_set_A.clean_label_npy, train_set_B.clean_label_npy, train_set_C.clean_label_npy, train_set_D.clean_label_npy]

    valid_loader = torch.utils.data.DataLoader(
        valid_set, 
        batch_size=args.batch_size, 
        shuffle=False
    )
    test_loader = torch.utils.data.DataLoader(
        test_set, 
        batch_size=args.batch_size, 
        shuffle=False
    )

    return (train_loaders, valid_loader, test_loader), \
        (train_image_npy_list, train_noisy_label_npy_list, train_clean_label_npy_list), \
        (valid_set.image_npy, valid_set.label_npy), \
        (test_set.image_npy, test_set.label_npy)



def prepare_data_kvasir_4clients_noisy_heter(
    args,
    data_dir='/home/zchen72/Dataset/Kvasir-Capsule/fl_img_10class/', 
    csv_dir='/home/zchen72/code/noiseFL/csv/Kvasir-Capsule-10class/noisy/',
    noise_type_list=None,
    noise_rate_list=None, # int or list
    ):
    """
    return train, valid and test dataloader list for 4 clients 
    intensity to [0,1]
    """
    # Prepare data
    transform_train, transform_test = get_transforms(
        args.resolution
    )

    csv_filename_A = 'noisy-train-4clients-A-shuffle-{}-{}.csv'.format(noise_type_list[0], noise_rate_list[0])
    csv_filename_B = 'noisy-train-4clients-B-shuffle-{}-{}.csv'.format(noise_type_list[1], noise_rate_list[1])
    csv_filename_C = 'noisy-train-4clients-C-shuffle-{}-{}.csv'.format(noise_type_list[2], noise_rate_list[2])
    csv_filename_D = 'noisy-train-4clients-D-shuffle-{}-{}.csv'.format(noise_type_list[3], noise_rate_list[3])

    ## dataset
    train_set_A = Noisy_Kvasir_Dataset(
        args=args,
        data_dir=data_dir, 
        csv_dir=csv_dir,
        csv_filename=csv_filename_A,
        transform=transform_train
    )
    train_set_B = Noisy_Kvasir_Dataset(
        args=args,
        data_dir=data_dir, 
        csv_dir=csv_dir,
        csv_filename=csv_filename_B,
        transform=transform_train
    )
    train_set_C = Noisy_Kvasir_Dataset(
        args=args,
        data_dir=data_dir, 
        csv_dir=csv_dir,
        csv_filename=csv_filename_C, 
        transform=transform_train
    )
    train_set_D = Noisy_Kvasir_Dataset(
        args=args,
        data_dir=data_dir, 
        csv_dir=csv_dir,
        csv_filename=csv_filename_D, 
        transform=transform_train
    )

    valid_set = Kvasir_Dataset(
        args=args,
        data_dir=data_dir, 
        csv_dir=csv_dir,
        csv_filename='valid-shuffle.csv', 
        transform=transform_test
    )
    test_set = Kvasir_Dataset(
        args=args,
        data_dir=data_dir, 
        csv_dir=csv_dir,
        csv_filename='test-shuffle.csv',
        transform=transform_test
    )


    ## dataloader
    train_loader_A = torch.utils.data.DataLoader(
        train_set_A, 
        batch_size=args.batch_size, 
        shuffle=True
    )
    train_loader_B = torch.utils.data.DataLoader(
        train_set_B, 
        batch_size=args.batch_size, 
        shuffle=True
    )
    train_loader_C = torch.utils.data.DataLoader(
        train_set_C, 
        batch_size=args.batch_size, 
        shuffle=True
    )
    train_loader_D = torch.utils.data.DataLoader(
        train_set_D, 
        batch_size=args.batch_size, 
        shuffle=True
    )

    # print("train_set_A.image_npy:\t", train_set_A.image_npy.shape)
    train_loaders = [train_loader_A, train_loader_B, train_loader_C, train_loader_D]
    train_image_npy_list = [train_set_A.image_npy, train_set_B.image_npy, train_set_C.image_npy, train_set_D.image_npy]
    train_noisy_label_npy_list = [train_set_A.noisy_label_npy, train_set_B.noisy_label_npy, train_set_C.noisy_label_npy, train_set_D.noisy_label_npy]
    train_clean_label_npy_list = [train_set_A.clean_label_npy, train_set_B.clean_label_npy, train_set_C.clean_label_npy, train_set_D.clean_label_npy]

    valid_loader = torch.utils.data.DataLoader(
        valid_set, 
        batch_size=args.batch_size, 
        shuffle=False
    )
    test_loader = torch.utils.data.DataLoader(
        test_set, 
        batch_size=args.batch_size, 
        shuffle=False
    )

    return (train_loaders, valid_loader, test_loader), \
        (train_image_npy_list, train_noisy_label_npy_list, train_clean_label_npy_list), \
        (valid_set.image_npy, valid_set.label_npy), \
        (test_set.image_npy, test_set.label_npy)



#############################################################################
def offline_load_npy_kvasir(
    args,
    data_dir='/home/zchen72/Dataset/Kvasir-Capsule/fl_img_10class/', 
    csv_dir='/home/zchen72/code/noiseFL/csv/Kvasir-Capsule-10class/',
    csv_filename=None,  
):
    img_dir = data_dir
    csv_file = pd.read_csv(
        filepath_or_buffer=os.path.join(
            csv_dir,
            csv_filename
        ), 
        sep=','
    )
    imgnames = csv_file['imgname']
    labels = csv_file['clean_label']
    start_time = time.time()
    image_list = []
    label_list = []
    for idx in range(len(imgnames)):
        image = cv2.imread(
            os.path.join(img_dir, imgnames[idx])
        )
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        label = labels[idx]
        image_list.append(image)
        label_list.append(label)
    # N*H*W*C
    # (4021, 512, 512, 3) (4021,) for train
    # (2139, 512, 512, 3) (2139,) for test
    image_npy = np.stack(image_list, axis=0) 
    print("image_npy:\t", image_npy.shape)
    label_npy = np.stack(label_list, axis=0)
    print_cz("{} pre load time: {:.2f} min".format(
            csv_filename,
            (time.time()-start_time)/60.0
        ), 
        f=args.logfile
    )
    return image_npy, label_npy

def offline_load_npy_kvasir_4clients(
    args,
    data_dir='/home/zchen72/Dataset/Kvasir-Capsule/fl_img_10class/', 
    csv_dir='/home/zchen72/code/noiseFL/csv/Kvasir-Capsule-10class/',
    ):
    """
    实际使用
    """
    # Prepare data
    train_A_image_npy, train_A_label_npy = offline_load_npy_kvasir(
        args=args,
        data_dir=data_dir, 
        csv_dir=csv_dir,
        csv_filename='train-4clients-A-shuffle.csv',
    )
    train_B_image_npy, train_B_label_npy = offline_load_npy_kvasir(
        args=args,
        data_dir=data_dir, 
        csv_dir=csv_dir,
        csv_filename='train-4clients-B-shuffle.csv',
    )
    train_C_image_npy, train_C_label_npy = offline_load_npy_kvasir(
        args=args,
        data_dir=data_dir, 
        csv_dir=csv_dir,
        csv_filename='train-4clients-C-shuffle.csv',
    )
    train_D_image_npy, train_D_label_npy = offline_load_npy_kvasir(
        args=args,
        data_dir=data_dir, 
        csv_dir=csv_dir,
        csv_filename='train-4clients-D-shuffle.csv',
    )
    train_image_npy_list = [train_A_image_npy, train_B_image_npy, train_C_image_npy, train_D_image_npy]
    train_label_npy_list = [train_A_label_npy, train_B_label_npy, train_C_label_npy, train_D_label_npy]

    valid_image_npy, valid_label_npy = offline_load_npy_kvasir(
        args=args,
        data_dir=data_dir, 
        csv_dir=csv_dir,
        csv_filename='valid-shuffle.csv',
    )
    test_image_npy, test_label_npy = offline_load_npy_kvasir(
        args=args,
        data_dir=data_dir, 
        csv_dir=csv_dir,
        csv_filename='test-shuffle.csv',
    )
    return (train_image_npy_list, train_label_npy_list), \
        (valid_image_npy, valid_label_npy), \
            (test_image_npy, test_label_npy)
#############################################################################

############################################

class KvasirDataset_Purified(Dataset):
    def __init__(
        self, 
        args,
        widx_purified,
        image_npy, 
        label_npy,
        transform=None
    ):
        super(KvasirDataset_Purified, self).__init__()     
        self.transform = transform   
        # (4021, 512, 512, 3) (4021,) for train
        # (2139, 512, 512, 3) (2139,) for test
        self.widx_purified = widx_purified
        
        if self.widx_purified is not None:
            self.image_npy_purified = image_npy[widx_purified]
            self.label_npy_purified = label_npy[widx_purified]
            # 注意x[None]会使得x加一维度，需保证widx_purified有效
        else:
            print_cz("Error: widx_purified is None, select all samples as clean", f=args.logfile)
            self.image_npy_purified = image_npy
            self.label_npy_purified = label_npy
        # print("len(image_npy[widx_purified]):\t", len(widx_purified), type(widx_purified)) # 232
        # print(widx_purified)
        # print("image_npy[widx_purified] shape:\t", image_npy[widx_purified].shape) #(232, 512, 512, 3)
        # print("len(self.label_npy_purified):\t", len(self.label_npy_purified)) # 232

    def __getitem__(self, idx):
        label = self.label_npy_purified[idx]
        # H*W*C
        image = self.image_npy_purified[idx] # (512, 512, 3)
        if self.transform is not None:
            # print("self.image_npy_purified, shape", self.image_npy_purified.shape)
            # print("idx:\t", idx)
            # print("image shape:\t", image.shape)
            res = self.transform(image=image)
            image = res['image'].astype(np.float32)
        else:
            image = image.astype(np.float32)
        # C*H*W
        image = image.transpose(2, 0, 1) # (3, 512, 512)  
        # if self.widx_purified is not None:
        #     print("idx:\t", type(idx), idx)
        #     print("self.widx_purified[idx]:\t", type(self.widx_purified[idx]), self.widx_purified[idx])      
        return image, label, idx, self.widx_purified[idx]
    
    def __len__(self):
        # 控制新的clean样本总数
        return len(self.label_npy_purified)

def single_purified_trainloader(
    args,
    widx_purified,
    image_npy, 
    label_npy,
    state='train', 
    ):
    # Prepare data
    transform_train, transform_test = get_transforms(
        args.resolution
    )
    
    ## dataloader
    if state.lower() == 'train':
        # default
        dataset = KvasirDataset_Purified(
            args=args,
            widx_purified=widx_purified,
            image_npy=image_npy, 
            label_npy=label_npy, 
            transform=transform_train
        )
        loader = torch.utils.data.DataLoader(
            dataset, 
            batch_size=args.batch_size, 
            shuffle=True
        )
    elif state.lower() == 'test':
        dataset = KvasirDataset_Purified(
            args=args,
            widx_purified=widx_purified,
            image_npy=image_npy, 
            label_npy=label_npy, 
            transform=transform_test
        )
        loader = torch.utils.data.DataLoader(
            dataset, 
            batch_size=args.batch_size, 
            shuffle=False
        )
    else:
        print_cz("Error dataloader: state-{}".format(state.lower()), f=args.logfile)
        loader = None
    return loader


############################################

class KvasirDataset_Inference(Dataset):
    def __init__(
        self, 
        args,
        image_npy, 
        label_npy,
        transform=None
    ):
        super(KvasirDataset_Inference, self).__init__()     
        self.transform = transform   
        # (4021, 512, 512, 3) (4021,) for train
        # (2139, 512, 512, 3) (2139,) for test
        self.image_npy = image_npy
        self.label_npy = label_npy   

    def __getitem__(self, idx):
        label = self.label_npy[idx]
        # H*W*C
        image = self.image_npy[idx] # (512, 512, 3)
        if self.transform is not None:
            res = self.transform(image=image)
            image = res['image'].astype(np.float32)
        else:
            image = image.astype(np.float32)
        # C*H*W
        image = image.transpose(2, 0, 1) # (3, 512, 512)        
        return image, label, idx, idx
    
    def __len__(self):
        # 控制新的clean样本总数
        return len(self.label_npy)

def single_inference_trainloader(
    args,
    image_npy, 
    label_npy,
    state='test', 
    ):
    # Prepare data
    transform_train, transform_test = get_transforms(
        args.resolution
    )
    if state.lower() == 'test':
        ## dataset
        dataset = KvasirDataset_Inference(
            args=args,
            image_npy=image_npy, 
            label_npy=label_npy, 
            transform=transform_test
        )
        ## dataloader
        loader = torch.utils.data.DataLoader(
            dataset, 
            batch_size=args.batch_size, 
            shuffle=False
        )
    elif state.lower() == 'train':
        ## dataset
        dataset = KvasirDataset_Inference(
            args=args,
            image_npy=image_npy, 
            label_npy=label_npy, 
            transform=transform_train
        )
        ## dataloader
        loader = torch.utils.data.DataLoader(
            dataset, 
            batch_size=args.batch_size, 
            shuffle=True
        )
    else:
        print_cz("Error dataloader: state-{}".format(state.lower()), f=args.logfile)
    return loader



###########

if __name__ == "__main__":

    import config

    args = config.get_args()
    logfile = open(os.path.join("./",'log.txt'), 'a')
    args.logfile = logfile

    (train_loaders, valid_loader, test_loader), \
        (train_image_npy_list, train_noisy_label_npy_list, train_clean_label_npy_list), \
            (valid_image_npy, valid_label_npy), \
                (test_image_npy, test_label_npy) = prepare_data_kvasir_4clients_noisy(
                    args,
                    data_dir=config.kvasir_data_dir,
                    csv_dir=config.noisy_kvasir_csv_dir,
                    noisy_type='symmetric',
                    noise_rate=0.2
                ) 
    
    print_cz('---'*3, f=args.logfile)
    print_cz('npy info', f=args.logfile)
    for (train_image_npy, train_noisy_label_npy, train_clean_label_npy) in zip(train_image_npy_list, train_noisy_label_npy_list, train_clean_label_npy_list):
        print(train_image_npy.shape, train_noisy_label_npy.shape, train_clean_label_npy.shape)
    print_cz('---'*3, f=args.logfile)
    print(valid_image_npy.shape, valid_label_npy.shape)
    print_cz('---'*3, f=args.logfile)
    print(test_image_npy.shape, test_label_npy.shape)
    print_cz('---'*3, f=args.logfile)

    # for i, (data, label, idxs) in enumerate(train_loader):
    #     print(i)
    #     print(data.shape, label.shape, idxs.shape)
    #     print(label)
    #     print(idxs)
    #     if i>3:
    #         break

    for i, (data, noisy_label, clean_label, idxs, _) in enumerate(train_loaders[0]):
        print_cz(i, f=args.logfile)
        print(data.shape, noisy_label.shape, clean_label.shape, idxs.shape)
        print_cz(noisy_label, f=args.logfile)
        print_cz(clean_label, f=args.logfile)
        print_cz(idxs, f=args.logfile)
        if i>3:
            break
    
    # 
    logfile.close()