import torch
import torch.nn as nn
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import torch.nn.functional as func
from collections import OrderedDict


def conv_layer(chann_in, chann_out, k_size, p_size):
    layer = nn.Sequential(
        OrderedDict([
            ('conv', nn.Conv2d(chann_in, chann_out, kernel_size=k_size, padding=p_size, bias=False)),
            ('bn', nn.BatchNorm2d(chann_out)),
            ('relu', nn.ReLU(inplace=True))
        ])
    )
    return layer

def vgg_conv_block(in_list, out_list, k_list, p_list, pooling_k, pooling_s):
    block = [ conv_layer(in_list[i], out_list[i], k_list[i], p_list[i]) for i in range(len(in_list)) ]
    block += [ nn.MaxPool2d(kernel_size = pooling_k, stride = pooling_s)]
    return nn.Sequential(*block)

def vgg_fc_layer(size_in, size_out):
    layer = nn.Sequential(
        OrderedDict([
            ('linear', nn.Linear(size_in, size_out, bias=False)),
            ('bn', nn.BatchNorm1d(size_out)),
            ('relu', nn.ReLU(inplace=True))
        ])
    )
    return layer

class VGG11_Slim2_Centroids_SingleHead(nn.Module):
    def __init__(self, n_classes=4):
        super(VGG11_Slim2_Centroids_SingleHead, self).__init__()
        #
        # Conv blocks (BatchNorm + ReLU activation added in each block)
        self.layer1 = vgg_conv_block([3,], [16,], [3,], [1,], 2, 2)
        self.layer2 = vgg_conv_block([16,], [32,], [3,], [1,], 2, 2)
        self.layer3 = vgg_conv_block([32,64,], [64,64,], [3,3,], [1,1,], 2, 2)
        self.layer4 = vgg_conv_block([64,64,], [64,64,], [3,3,], [1,1,], 2, 2)
        self.layer5 = vgg_conv_block([64,128,], [128,128,], [3,3,], [1,1,], 2, 2)
        #
        self.avgpool = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)))
        # FC layers

        self.fc1 = vgg_fc_layer(128, 64)

        # Final layer
        param_init = nn.Parameter(
            torch.rand(n_classes, 64).data.normal_(0, 0.01)
        )
        self.register_parameter("centroids_param", param_init)
    
    def update_centroids_param(self, new_centroids_param):
        # print("self.centroids_param:\t", self.centroids_param)
        # print("update centroids_param")
        # self.centroids_param = None
        self.centroids_param = new_centroids_param

    def forward(self, x, 
    ):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        vgg16_features = self.avgpool(
            self.layer5(
                out
                )
            )
        feature_view = vgg16_features.view(out.size(0), -1)

        out_fc1 = self.fc1(feature_view)
        # print("out_tmp.device:\t", out_tmp.device)
        # print("centroids_param.device:\t", centroids_param.device)
        out = torch.nn.functional.linear(input=out_fc1, weight=self.centroids_param)
        return out, feature_view, out_fc1

if __name__ == '__main__':
    
    model = VGG11_Slim2_Centroids_SingleHead()

    param_centroids = model.state_dict()["centroids_param"]
    print("centroids:\t", param_centroids.shape)
    print("sum:\t", torch.abs(param_centroids).sum(dim=-1).sum(dim=-1).view(-1))
    
    param_fc = model.state_dict()["fc2.linear.weight"]
    print("fc2:\t", param_fc.shape)
    print("sum:\t", torch.abs(param_fc).sum(dim=-1).sum(dim=-1).view(-1))

    