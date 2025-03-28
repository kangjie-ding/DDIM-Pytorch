import os
import sys

import torch
import torch.nn as nn
from torchinfo import summary

cur_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(cur_dir, "../"))

from utils.time_embedding import time_embedding
from models.blocks import *

class UNetModel(nn.Module):
    def __init__(self, channel_mul_layer=(1, 2, 4, 8, 16), input_channels=3, output_channels=3, model_channels=64, res_num_layer=2, num_head=4, use_conv=True, dropout=0.1):
        super().__init__()
        self.model_channels = model_channels
        self.time_embedding_dim = 4*model_channels
        self.time_embedding_transform = nn.Sequential(nn.Linear(self.model_channels, self.time_embedding_dim), nn.SiLU())
        # define flags to add Attention Block
        attention_mul = [8, 32]
        attention_flag = 1
        # define input block
        self.input_block = nn.Sequential(nn.Conv2d(input_channels, model_channels, kernel_size=3, padding=1), nn.SiLU())
        cur_channels = self.model_channels

        # define down blocks
        down_block_channels = []
        self.down_blocks = nn.ModuleList([])
        for i, mul in enumerate(channel_mul_layer):
            layer = []
            for j in range(res_num_layer):
                attention_flag *= 2
                layer.append(ResidualBlock(cur_channels, mul*self.model_channels, self.time_embedding_dim, dropout))
                cur_channels = mul*self.model_channels
                if attention_flag in attention_mul:
                    layer.append(AttentionBlock(cur_channels, num_head))
            down_block_channels.append(cur_channels)
            self.down_blocks.append(TimeStepSupportedSequential(*layer))
            self.down_blocks.append(TimeStepSupportedSequential(DownSampleBlock(cur_channels, use_conv=use_conv)))

        # define middle blocks
        self.middle_blocks = nn.ModuleList([])
        layer = []
        for i in range(res_num_layer):
            layer.append(ResidualBlock(cur_channels, cur_channels, self.time_embedding_dim, dropout))
            if i!=res_num_layer-1:
                layer.append(AttentionBlock(cur_channels, num_head))
        self.middle_blocks.append(TimeStepSupportedSequential(*layer))

        # define up blocks
        self.up_blocks = nn.ModuleList([])
        for i, mul in enumerate(channel_mul_layer[::-1]):
            layer = []
            self.up_blocks.append(TimeStepSupportedSequential(UpSampleBlock(cur_channels, use_conv=use_conv)))
            for j in range(res_num_layer):
                attention_flag /= 2
                if j==0:
                    layer.append(ResidualBlock(down_block_channels.pop()+cur_channels, mul*self.model_channels, self.time_embedding_dim, dropout))
                else:
                    layer.append(ResidualBlock(cur_channels, mul*self.model_channels, self.time_embedding_dim, dropout))
                cur_channels = mul*self.model_channels
                if attention_flag in attention_mul:
                    layer.append(AttentionBlock(cur_channels, num_head))
            self.up_blocks.append(TimeStepSupportedSequential(*layer))

        # define ouput block
        self.output_block = nn.Sequential(nn.Conv2d(cur_channels, output_channels, kernel_size=3, padding=1), nn.SiLU())

    def forward(self, x, t):
        time_feature = self.time_embedding_transform(time_embedding(t, self.model_channels))
        cat_features = []
        x = self.input_block(x)
        # down
        for module in self.down_blocks:
            x = module(x, time_feature)
            if not (len(module)==1 and isinstance(module[0], DownSampleBlock)):
                cat_features.append(x)
        # middle
        x = self.middle_blocks[0](x, time_feature)
        # up
        for module in self.up_blocks:
            if len(module)==1 and isinstance(module[0], UpSampleBlock):
                x = module(x, time_feature)
            else:
                x = module(torch.cat((x, cat_features.pop()), dim=1), time_feature)
        return self.output_block(x)

    
if __name__=="__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    unet = UNetModel()
    x = torch.randn((2, 3, 64, 64)).to(device=device)
    t = torch.randint(1, 1000, (2, 1)).to(device=device)
    unet = unet.to(device=device)
    summary(unet, input_size=[x.shape, t.shape])
    output = unet(x, t)
    print(output.shape)
        
                

