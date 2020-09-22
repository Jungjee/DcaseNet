import pdb

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.model_utilities import ConvBlock, init_gru, init_layer, interpolate

def get_DcaseNet_v3(**args):
    return DcaseNet_v3(block = SEBasicBlock, **args)

class DcaseNet_v3(nn.Module):
    def __init__(self,
        block,
        filts_ASC = 256,
        blocks_ASC = 3,
        strides_ASC = 2,
        code_ASC = 128,
        pool_type='avg',
        pool_size=(2,2)):
        
        super().__init__()

        #####
        # Common
        #####
        self.pool_type = pool_type
        self.pool_size = pool_size
        
        self.conv_block1 = ConvBlock(in_channels=1, out_channels=64)
        self.conv_block2 = ConvBlock(in_channels=64, out_channels=128)
        self.conv_block3 = ConvBlock(in_channels=128, out_channels=256)
        self.conv_block4_1 = ConvBlock(in_channels=256, out_channels=256)
        self.conv_block4_2 = ConvBlock(in_channels=256, out_channels=256)

        self.gru_1 = nn.GRU(input_size=512, hidden_size=128, 
            num_layers=1, batch_first=True, bidirectional=True)
        self.gru_2 = nn.GRU(input_size=512, hidden_size=128, 
            num_layers=1, batch_first=True, bidirectional=True)
        self.event_fc = nn.Linear(256, 14, bias=True)

        #####
        # ASC
        #####
        self.inplane = 256
        self.layer_ASC = self._make_layer(
            block = block,
            planes = 384,
            blocks = blocks_ASC,
            stride=strides_ASC,
            reduction=16
        )
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.code_ASC = nn.Linear(384, code_ASC)
        self.fc_ASC = nn.Linear(code_ASC, 10)

        #####
        # TAG
        #####
        self.layer_TAG = nn.Sequential(
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(1024, 80)
        )

        #####
        # Initialize
        #####
        self.init_weights()

    def init_weights(self):

        init_gru(self.gru_1)
        init_gru(self.gru_2)
        init_layer(self.event_fc)

    def _make_layer(self, block, planes, blocks, stride, reduction):
        strides = [stride] + [1] * (blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.inplane, planes, stride, reduction))
            self.inplane = planes

        return nn.Sequential(*layers)

    def forward(self, x, mode = ''):
        d_return = {}   # dictionary to return

        #x: (#bs, #ch, #mel, #seq)
        #forward frame-level
        x = self.conv_block1(x, self.pool_type, pool_size=self.pool_size)
        x = self.conv_block2(x, self.pool_type, pool_size=self.pool_size)
        x = self.conv_block3(x, self.pool_type, pool_size=self.pool_size)
        x_1 = self.conv_block4_1(x, self.pool_type, pool_size=(2, 5))   #common branch
        x_2 = self.conv_block4_2(x, self.pool_type, pool_size=(2, 5))   #task specific branch
        #x: (#bs, #filt, #mel, #seq)

        if 'ASC' in mode:
            out_ASC = x_2
            out_ASC = self.layer_ASC(out_ASC)
            out_ASC = self.avgpool(out_ASC).view(x.size(0), -1)
            #x: (#bs, #filt)    
            out_ASC = self.code_ASC(out_ASC)
            #x: (#bs, #filt)  
            out_ASC = self.fc_ASC(out_ASC)
            d_return['ASC'] = out_ASC
        x = torch.cat([x_1, x_2], dim=1)    #x: (#bs, #filt, #mel, #seq)
        if 'SED' not in mode and 'TAG' not in mode: return d_return

        if self.pool_type == 'avg':
            x = torch.mean(x, dim=2)
        elif self.pool_type == 'max':
            (x, _) = torch.max(x, dim=2)
        #x: (#bs, #filt,#seq)
        x = x.transpose(1,2)
        #x: (#bs, #seq, #filt)
        (x_1, _) = self.gru_1(x)
        (x_2, _) = self.gru_2(x)
        

        if 'SED' in mode:
            out_SED = x_2
            #out_SED = self.layer_SED(x)
            out_SED = torch.sigmoid(self.event_fc(out_SED))
            out_SED = out_SED.repeat_interleave(repeats=8, dim=1)
            d_return['SED'] = out_SED
        x = torch.cat([x_1, x_2], dim=-1)    #x: (#bs, #seq, #filt)
        if 'TAG' not in mode: return d_return
        
        if 'TAG' in mode:
            out_TAG = torch.mean(x, dim=1)
            out_TAG = self.layer_TAG(out_TAG)
            d_return['TAG'] = out_TAG
        return d_return

#####
# ASC
#####
def conv3x3(in_planes, out_planes, stride=1):
    #changed from 1d to 2d
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class SEBasicBlock(nn.Module):
    def __init__(self, inplanes, planes, stride=1, reduction=16):
        super(SEBasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.se = SELayer(planes, reduction)
        if inplanes != planes:
            self.downsample = nn.Sequential(nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride, bias=False),
                                            nn.BatchNorm2d(planes))
        else:
            self.downsample = lambda x: x
        self.stride = stride

    def forward(self, x):
        residual = self.downsample(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.se(out)

        out += residual
        out = self.relu(out)

        return out

if __name__ == '__main__':
    pass