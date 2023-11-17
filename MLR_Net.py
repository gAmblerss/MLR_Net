import torch
import torch.nn as nn
import MLRblock

class Block_input(nn.Module):
    def __init__(self,input_channel,output_channel,stride=1, **kwargs):
        super(Block_input,self).__init__( **kwargs)
        self.conv1 = nn.Conv2d(input_channel,output_channel,kernel_size=3,stride=stride,padding=0,bias=False)
        self.bn1 = nn.BatchNorm2d(output_channel)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(output_channel,output_channel,kernel_size=3,stride = stride,padding=0,bias=False)
        self.bn2 = nn.BatchNorm2d(output_channel)
        self.relu = nn.ReLU()
        self.Maxpool = nn.MaxPool2d(kernel_size=3,padding=0)

    def forward(self,x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        output,indices = self.Maxpool(out)
        return output,indices

class Block_encoding(nn.Module):
    def __init__(self,input_channels,output_channels,strides=1, **kwargs):
        super(Block_encoding, self).__init__( **kwargs)
        self.resnet = MLRblock.ResNet(input_channels,output_channels,stride=strides)
        self.path = nn.Sequential(
            nn.Conv2d(input_channels,output_channels,kernel_size=7,strides = strides,bias=False),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,padding=0)
        )
    def forward(self,x):
        output = self.resnet(x)
        output,indices = self.path(output)

        return output,indices

class Block_decoding(nn.Module):
    def __init__(self,input_channels,output_channels,strides=1,**kwargs):
        super(Block_decoding,self).__init__(**kwargs)
        self.MaxUnPool = nn.MaxUnpool2d(kernel_size=2, stride=1)
        self.path = nn.Sequential(
            nn.Conv2d(input_channels,output_channels,strides=strides,padding=False),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(),
        )
        self.resnet = MLRblock.ResNet(output_channels, output_channels, strides)
    def forward(self,x,indices_maxpool):
        output = self.MaxUnPool(x,indices_maxpool)
        output = self.path(output)

        return output

class Block_output(nn.Module):
    def __init__(self,input_channels,output_channels,strides=1,**kwargs):
        super(Block_output,self).__init__(**kwargs)
        self.MaxUnPool = nn.MaxUnpool2d(kernel_size=2, strides=1)
        self.path = nn.Sequential(
            nn.Conv2d(input_channels,output_channels,strides=strides,padding=False),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(),
            nn.Conv2d(output_channels, output_channels, strides=strides, padding=False),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(),
        )
    def forward(self,x,indices_maxpool):
        output = self.MaxUnPool(x,indices_maxpool)
        output = self.path(output)
        output = nn.Softmax(output)

        return  output

def link1(input_channels,x):
    GAP = nn.AdaptiveAvgPool2d(input_channels)
    GMP = nn.AdaptiveMaxPool2d(input_channels)
    output_GAP = GAP(x)
    output_GMP = GMP(x)
    output = output_GMP+output_GAP
    conv1 = nn.Conv2d(input_channels,input_channels,kernel_size=2,stride=1,bias=False)
    relu = nn.ReLU()
    conv2 = nn.Conv2d(input_channels,input_channels,kernel_size=2,stride=1,bias=False)
    out_c1=conv1(output)
    out_r = relu(out_c1)
    out_c2=conv2(out_r)
    out_l1 = torch.cat((x,out_c2),dim=1)
    return out_l1

def link2(input_channels,x):
    Max_pool = nn.AdaptiveMaxPool2d(input_channels)
    #Avg_pool = nn.AdaptiveAvgPool2d(input_channels)
    conv = nn.Conv2d(input_channels,input_channels,kernel_size=2,stride=1,bias=False)
    relu = nn.ReLU()
    out_pool = Max_pool(x)
    out_c = conv(out_pool)
    out_r = relu(out_c)
    return torch.cat((x,out_r),dim=1)
