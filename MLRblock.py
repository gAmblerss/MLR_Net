import torch
import torch.nn as nn


class ResNet(nn.Module):
    def __init__(self,input_channels,output_channels,strides=1, **kwargs):
        super(ResNet,self).__init__( **kwargs)
        #线路1 7X7卷积
        self.path1 = nn.Sequential(
            nn.Conv2d(input_channels,output_channels,kernel_size=7,strides = strides,bias=False),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(),
            nn.Conv2d(input_channels, output_channels, kernel_size=7, strides=strides, bias=False),
            nn.BatchNorm2d(output_channels),
            nn.ReLU()
        )
        #线路2 5X5卷积
        self.path2 = nn.Sequential(
            nn.Conv2d(input_channels,output_channels,kernel_size=5,strides = strides,bias=False),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(),
            nn.Conv2d(input_channels, output_channels, kernel_size=5, strides=strides, bias=False),
            nn.BatchNorm2d(output_channels),
            nn.ReLU()
        )
        #线路3 3X3卷积
        self.path3 = nn.Sequential(
            nn.Conv2d(input_channels,output_channels,kernel_size=3,strides = strides,bias=False),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(),
            nn.Conv2d(input_channels, output_channels, kernel_size=3, strides=strides, bias=False),
            nn.BatchNorm2d(output_channels),
            nn.ReLU()
        )
        #线路四 1X1卷积
        self.p4_1 = nn.Conv2d(input_channels,output_channels,kernel_size=1,strides=strides,bias=False)
        self.bn4_1 = nn.BatchNorm2d(output_channels)

    def forward(self,x):
        out_1 = self.path1(x)
        out_2 = self.path2(x)
        out_3 = self.path3(x)
        out_4 = self.p4_1(x)
        out_4 = self.bn4_1(out_4)
        out_c = torch.cat((out_1,out_2),dim=1)
        out_a =out_c+out_3+out_4
        return out_a