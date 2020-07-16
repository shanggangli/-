#-*- codeing=utf-8 -*-
#@time: 2020/7/16 10:40
#@Author: Shang-gang Lee
import torch
import torch.nn as nn
from torch.nn import functional as F
class ResBlack(nn.Module):
    def __init__(self,in_channels,out_channels,stride=1):
        super(ResBlack, self).__init__()
        self.Black=nn.Sequential(
            nn.Conv2d(in_channels=in_channels,out_channels=out_channels, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=out_channels,out_channels=out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.shortcut=nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
                nn.BatchNorm2d(out_channels))
    def forward(self,x):
        output=self.Black(x)
        residual=self.shortcut(x)
        output+=residual
        return output

class ResNet18(nn.Module):
    def __init__(self):
        super(ResNet18, self).__init__()
        self.pre=nn.Sequential(
            nn.Conv2d(in_channels=3,out_channels=64,kernel_size=7,stride=2,padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3,stride=2,padding=1)
        )
        self.layer1=self.make_layer(64,64,2)                    #(64,56,56)
        self.layer2=self.make_layer(64,128,2,stride=2)          #(64,28,28)
        self.layer3=self.make_layer(128,256,2,stride=2)         #(64,14,14)
        self.layer4=self.make_layer(256,512,2,stride=2)         #(64,7,7)
        self.fc=nn.Linear(512,4)                                #(64,4)
    def make_layer(self,in_channels,out_channels,num,stride=1):
        layer=[]
        layer.append(ResBlack(in_channels,out_channels,stride)) # !!! stride
        for i in range(1,num):                                  # !!!  range(1,num)
            layer.append(ResBlack(out_channels,out_channels))
        return nn.Sequential(*layer)
    def forward(self,x):
        x = self.pre(x)

        x=self.layer1(x)
        x=self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = F.avg_pool2d(x,7)

        x = x.view(x.size(0), -1)
        x=self.fc(x)
        return torch.sigmoid(x)
# test model        
resnet=ResNet18()
print(resnet)
T=torch.randn(1,3,224,224)
output=resnet(T)
print(output)
