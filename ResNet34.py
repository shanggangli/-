import torch as t
from torchvision import transforms, datasets
from torch import nn,optim
from torch.nn import functional as F

class Res_block(nn.Module):
    def __init__(self,in_channels,out_channels,stride=1,shortcut=None):
        super(Res_block, self).__init__()
        self.conv=nn.Sequential( #Black architecture
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=stride, padding=1,bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=out_channels,out_channels=out_channels, kernel_size=3, stride=1, padding=1,bias=False),
            nn.BatchNorm2d(out_channels),
        )
        self.right = shortcut

    def forward(self,x):
        out=self.conv(x)
        residual=x if self.right is None else self.right(x)
        out+=residual
        return F.relu(out) ###

class ResNet(nn.Module):
    def __init__(self,num_classes=1000):
        super(ResNet, self).__init__()
        self.pre=nn.Sequential(
            nn.Conv2d(in_channels=3,out_channels=64,kernel_size=7,stride=2,padding=3,bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3,stride=2,padding=1)
        )

        self.layer1=self.make_layer(64,64,3)
        self.layer2 = self.make_layer(64, 128, 4, stride=2)
        self.layer3 = self.make_layer(128, 256, 6, stride=2)
        self.layer4 = self.make_layer(256,512,3, stride=2)

        self.fc=nn.Linear(512,num_classes)

    def make_layer(self,in_channels,out_channels,block_num,stride=1):
        shortcut = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
            nn.BatchNorm2d(out_channels))

        layers=[]
        layers.append(Res_block(in_channels,out_channels,stride,shortcut))

        for i in range(1,block_num):
            layers.append(Res_block(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self,x):
        x = self.pre(x)

        x=self.layer1(x)
        x=self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = F.avg_pool2d(x, 7) ####
        x = x.view(x.size(0), -1)
        return self.fc(x)

model=ResNet()
print(model)
input= t.autograd.Variable(t.randn(1,3,224,224))
o=model(input)
print(o.size())
