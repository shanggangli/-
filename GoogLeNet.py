import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F

# hyper parameters
EPOCH=1 # just train 1 epoch
LR=0.01 #learing rate=0.01
batch_size=64 #batch_size=64

class inception(nn.Module):
    def __init__(self,in_c1,out_c1,c2,c3,c4):
        super(inception, self).__init__()
        # 1x1 layer
        self.Channel_1=nn.Conv2d(in_channels=in_c1,out_channels=out_c1,kernel_size=1)

        #1x1 layer after 3x3layer
        self.Channel_21=nn.Conv2d(in_channels=in_c1,out_channels=c2[0],kernel_size=1)
        self.Channel_22=nn.Conv2d(in_channels=c2[0],out_channels=c2[1],kernel_size=3,stride=1,padding=1)

        #1x1 layer after 5x5layer
        self.Channel_31=nn.Conv2d(in_channels=in_c1,out_channels=c3[0],kernel_size=1)
        self.Channel_32=nn.Conv2d(in_channels=c3[0],out_channels=c3[1],kernel_size=5,stride=1,padding=2)

        #maxpool 3x3 after 3x3layer
        self.Channel_41=nn.MaxPool2d(kernel_size=3,stride=1,padding=1)
        self.Channel_42=nn.Conv2d(in_channels=in_c1,out_channels=c4,kernel_size=3,stride=1,padding=1)

    def forward(self,input):
        C1=F.relu(self.Channel_1(input)) #Channel_1 forward
        #print(C1.shape)
        C2=F.relu(self.Channel_22(F.relu(self.Channel_21(input)))) # Channel_2 forward
        #print(C2.shape)
        C3=F.relu(self.Channel_32(F.relu(self.Channel_31(input)))) # Channel_3 forward
        #print(C3.shape)
        C4=F.relu(self.Channel_42(F.relu(self.Channel_41(input)))) # Channel_4 forward
        #print(C4.shape)
        cat=torch.cat((C1, C2, C3, C4), dim=1) #merge: (C1,C2,C3,C4) in dim=1
        #print(cat.shape)
        return cat

class GoogLeNet(nn.Module):
    def __init__(self):
        super(GoogLeNet, self).__init__()
        self.b1 = nn.Sequential(  # (1,224,224)
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=7, stride=2, padding=3),  # (64,112,122)
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),  # (64,56,56)

            nn.Conv2d(64, 192, 3, 1, 1),  # (192,56,56)
            nn.ReLU(True),
            nn.MaxPool2d(3, 2, 1),  # (192,28,28)

            # inception(in_c1,out_c1,c2,c3,c4)
            #3a
            inception(192,64,(94,128),(16,32),32), #(64+128+32+32=265,28,28)->(265,28,28)
            #3b
            inception(256, 128, (128, 192), (32, 96), 64),  # (128+192+96+64,28,28)->(480,28,28)
            nn.MaxPool2d(3,2,1),# (480,14,14)

            #4a
            inception(480,192,(96,208),(16,48),64),#(512,14,14)
            #4b
            inception(512,160,(112,224),(24,64),64),#(512,14,14)
            #4c
            inception(512,128,(128,256),(24,64),64),#(512,14,14)
            #4d
            inception(512,112,(144,288),(32,64),64),#(528,14,14)
            #4e
            inception(528,256,(160,320),(32,128),128),#(832,14,14)
            nn.MaxPool2d(3,2,1),#(832,7,7)

            #5a
            inception(832,256,(160,320),(32,128),128),#(832,7,7)
            #5b
            inception(832,384,(192,384),(48,128),128),#(1024,7,7)
            nn.AvgPool2d(kernel_size=7,stride=1),#(1024,1,1)
            nn.Dropout(0.4)
        )
        self.fc=nn.Sequential(
            nn.Linear(1024,10), # full connetion

        )
    def forward(self,x):
        out=self.b1(x)
        #print('1:',out.shape)
        out=out.view(out.size(0),-1)
        #print('2:', out.shape)
        out=self.fc(out)
        softmax=nn.Softmax(dim=0) #softmax
        return softmax(out)

model=GoogLeNet()


# test model 
'''test_model=torch.randn(size=(1,1,224,224))
o=model(test_model)
print(o.size())'''
opitimter=torch.optim.Adam(model.parameters(),lr=LR)
loss_fuc=nn.CrossEntropyLoss()

# loading data
train_data=datasets.MNIST(root='C:/untitled/动手学深度学习/卷积神经网络/mnist/',train=True,
                          transform=transforms.Compose([transforms.Resize(size=224),transforms.ToTensor()]),
                          download=False)
test_data=datasets.MNIST(root='C:/untitled/动手学深度学习/卷积神经网络/mnist/',train=False,
                         transform=transforms.Compose([transforms.Resize(size=224),transforms.ToTensor()]),
                         download=False)

train_loader=torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True) #loader data
test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=True)

test_x=torch.unsqueeze(test_data.test_data,dim=1).type(torch.FloatTensor)[:2000]/255 # test data
test_y=test_data.test_labels[:2000]
print('训练集：',test_x.shape) #torch.Size([2000, 1, 28, 28])
print('测试集：',test_y.shape) #torch.Size([2000])

for epoch in range(EPOCH):
    for step,(b_x,b_y) in enumerate(train_loader):
        output=model(b_x) #forward
        loss=loss_fuc(output,b_y) #loss function--CrossEntropyLoss
        opitimter.zero_grad() #clear grad
        loss.backward() # grad and backward
        opitimter.step() # updata weights

    if step%50==0:
        test_output=model(test_x) # test
        test_loss = loss_fuc(test_output, test_y)
        pred_y = torch.max(test_output, 1)[1] #predtion
        accuracy=torch.eq(pred_y,test_y).sum().float().item()/len(test_y) # accuracy rate= accuracy%len(test_y)
        print('epoch:',epoch,'|step:',step,'|loss:%.2f'%test_loss.data.numpy(),'|accuracy:%.2f'% accuracy)
