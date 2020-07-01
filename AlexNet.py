import time
import torch
from torchvision import transforms, datasets
from torch import nn,optim
import torch.utils.data as Data


batch_size=64
Lr=0.01
Epoch=1
# 加载数据
'''train_data=datasets.MNIST(root='C:/untitled/动手学深度学习/卷积神经网络/mnist/',train=True,transform=transforms.Compose([transforms.Resize(size=224),transforms.ToTensor()]),
                          download=False)
test_data=datasets.MNIST(root='C:/untitled/动手学深度学习/卷积神经网络/mnist/',train=False,transform=transforms.Compose([transforms.Resize(size=224),transforms.ToTensor()]),
                         download=False)

train_loader=torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=True)

test_x=torch.unsqueeze(test_data.test_data,dim=1).type(torch.FloatTensor)[:2000]/255
test_y=test_data.test_labels[:2000]
print('训练集：',test_x.shape)
print('测试集：',test_y.shape)'''

class AlexNet(nn.Module):
    def __init__(self):
        super(AlexNet, self).__init__()
        self.conv=nn.Sequential(#(1,224,224)
            nn.Conv2d(in_channels=1,out_channels=96,kernel_size=11,stride=4,padding=3), #(96,55,55)
            nn.ReLU(),
            nn.MaxPool2d(3,2), #(96,27,27)

            nn.Conv2d(96,256,5,1,2), #(256,27,27)
            nn.ReLU(),
            nn.MaxPool2d(3,2), #(256,13,13)

            nn.Conv2d(256,384,3,1,1), #(384,13,13)
            nn.ReLU(),
            nn.Conv2d(384,384,3,1,1), #(384,13,13)
            nn.ReLU(),
            nn.Conv2d(384,256,3,1,1), #(265,13,13)
            nn.ReLU(),
            nn.MaxPool2d(3,2), #(265,6,6)
        )

        self.fc=nn.Sequential( #全连接
            nn.Linear(256*6*6,4096),
            nn.ReLU(),
            nn.Dropout(0.5),

            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),

            nn.Linear(4096,10),
        )
    def forward(self,X):
        feature=self.conv(X)
        #print('X:',X.shape)
        #print('feature:',feature.shape)
        output=self.fc(feature.view(X.shape[0],-1))
        return output
net=AlexNet()
#print(net)
#训练
input= torch.randn(size=(1,1,224,224))
o=net(input)
print(o.shape)

'''optimtier=torch.optim.Adam(net.parameters(),lr=Lr)
Loss_func=nn.CrossEntropyLoss()
for epoch in range(Epoch):
    for step,(b_x,b_y) in enumerate(train_loader):
        print(b_x.shape)
        output=net(b_x)
        loss=Loss_func(output,b_y)
        optimtier.zero_grad()
        loss.backward()
        optimtier.step()

    if step%50==0:
        test_output, last_layer = net(test_x)
        pred_y = torch.max(test_output, 1)[1].data.numpy()
        accuracy = float((pred_y == test_y.data.numpy()).astype(int).sum()) / float(test_y.size(0))
        print('Epoch: ', epoch, '| train loss: %.4f' % loss.data.numpy(), '| test accuracy: %.2f' % accuracy)'''


