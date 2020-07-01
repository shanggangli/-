import torch
from torchvision import transforms, datasets
from torch import nn,optim
import torch.utils.data as Data
class LeNet(nn.Module):
    def __init__(self):
        super(LeNet,self).__init__()
        self.conv=nn.Sequential(    # (1,28,28)
            nn.Conv2d(in_channels=1,out_channels=6,kernel_size=5),   #(6,24,24)
            nn.ReLU(), #(6,24,24)
            nn.MaxPool2d(kernel_size=2,stride=2), #(6,12,12)

            nn.Conv2d(in_channels=6,out_channels=16,kernel_size=5), #(16,8,8)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2) #(16,4,4)
        )
        self.fc=nn.Sequential(
            nn.Linear(16*4*4,120), # dim=3 -> dim=2
            nn.ReLU(),
            nn.Linear(120,84),
            nn.ReLU(),
            nn.Linear(84,10) #output:10
        )
    def forward(self,X):
        feature=self.conv(X)
        output=self.fc(feature.view(X.shape[0],-1))
        return output,X

net=LeNet()
#print(net) #观察模型参数情况

# 加载数据
# parameters
batch_size=256
LR=0.01
Epoch=1

train_dataset=datasets.MNIST(root='C:/untitled/动手学深度学习/minst/',train=True,transform=transforms.ToTensor(),download=False)
test_dataset=datasets.MNIST(root='C:/untitled/动手学深度学习/minst/',train=False,transform=transforms.ToTensor(),download=False)
print(train_dataset.train_data.shape) #torch.Size([60000, 28, 28])
print(train_dataset.train_labels.shape) #torch.Size([60000])

train_loader = Data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True) #loader traindata

test_x=torch.unsqueeze(test_dataset.test_data,dim=1).type(torch.FloatTensor)[:2000]/255  #将数据压缩成（0~1）
print(test_x.shape)# torch.Size([2000, 1, 28, 28])
test_y=test_dataset.test_labels[:2000]
print(test_y.shape)# torch.Size([2000])

# 训练
optimizer=torch.optim.Adam(net.parameters(),lr=LR)
loss_fuc=nn.CrossEntropyLoss()
for epoch in range(Epoch):
    for step,(b_x,b_y) in enumerate(train_loader):
        output=net(b_x)[0]
        loss=loss_fuc(output,b_y) #loss
        optimizer.zero_grad() #clear gradients data
        loss.backward() #backward and calulate gradients
        optimizer.step() # apply gradients and updata parameters

        if step%50==0:
            test_output, last_layer = net(test_x)
            pred_y = torch.max(test_output, 1)[1].data.numpy()
            accuracy = float((pred_y == test_y.data.numpy()).astype(int).sum()) / float(test_y.size(0))
            print('Epoch: ', epoch, '| train loss: %.4f' % loss.data.numpy(), '| test accuracy: %.2f' % accuracy)

test_output, _ = net(test_x[:10])
pred_y = torch.max(test_output, 1)[1].data.numpy()
print(pred_y, 'prediction number')
print(test_y[:10].numpy(), 'real number')