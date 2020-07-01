import torch
from torchvision import transforms, datasets
from torch import nn,optim
from torch.nn import functional as F

batch_size=128
train_data=datasets.MNIST(root='C:/untitled/动手学深度学习/minst/',train=True,transform=transforms.Compose([transforms.Resize(size=(32)),transforms.ToTensor()]),
                          download=False)
test_data=datasets.MNIST(root='C:/untitled/动手学深度学习/minst/',train=False,transform=transforms.Compose([transforms.Resize(size=(32)),transforms.ToTensor()]),
                         download=False)
#print(test_data.test_data.shape)
#print(train_data.train_data.shape)
train_loader=torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
test_x=torch.unsqueeze(test_data.test_data,dim=1).type(torch.FloatTensor)[:2000]/255
#print(test_x.shape)
test_y=test_data.test_labels[:2000]

class NIN(nn.Module):
    def __init__(self):
        super(NIN, self).__init__()
        self.conv=nn.Sequential(
            # (1,1,32,32)
            nn.Conv2d(1,192,kernel_size=5,stride=1,padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(192,160,kernel_size=1,stride=1,padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(160,96,kernel_size=1,stride=1,padding=0),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(kernel_size=3,stride=2,padding=1),
            nn.Dropout(0.5),
            #(1, 96, 16, 16)

            nn.Conv2d(96, 192, kernel_size=5, stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(192, 192, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(192, 192, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(kernel_size=3, stride=2, padding=1),
            nn.Dropout(0.5),
            #(1,192,8,8)

            nn.Conv2d(192, 192, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(192, 192, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(192, 10, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(kernel_size=8, stride=1, padding=0),
            #(1,10,1,1)
        )
    def forward(self,x):
        x=self.conv(x)
        #print(x.shape)
        logits=x.view(x.size(0),10)
        probas = F.softmax(logits)
        return logits,probas

nin=NIN()
Lr=0.01
Epoch=1

'''input=torch.randn(size=(1,1,28,28))
o=nin(input)
print(o[0].size())'''

optimtier=torch.optim.Adam(nin.parameters(),lr=Lr)
Loss_func=nn.CrossEntropyLoss()
for epoch in range(Epoch):
    for step,(b_x,b_y) in enumerate(train_loader):
        #print('b_x',b_x.shape)
        output=nin(b_x)
        loss=Loss_func(output[1],b_y)
        optimtier.zero_grad()
        loss.backward()
        optimtier.step()
        if step/100==0:
            test_output=nin(test_x)
            pred_y = torch.max(test_output[1], 1)[1]
            # print('pred',pred_y.size())
            # print('test',test_y.shape)
            accuracy = float(torch.eq(pred_y, test_y).sum().item()) / len(test_y)
            print(torch.eq(pred_y, test_y).sum().item())
            print('Epoch: ', epoch, '| train loss: %.4f' % loss.data.numpy(), '| test accuracy: %.2f' % accuracy)
