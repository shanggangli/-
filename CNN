import torch
import torch.nn as nn
import numpy as np
import torchvision.transforms as transforms
import torchvision.datasets as datasets

# hyper parameters
EPOCH=1
batch_size=64
time_step=28
input_size=28
LR=0.01

#加载数据
train_data=datasets.MNIST(root='C:/untitled/动手学深度学习/minst/',train=True,transform=transforms.ToTensor(),
                          download=False)
testdata=datasets.MNIST(root='C:/untitled/动手学深度学习/minst/',train=False,transform=transforms.ToTensor(),
                          download=False)
train_loader = torch.utils.data.DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)
test_x=torch.unsqueeze(testdata.test_data,dim=1).type(torch.FloatTensor)[:2000]/255 # test_x(0,1)
print(test_x.size())
test_y=testdata.test_labels[:2000]

#建模
class RNN(nn.Module):
    def __init__(self):
        super(RNN, self).__init__()
        self.rnn=nn.RNN(input_size=28,hidden_size=64,num_layers=1,batch_first=True)
        self.out=nn.Linear(64,10) #output shape(10)
    def forward(self,input):
        '''
        input of shape (batch,seq_len, input_size)
        h_0 of shape (batch,num_layers * num_directions,hidden_size)

        r_out=[batch,seq_len, num_directions * hidden_size]
        h_n=[ batch,num_layers * num_directions, hidden_size]
        '''
        r_out,h_n=self.rnn(input,None)
        out = self.out(r_out[:, -1, :])
        return out
rnn=RNN()
print(rnn)

optimiter=torch.optim.Adam(rnn.parameters(),lr=LR)
loss_func=nn.CrossEntropyLoss()

for epoch in range(EPOCH):
    for step,(x,b_y) in enumerate(train_loader):
        input=x.view(-1,28,28)
        output=rnn(input)
        loss=loss_func(output,b_y)
        optimiter.zero_grad()
        loss.backward()
        optimiter.step()

        if step %50==0:
            test_x=test_x.view(-1,28,28)
            test_output=rnn(test_x)
            pred_y = torch.max(test_output, 1)[1]
            #print('pred',pred_y.size())
            #print('test',test_y.shape)
            accuracy = float(torch.eq(pred_y,test_y).sum().item()) / len(test_y) 
            print(torch.eq(pred_y,test_y).sum().item())
            print('Epoch: ', epoch, '| train loss: %.4f' % loss.data.numpy(), '| test accuracy: %.2f' % accuracy)
