import torch
from torch import nn
from torch import optim
from torchvision import datasets,transforms
import matplotlib.pyplot as plt
import numpy as np

train_loader = torch.utils.data.DataLoader(datasets.MNIST('./data/MNIST/raw',train=True,
                                                          transform=transforms.Compose([
                                                          transforms.Resize((28,28)),
                                                          transforms.ToTensor()
                                                          ]),download =True),
                                           batch_size=512,shuffle=True
                                           )

test_loader = torch.utils.data.DataLoader(datasets.MNIST('./data/MNIST/raw',train=False,
                                                          transform=transforms.Compose([
                                                          transforms.Resize((28,28)),
                                                          transforms.ToTensor()
                                                          ]),download =True),
                                           batch_size=512,shuffle=True
                                           )

class lenet5(nn.Module):
    def __init__(self):
        super(lenet5,self).__init__()
        self.model = nn.Sequential(
            nn.Linear(28*28,24),
            nn.ReLU(),
            nn.Linear(24,10),
            nn.Softmax()
        )
    def forward(self, x):
        x = x.view(x.size(0),784)
        pred = self.model(x)
        return pred

train_loss = []
test_loss = []
train_acc = []
test_acc = []
epoch_num = []
def main():
    criteon = nn.CrossEntropyLoss()
    model = lenet5()
    optimizer = optim.Adam(model.parameters(),lr=1e-4)
    for epoch in range(35):
        total_correct = 0
        total_num = 0
        for step,(x,label) in enumerate(train_loader):
            pred = model(x)
            train_los = criteon(pred,label)
            optimizer.zero_grad()
            train_los.backward()
            optimizer.step()

            pred = pred.argmax(dim=1)
            total_correct += torch.eq(pred, label).sum().numpy()
            total_num += x.size(0)
        train_ac = total_correct / total_num

        train_loss.append(train_los)
        train_acc.append(train_ac)
        epoch_num.append(epoch)

        total_correct1 = 0
        total_num1 = 0
        for step,(x,label) in enumerate(test_loader):
            pred = model(x)
            test_los = criteon(pred, label)
            optimizer.zero_grad()
            test_los.backward()
            optimizer.step()


            pred = pred.argmax(dim=1)
            total_correct1 += torch.eq(pred, label).sum().numpy()
            total_num1 += x.size(0)
        test_ac = total_correct1 / total_num1
        test_loss.append(test_los)
        test_acc.append(test_ac)

        print("epoch:",epoch,"train_acc:%f.4"%(train_ac),"test_acc:",test_ac,'train_loss:',train_los.data.numpy(),"test_loss:",test_los.data.numpy())#打印每轮的输出效果

    plt.plot(epoch_num, train_loss,color='r',linestyle="-",label="train_loss")
    plt.plot(epoch_num, test_loss, color='g',linestyle="-",label='test_loss')
    plt.plot(epoch_num, train_acc, color='b',linestyle="-",label='train_acc')
    plt.plot(epoch_num, test_acc, color='m',linestyle="-",label='test_acc')
    plt.legend()
    plt.show()

if __name__ == '__main__':
    main()
