import torch
import numpy as np
import torch.nn as nn
import torchvision.datasets as normal_datasets
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch.nn.functional as F

num_epochs = 2
batch_size = 500
learning_rate = 0.001

def get_variable(x):
    x = Variable(x)
    return x.cuda() if torch.cuda.is_available() else x

train_dataset = normal_datasets.MNIST(
    root='./mnist/',  # 数据集保存路径
    train=True,  # 是否作为训练集
    transform=transforms.ToTensor(),  # 数据如何处理, 可以自己自定义
    download=True)  # 路径下没有的话, 可以下载

test_dataset = normal_datasets.MNIST(root='./mnist/',
                                     train=False,
                                     transform=transforms.ToTensor()
                                     )

train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size,
                                          shuffle=False)

# 两层卷积
class CNN(torch.nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(
            torch.nn.Conv2d(1, 8, kernel_size=3,padding=1),
            nn.BatchNorm2d(8),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),
        )
        self.conv2 = nn.Sequential(
            torch.nn.Conv2d(8, 32, kernel_size=3,padding=1),
            nn.BatchNorm2d(32),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2)
        )
        self.fc = nn.Sequential(
			nn.Linear(7 * 7 * 32, 128),
            nn.BatchNorm1d(128),
            nn.Dropout(0.3),
			nn.Linear(128, 10))

    def forward(self,x):
        out = self.conv1(x)

        out = self.conv2(out)
        out = out.view(out.size(0), -1)  # reshape
        out = self.fc(out)
        return out

cnn = CNN()
if torch.cuda.is_available():
    cnn = cnn.cuda()

# 选择损失函数和优化方法
loss_func = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(cnn.parameters(), lr=learning_rate)
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        images = get_variable(images)
        labels = get_variable(labels)

        outputs = cnn(images)
        loss = loss_func(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        torch.save(cnn.state_dict(), 'params3.pth')
    # dic = cnn.state_dict()
    # print(dic['conv2.0.bias'])

    total_correct = 0
    total_num = 0
    for x, label in test_loader:
        pred = cnn(x)
        pred = pred.argmax(dim=1)
        total_correct += torch.eq(pred, label).float().sum().item()
        total_num += x.size(0)
    acc = total_correct / total_num
    print("acc:", acc)

# Save the Trained Model
# for parameters in cnn.parameters():
#     print(len(parameters))
#     a  = parameters

#加载保存的参数
# dic = torch.load('params.pth')

# print(dic)
# cnn.load_state_dict(dic)
# a = dic['fc.weight']
# print(a.shape)
# torch.save(cnn.state_dict(), 'cnn.pkl')


#
# def Record_Tensor(tensor,name):
# 	print ("Recording tensor "+name+" ...")
# 	f = open('C:\\Users\\yu guo long\\PycharmProjects\\untitled3\\pytorch\\data\\'+name+'.dat', 'w')
# 	# print(type(tensor))
# 	# array = tensor.eval();
# 	# array=np.array(tensor)
# 	array = tensor
# 	#print ("The range: ["+str(np.min(array))+":"+str(np.max(array))+"]")
# 	if(np.size(np.shape(array))==1):
# 		Record_Array1D(array,name,f)
# 	else:
# 		if(np.size(np.shape(array))==2):
# 			Record_Array2D(array,name,f)
# 		else:
# 			if(np.size(np.shape(array))==3):
# 				Record_Array3D(array,name,f)
# 			else:
# 				Record_Array4D(array,name,f)
# 	f.close();
#
# def Record_Array1D(array,name,f):
# 	print(array.shape)
# 	for i in range(np.shape(array)[0]):
# 		f.write(str(array[i])+"\n");
#
# def Record_Array2D(array,name,f):
# 	array = np.ascontiguousarray(array.transpose())
# 	# array = np.ascontiguousarray(array.transpose())
# 	# array = array.transpose([1,0])
# 	print(array.shape)
# 	for i in range(np.shape(array)[0]):
# 		for j in range(np.shape(array)[1]):
# 			f.write(str(array[i][j])+"\n");
#
# def Record_Array3D(array,name,f):
# 	# print(array.shape[0])
# 	for i in range(np.shape(array)[0]):
# 		for j in range(np.shape(array)[1]):
# 			for k in range(np.shape(array)[2]):
# 				f.write(str(array[i][j][k])+"\n");
#
# def Record_Array4D(array,name,f):
# 	array = np.ascontiguousarray(array.transpose([2, 3, 1, 0]))
# 	# array = array.transpose([2, 3, 1, 0])
# 	print(array.shape)
# 	for i in range(np.shape(array)[0]):
# 		for j in range(np.shape(array)[1]):
# 			for k in range(np.shape(array)[2]):
# 				for l in range(np.shape(array)[3]):
# 					f.write(str(array[i][j][k][l])+"\n");
# #
# dic = torch.load('params.pth')
# for key in dic:
# 	print(key)
#
# # dic = cnn.state_dict()
# # for key in dic:
# # 	print(key)
#
# W_conv1 = dic['conv1.0.weight'].data.numpy()
# # print(type(W_conv1))
# b_conv1 = dic['conv1.0.bias'].data.numpy()
# W_conv2 = dic['conv2.0.weight'].data.numpy()
# b_conv2 = dic['conv2.0.bias'].data.numpy()
# W_fc1 = dic['fc.0.weight'].data.numpy()
# b_fc1 = dic['fc.0.bias'].data.numpy()
# W_fc2 = dic['fc.1.weight'].data.numpy()
# b_fc2 = dic['fc.1.bias'].data.numpy()

# Record_Tensor(W_conv1,"W_conv1")
# Record_Tensor(b_conv1,"b_conv1")
# Record_Tensor(W_conv2,"W_conv2")
# Record_Tensor(b_conv2,"b_conv2")
# Record_Tensor(W_fc1,"W_fc1")
# Record_Tensor(b_fc1,"b_fc1")
# Record_Tensor(W_fc2,"W_fc2")
# Record_Tensor(b_fc2,"b_fc2")

