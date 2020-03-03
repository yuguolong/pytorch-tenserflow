import torch
import numpy as np
import cv2
import torch.nn as nn
import torchvision.datasets as normal_datasets
import torchvision.transforms as transforms
from torch.autograd import Variable
import tensorflow as tf
from torchvision import datasets,transforms

np.set_printoptions(threshold=np.inf)

dic = torch.load('params_1.pth')

test_loader = torch.utils.data.DataLoader(datasets.MNIST('./data/MNIST/raw',train=False,
                                                          transform=transforms.Compose([
                                                          transforms.Resize((28,28)),
                                                          transforms.ToTensor()
                                                          ]),download =True),
                                           batch_size=128,shuffle=True
                                           )

# for i in dic:
#     print(type(i))

# for key in dic:
#     print(key)
img = cv2.imread('./mnist_train/train_2.bmp',0)
img = np.array(img/255.0)
img=img.astype(np.float)
print(type(img))
img = np.reshape(img,[1,784])

W_fc0 = dic['model.0.weight'].data.numpy()
W_fc0 = np.transpose(W_fc0,[1,0])
W_fc0 = W_fc0.astype(np.float)
b_fc0 = dic['model.0.bias'].data.numpy()
b_fc0 = b_fc0.astype(np.float)
W_fc2 = dic['model.2.weight'].data.numpy()

W_fc2 = np.transpose(W_fc2,[1,0])
W_fc2 = W_fc2.astype(np.float)
b_fc2 = dic['model.2.bias'].data.numpy()
b_fc2 = b_fc2.astype(np.float)

h_fc1 = tf.nn.relu(tf.matmul(img, W_fc0) + b_fc0)
h_fc1 = tf.nn.relu(tf.matmul(h_fc1, W_fc2) + b_fc2)

h_fc1 = (tf.arg_max(h_fc1, 1))


sess = tf.Session() #生成一个会话，通过一个会话session来计算结果
a = sess.run(h_fc1)
# print('pred:',a,'label:',label)
print(a)

for x, label in test_loader:
    # x, label = iter(test_loader).next()
    # x = np.reshape(x, [128, 784])

    # h_fc1 = tf.nn.relu(tf.matmul(x, W_fc0) + b_fc0)
    # h_fc1 = tf.nn.relu(tf.matmul(h_fc1, W_fc2) + b_fc2)
    # # h_fc1 = tf.nn.softmax(tf.matmul(h_fc1, W_fc4) + b_fc4)
    # h_fc1 = (tf.arg_max(h_fc1, 1))
    # print('label:', label)
    #
    # sess = tf.Session() #生成一个会话，通过一个会话session来计算结果
    # a = sess.run(h_fc1)
    # # print('pred:',a,'label:',label)
    # print(a)

    break


# for i in a:
#     print(i)
# sess.close() #关闭会话