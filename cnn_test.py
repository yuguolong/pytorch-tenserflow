import torch
import numpy as np
import cv2
import tensorflow.contrib.slim as slim
import torchvision.datasets as normal_datasets
import torchvision.transforms as transforms
import torch
from torch.autograd import Variable
import tensorflow as tf
from torchvision import datasets,transforms

np.set_printoptions(threshold=np.inf)
dic = torch.load('params3.pth')

test_loader = torch.utils.data.DataLoader(datasets.MNIST('./data/MNIST/raw',train=False,
                                                          transform=transforms.Compose([
                                                          transforms.Resize((28,28)),
                                                          transforms.ToTensor()
                                                          ]),download =True),
                                           batch_size=128,shuffle=False
                                           )

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')
def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],padding='SAME')

W_conv1 = dic['conv1.0.weight'].data.numpy()
W_conv1 = np.transpose(W_conv1,[2,3,1,0])
b_conv1 = dic['conv1.0.bias'].data.numpy()

W_conv2 = dic['conv2.0.weight'].data.numpy()
W_conv2 = np.transpose(W_conv2,[2,3,1,0])
b_conv2 = dic['conv2.0.bias'].data.numpy()

W_fc0 = dic['fc.0.weight'].data.numpy()
W_fc0 = np.transpose(W_fc0,[1,0])
b_fc0 = dic['fc.0.bias'].data.numpy()

W_fc1 = dic['fc.1.weight'].data.numpy()
W_fc1 = np.transpose(W_fc1,[1,0])
b_fc1 = dic['fc.1.bias'].data.numpy()
#
for x, label in test_loader:
    x = x
    x = tf.transpose(x, perm=[0,2,3,1])

    h_conv1 = tf.nn.relu(conv2d(x, W_conv1)+b_conv1)
    h_conv1 = max_pool_2x2(h_conv1)
    h_conv1 = tf.nn.relu(conv2d(h_conv1, W_conv2) + b_conv2)
    h_conv1 = max_pool_2x2(h_conv1)
    # print(h_conv1)

    with tf.Session() as sess:
        h_conv1 = h_conv1.eval()
    # h_conv1 = h_conv1.eval()
    h_conv1 = np.transpose(h_conv1, [0, 3, 1, 2])
    # print(h_conv1)
    # h_conv1 = tf.convert_to_tensor(h_conv1)
    # h_conv1 = tf.transpose(h_conv1, perm=[0, 3, 1, 2])

    h_conv1 = tf.reshape(h_conv1, [-1, 7 * 7 * 32])
    h_fc1 = tf.nn.relu(tf.matmul(h_conv1, W_fc0) + b_fc0)
    h_fc1 = tf.nn.softmax(tf.matmul(h_fc1, W_fc1) + b_fc1)
    h_fc1 = (tf.arg_max(h_fc1, 1))

    sess = tf.Session()
    a = sess.run(h_fc1)

    print('pred:', a, '\n', 'label:', label)
    # print('pred:',a,'\n','label:',label[0:1])
    # break

# for i in a:
#     print(i)
# sess.close() #关闭会话