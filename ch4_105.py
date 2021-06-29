# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pylab as plt 
from mnist import load_mnist
from ch4_103 import TwoLayerNet

(x_train, t_train),(x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)

train_loss_list = []
train_acc_list = []
test_acc_list = []



#超參數
iter_num = 10000    #梯度法的更新次數
train_size = x_train.shape[0]   #訓練資料的大小
batch_size = 100    #小批次執行的大小
learning_rate = 0.1     #學習率
#每1 epoch的重複次數
iter_per_epoch = max(train_size / batch_size, 1)


network = TwoLayerNet(input_size=784, hidden_size=100, output_size=10)

for i in range(iter_num):
    #取得小批次
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]
    
    #計算梯度
    #數值微分版 grad = network.numerical_gradient(x_batch, t_batch)
    grad = network.gradient(x_batch, t_batch)   #誤差反向傳播法版
    
    #更新參數
    for key in('W1', 'b1', 'W2', 'b2'):
        network.params[key] -= learning_rate * grad[key]
        
    loss = network.loss(x_batch, t_batch)
    train_loss_list.append(loss)
    
    #計算1 epoch的便是準確度
    if i%iter_per_epoch == 0:
        train_acc = network.accuracy(x_train, t_train)
        test_acc = network.accuracy(x_test, t_test)
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)
        print("train_acc, test_acc |" +str(train_acc) + ", " +str(test_acc))
        



plt.subplot(2,1,1)  
x = np.arange(iter_num)
plt.plot(x,train_loss_list)
plt.xlabel("iteration")
plt.ylabel("loss")



plt.subplot(2,1,2)
x = np.arange(len(train_acc_list))
plt.plot(x, train_acc_list, label='train acc')
plt.plot(x, test_acc_list, label='test acc', linestyle='--')
plt.xlabel("epochs")
plt.ylabel("accuracy")
plt.ylim(0, 1.0)   
plt.show 