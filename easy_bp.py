import torch 
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import h5py
Num=50#迭代次数
# Loading the data (cat/non-cat)
train_dataset = h5py.File('train_catvnoncat.h5', "r")
train_set_x_orig = np.array(train_dataset["train_set_x"][:])  # train set features
train_set_y_orig = np.array(train_dataset["train_set_y"][:])  # train set labels

test_dataset = h5py.File('test_catvnoncat.h5', "r")
test_set_x_orig = np.array(test_dataset["test_set_x"][:])  # test set features
test_set_y_orig = np.array(test_dataset["test_set_y"][:])  # test set labels

classes = np.array(test_dataset["list_classes"][:])  # the list of classes


test_set_y = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))
train_set_x=np.zeros((209,64*64*3),float)
test_set_x=np.zeros((50,64*64*3),float)
x=torch.zeros(209,64*64*3)
Te_x=torch.zeros(50,64*64*3)
for t in range(209):
    train_set_x[t,:]=train_set_x_orig[t,:,:,:].flatten()
for t in range(50):
    test_set_x[t,:]=test_set_x_orig[t,:,:,:].flatten()
Tr_x=torch.from_numpy(train_set_x)
Tr_y=torch.from_numpy(train_set_y_orig)
Te_y=torch.from_numpy(test_set_y_orig)
test_set_x=torch.from_numpy(test_set_x)
test_set_y=torch.from_numpy(test_set_y)
Te_x=Te_x.float()
Tr_x=Tr_x.float()
for t in range(50):
    Te_x[t,:]=(test_set_x[t,:]-torch.mean(test_set_x[t,:]))/torch.std(test_set_x[t,:])#209*12288

for t in range(209):
   x[t,:]=(Tr_x[t,:]-torch.mean(Tr_x[t,:]))/torch.std(Tr_x[t,:])#209*12288

print(Te_x.size())
y=Tr_y.long()#209*1

#画原始数据
plt.ion() 
plt.figure()
for t in range(50):
  plt.subplot(8,8,t+1)
  plt.imshow(test_set_x_orig[t,:,:,:], interpolation = 'nearest')
  plt.axis('off')
#建立网络
class Net(torch.nn.Module):
    def __init__(self, n_feature, n_hidden, n_output):
        super(Net, self).__init__()
        self.hidden = torch.nn.Linear(n_feature, n_hidden)   # hidden layer
        self.out = torch.nn.Linear(n_hidden, n_output)   # output layer

    def forward(self, x):
        x = F.relu(self.hidden(x))      # activation function for hidden layer
        x = self.out(x)
        return x

net = Net(n_feature=64*64*3, n_hidden=100, n_output=2)     # define the network
print(net)  # net architecture

#训练网络

optimizer = torch.optim.SGD(net.parameters(), lr=0.2)
loss_func = torch.nn.CrossEntropyLoss()  # the target label is NOT an one-hotted
plt.figure()
error=np.zeros(Num)
error1=np.zeros(Num)
for t in range(Num):
    net.train()
    out = net(x)                 # input x and predict based on x
    loss = loss_func(out,y)     # must be (1. nn output, 2. target), the target label is NOT one-hotted

    optimizer.zero_grad()   # clear gradients for next train
    loss.backward()         # backpropagation, compute gradients
    optimizer.step()        # apply gradients
    prediction = torch.max(out, 1)[1]
    pred_y = prediction.data.numpy()
    error[t]=sum(abs(y.numpy()-pred_y)) 
    net.eval()
    outputs=net(Te_x)
    prediction = torch.max(outputs, 1)[1]
    pred_y1 = prediction.data.numpy()
    error1[t] =sum(abs(Te_y.numpy()-pred_y1)) 
x_label=np.linspace(0,Num,Num)  
plt.plot(x_label,(error/209),c='r')
plt.plot(x_label,(error1/50),c='b')
plt.ylim(0,1)
#测试网络


#画图
plt.figure()
black_shadow=np.zeros((64,64),float)
for t in range(50):
    plt.subplot(8,8,t+1)
    if pred_y1[t]==1:
      plt.imshow(test_set_x_orig[t,:,:,:], interpolation = 'nearest')
      plt.axis('off')
    else:
      plt.imshow(black_shadow, interpolation = 'nearest')
      plt.axis('off')
plt.ioff()
plt.show()
            
