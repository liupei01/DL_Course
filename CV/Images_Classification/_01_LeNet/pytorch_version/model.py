import torch.nn as nn
import torch.nn.functional as F


class LeNet(nn.Module): #定义一个LeNet类，继承nn.Module父类
    def __init__(self):
        super(LeNet, self).__init__() #初始化父类，给LeNet调用
        self.conv1 = nn.Conv2d(3, 16, 5) #输入3channel，输出16channel，卷积核5x5，步长默认1，默认无填充
        self.pool1 = nn.MaxPool2d(2, 2) # 池化窗口2x2，步长2，默认无填充
        self.conv2 = nn.Conv2d(16, 32, 5)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(32*5*5, 120)
        self.fc2 = nn.Linear(120, 84) # 输入通道120，输出通道84，默认有偏置bias
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))    # input(3, 32, 32) output(16, 28, 28)
        x = self.pool1(x)            # output(16, 14, 14)
        x = F.relu(self.conv2(x))    # output(32, 10, 10)
        x = self.pool2(x)            # output(32, 5, 5)
        x = x.view(-1, 32*5*5)       # output(32*5*5)
        x = F.relu(self.fc1(x))      # output(120)
        x = F.relu(self.fc2(x))      # output(84)
        x = self.fc3(x)              # output(10)
        return x

# import torch
# input1 = torch.rand([32,3,32,32])
# model = LeNet()
# print(model)
# output = model(input1)
