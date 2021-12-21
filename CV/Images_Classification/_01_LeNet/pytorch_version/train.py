import torch
import torchvision
import torch.nn as nn
from model import LeNet
import torch.optim as optim
import torchvision.transforms as transforms


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    # 设置图像预处理方法
    transform = transforms.Compose(
        [transforms.ToTensor(),# 将图像转化为tensor方便并行计算，并将范围【0，255】的值变成范围【0，1】
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]) # 标准化【均值设置】【方差设置】

    # 50000张训练图片
    # 第一次使用时要将download设置为True才会自动去下载数据集
    train_set = torchvision.datasets.CIFAR10(root='./data', train=True,   # 数据集转化为dataset
                                             download=False, transform=transform) # 设置为训练集，使用定义的transform设置处理图片
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=128,
                                               shuffle=True, num_workers=8) # 打乱；线程数目设置；

    # 10000张验证图片
    # 第一次使用时要将download设置为True才会自动去下载数据集
    val_set = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=False, transform=transform)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=10000,
                                             shuffle=False, num_workers=8)
    
    # classes = ('plane', 'car', 'bird', 'cat',
    #            'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    net = LeNet()
    net = net.to(device)
    loss_function = nn.CrossEntropyLoss() # 损失函数设置为 交叉熵损失函数
    optimizer = optim.Adam(net.parameters(), lr=0.001) # 优化器设置为adam

    for epoch in range(5):  # loop over the dataset multiple times

        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data[0].to(device), data[1].to(device)

            # zero the parameter gradients
            optimizer.zero_grad() # 清空参数的梯度，不然会累加
            # forward + backward + optimize
            outputs = net(inputs) # 正向传播
            loss = loss_function(outputs, labels) # 计算损失
            loss.backward() # 反向求每个参数的梯度
            optimizer.step() # 更新参数

            # print statistics
            running_loss += loss.item() # 累加训练的损失

        with torch.no_grad(): # 这句代码下面的运算不参与参数的梯度更新
            val_data_iter = iter(val_loader)# 转化为迭代器模式
            val_image, val_label = val_data_iter.next()# 使用next() 可以取出迭代器里的数据
            val_image,val_label = val_image.to(device), val_label.to(device)
            outputs = net(val_image)  # [batch, 10]
            predict_y = torch.max(outputs, dim=1)[1]
            accuracy = torch.eq(predict_y, val_label).sum().item() / val_label.size(0)

            print('[%d] train_loss: %.3f  test_accuracy: %.3f' %
                  (epoch + 1, running_loss / 500, accuracy))  # 打印一次epoch的训练损失和测试集准确率

    print('Finished Training')

    # 保存模型
    save_path = './Lenet.pth'
    torch.save(net.state_dict(), save_path)


if __name__ == '__main__':
    main()
