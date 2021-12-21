import torch
import torchvision.transforms as transforms
from PIL import Image

from model import LeNet

# 这个地方运算量不大，所以直接写的CPU运算了

def main():
    transform = transforms.Compose(
        [transforms.Resize((32, 32)), # 因为导入的图片尺寸不一样，所以需要resize成网络匹配的尺寸
         transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    net = LeNet()
    net.load_state_dict(torch.load('Lenet.pth')) # 加载训练好的模型

    im = Image.open('1.jpg') # 使用PIL库读取图片
    im = transform(im)  # [C, H, W]
    im = torch.unsqueeze(im, dim=0)  # [N, C, H, W]

    with torch.no_grad():
        outputs = net(im)
        predict = torch.max(outputs, dim=1)[1].data.numpy()
    print(classes[int(predict)])


if __name__ == '__main__':
    main()
