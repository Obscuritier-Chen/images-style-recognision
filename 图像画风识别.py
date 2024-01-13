import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import time
from torchvision.models import resnet18

# 加载数据集
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
dataset = ImageFolder('E:\wikiart\wikiart', transform=transform)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# 定义模型
model = resnet18(pretrained=True)
model = nn.Sequential(*list(model.children())[:5])  # 只使用前两个残差块
num_ftrs = model[-1][-1].conv2.out_channels  # 获取最后一个卷积层的输出通道数
model.add_module('avgpool', nn.AdaptiveAvgPool2d((1, 1)))  # 添加全局平均池化层
model.add_module('flatten', nn.Flatten())  # 添加flatten层
model.add_module('classifier', nn.Linear(num_ftrs, len(dataset.classes)))  # 添加分类全连接层

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

# 训练模型
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = model.to(device)
for epoch in range(10):  # 这里我们只训练10轮，你可以根据需要调整
    running_loss = 0.0
    start_time = time.time()
    i=0 
    for inputs, labels in dataloader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        i+=1

        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        if i%100==0:
            print(f'Epoch {epoch+1}/{10} Step [{i}/{len(dataloader)}] Loss: {loss.item():.4f} Time: {time.time()-start_time:.4f}s')\

        #running_loss += loss.item() * inputs.size(0)
    #epoch_loss = running_loss / len(dataset)
    #print(f'Epoch {epoch+1}/{10} Loss: {epoch_loss:.4f} Time: {time.time()-start_time:.4f}s')