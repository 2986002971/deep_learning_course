import os
import shutil

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.utils.tensorboard import SummaryWriter

log_dir = "runs/fashion_mnist"


class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool = nn.AvgPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.sigmoid(self.conv1(x)))
        x = self.pool(F.sigmoid(self.conv2(x)))
        x = x.view(-1, 16 * 4 * 4)
        x = F.sigmoid(self.fc1(x))
        x = F.sigmoid(self.fc2(x))
        x = self.fc3(x)
        return x

    def get_features(self, x):
        """获取中间层特征图"""
        features = []
        x = self.conv1(x)
        features.append(x)  # conv1 features
        x = F.sigmoid(x)
        x = self.pool(x)

        x = self.conv2(x)
        features.append(x)  # conv2 features
        x = F.sigmoid(x)
        x = self.pool(x)
        return features


def visualize_features(model, dataloader, device, writer):
    model.eval()
    # 获取一个batch的图像
    images, _ = next(iter(dataloader))
    img_grid = vutils.make_grid(images[:16], normalize=True)
    writer.add_image("Original Images", img_grid)

    # 获取特征图
    images = images[:16].to(device)
    features = model.get_features(images)

    # 可视化每个卷积层的特征图
    for layer_idx, feature_maps in enumerate(features):
        # 选择第一张图片的所有特征图
        feature_maps = feature_maps[0].detach().cpu()
        feature_grid = vutils.make_grid(
            feature_maps.unsqueeze(1), normalize=True, nrow=8
        )
        writer.add_image(f"Feature Maps/Layer_{layer_idx+1}", feature_grid)


def visualize_kernels(model, writer):
    # 可视化第一个卷积层的卷积核
    kernels = model.conv1.weight.detach().cpu()
    # 确保kernels的形状正确 [out_channels, in_channels, height, width]
    kernel_grid = vutils.make_grid(kernels, normalize=True, nrow=8, padding=1)
    writer.add_image("Kernels/Conv1", kernel_grid)

    # 可视化第二个卷积层的卷积核
    kernels = model.conv2.weight.detach().cpu()
    # 重新排列kernels以便可视化
    kernels = kernels.view(-1, 1, kernels.shape[2], kernels.shape[3])
    kernel_grid = vutils.make_grid(kernels, normalize=True, nrow=8, padding=1)
    writer.add_image("Kernels/Conv2", kernel_grid)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
)
trainset = torchvision.datasets.FashionMNIST(
    root="./data", train=True, download=True, transform=transform
)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=1024, shuffle=True)
testset = torchvision.datasets.FashionMNIST(
    root="./data", train=False, download=True, transform=transform
)
testloader = torch.utils.data.DataLoader(testset, batch_size=1024)

model = LeNet().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

if os.path.exists(log_dir):
    shutil.rmtree(log_dir)
os.makedirs(log_dir)
writer = SummaryWriter(log_dir)

epochs = 30
for epoch in range(epochs):
    train_loss = 0.0
    train_correct = 0
    train_total = 0
    model.train()
    for inputs, labels in trainloader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        train_total += labels.size(0)
        train_correct += (predicted == labels).sum().item()

    print(
        f"Epoch {epoch + 1}, Train Accuracy: {train_correct / train_total * 100:.2f}%"
    )
    writer.add_scalar("Loss/train", train_loss / len(trainloader), epoch)

    test_correct = 0
    test_total = 0
    model.eval()
    with torch.no_grad():
        for inputs, labels in testloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            _, predicted = torch.max(outputs.data, 1)
            test_total += labels.size(0)
            test_correct += (predicted == labels).sum().item()

    print(f"Test Accuracy: {test_correct / test_total * 100:.2f}%")
    writer.add_scalars(
        "Accuracy",
        {"train": train_correct / train_total, "test": test_correct / test_total},
        epoch,
    )

visualize_features(model, testloader, device, writer)
visualize_kernels(model, writer)

writer.close()
