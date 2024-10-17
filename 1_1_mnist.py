import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import shutil
import os

log_dir = "runs/mnist"


class MLP(nn.Module):
    def __init__(self, dropout_rate=0.5):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 28*28)
        self.bn1 = nn.BatchNorm1d(28*28)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(28*28, 28*28)
        self.bn2 = nn.BatchNorm1d(28*28)
        self.dropout2 = nn.Dropout(dropout_rate)
        self.fc3 = nn.Linear(28*28, 10)

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = self.fc1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc3(x)
        return x


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
transform = transforms.ToTensor()

train_dataset = torchvision.datasets.MNIST(
    root="./data", train=True, download=True, transform=transform
)
test_dataset = torchvision.datasets.MNIST(
    root="./data", train=False, download=True, transform=transform
)

train_loader = DataLoader(train_dataset, batch_size=2048, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=2048)


if os.path.exists(log_dir):
    shutil.rmtree(log_dir)
os.makedirs(log_dir)

writer = SummaryWriter(log_dir)
model = MLP().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

epochs = 20
for epoch in range(epochs):
    model.train()
    train_loss = 0
    correct_train = 0
    total_train = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        _, predicted = output.max(1)
        total_train += target.size(0)
        correct_train += predicted.eq(target).sum().item()

    train_loss /= len(train_loader)
    train_accuracy = 100. * correct_train / total_train
    writer.add_scalar("Training Loss", train_loss, epoch)
    writer.add_scalar("Training Accuracy", train_accuracy, epoch)


    model.eval()
    correct_test = 0
    total_test = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            _, predicted = output.max(1)
            total_test += target.size(0)
            correct_test += predicted.eq(target).sum().item()

    test_accuracy = 100. * correct_test / total_test
    writer.add_scalar("Testing Accuracy", test_accuracy, epoch)


writer.close()
