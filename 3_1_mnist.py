import os
import shutil

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets
from torchvision.transforms import ToTensor
from tqdm import tqdm


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(28 * 28, 28 * 28)
        self.fc2 = nn.Linear(28 * 28, 10 * 10)
        self.fc3 = nn.Linear(10 * 10, 10)

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        x = F.softmax(x, dim=1)
        return x


def seed_everything(seed=42):
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


seed_everything()

log_dir = "runs/mnist"
if os.path.exists(log_dir):
    shutil.rmtree(log_dir)
os.makedirs(log_dir)
writer = SummaryWriter(log_dir)


training_data = datasets.MNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor(),
)
test_data = datasets.MNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor(),
)
train_loader = DataLoader(training_data, batch_size=256, shuffle=True)
test_loader = DataLoader(test_data, batch_size=256, shuffle=True)

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"
model = MLP().to(device)
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
criterion = nn.CrossEntropyLoss()

epochs = 10
for epoch in tqdm(range(epochs)):
    model.train()
    train_loss = 0
    correct_train = 0
    total_train = 0
    for data, target in train_loader:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() / target.size(0)
        _, predicted = torch.max(output, 1)
        total_train += target.size(0)
        correct_train += (predicted == target).sum().item()

    writer.add_scalar("Training Loss", train_loss, epoch)

    test_loss = 0
    correct_test = 0
    total_test = 0
    model.eval()
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            test_loss += loss.item() / target.size(0)
            _, predicted = torch.max(output, 1)
            total_test += target.size(0)
            correct_test += (predicted == target).sum().item()

    writer.add_scalar("Test Loss", test_loss, epoch)
    writer.add_scalars(
        "Accuracy",
        {"train": correct_train / total_train, "test": correct_test / total_test},
        epoch,
    )

writer.close()
