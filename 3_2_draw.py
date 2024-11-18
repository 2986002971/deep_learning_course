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
    def __init__(self, init_type="gaussian"):
        super().__init__()
        self.fc1 = nn.Linear(28 * 28, 28 * 28)
        self.fc2 = nn.Linear(28 * 28, 10 * 10)
        self.fc3 = nn.Linear(10 * 10, 10)

        self.initialize_weights(init_type)

    def initialize_weights(self, init_type):
        if init_type == "gaussian":
            nn.init.normal_(self.fc1.weight)
            nn.init.normal_(self.fc2.weight)
            nn.init.normal_(self.fc3.weight)

        elif init_type == "xavier":
            nn.init.xavier_uniform_(self.fc1.weight)
            nn.init.xavier_uniform_(self.fc2.weight)
            nn.init.xavier_uniform_(self.fc3.weight)

        else:
            nn.init.kaiming_uniform_(self.fc1.weight)
            nn.init.kaiming_uniform_(self.fc2.weight)
            nn.init.kaiming_uniform_(self.fc3.weight)

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
model_gaussian = MLP(init_type="gaussian").to(device)
model_xavier = MLP(init_type="xavier").to(device)
model_kaiming = MLP(init_type="kaiming").to(device)
optimizer_gaussian = torch.optim.SGD(model_gaussian.parameters(), lr=0.01)
optimizer_xavier = torch.optim.SGD(model_xavier.parameters(), lr=0.01)
optimizer_kaiming = torch.optim.SGD(model_kaiming.parameters(), lr=0.01)
criterion = nn.CrossEntropyLoss()

epochs = 50
for epoch in tqdm(range(epochs)):
    model_gaussian.train()
    model_xavier.train()
    model_kaiming.train()
    train_loss_gaussian = 0
    train_loss_xavier = 0
    train_loss_kaiming = 0
    correct_train_gaussian = 0
    correct_train_xavier = 0
    correct_train_kaiming = 0
    total_train = 0
    for data, target in train_loader:
        data, target = data.to(device), target.to(device)

        optimizer_gaussian.zero_grad()
        output_gaussian = model_gaussian(data)
        loss_gaussian = criterion(output_gaussian, target)
        loss_gaussian.backward()
        optimizer_gaussian.step()
        train_loss_gaussian += loss_gaussian.item() / target.size(0)
        _, predicted_gaussian = torch.max(output_gaussian, 1)
        total_train += target.size(0)
        correct_train_gaussian += (predicted_gaussian == target).sum().item()

        optimizer_xavier.zero_grad()
        output_xavier = model_xavier(data)
        loss_xavier = criterion(output_xavier, target)
        loss_xavier.backward()
        optimizer_xavier.step()
        train_loss_xavier += loss_xavier.item() / target.size(0)
        _, predicted_xavier = torch.max(output_xavier, 1)
        correct_train_xavier += (predicted_xavier == target).sum().item()

        optimizer_kaiming.zero_grad()
        output_kaiming = model_kaiming(data)
        loss_kaiming = criterion(output_kaiming, target)
        loss_kaiming.backward()
        optimizer_kaiming.step()
        train_loss_kaiming += loss_kaiming.item() / target.size(0)
        _, predicted_kaiming = torch.max(output_kaiming, 1)
        correct_train_kaiming += (predicted_kaiming == target).sum().item()

    writer.add_scalars(
        "Training Loss",
        {
            "gaussian": train_loss_gaussian,
            "xavier": train_loss_xavier,
            "kaiming": train_loss_kaiming,
        },
        epoch,
    )

    test_loss_gaussian = 0
    test_loss_xavier = 0
    test_loss_kaiming = 0
    correct_test_gaussian = 0
    correct_test_xavier = 0
    correct_test_kaiming = 0
    total_test = 0
    model_gaussian.eval()
    model_xavier.eval()
    model_kaiming.eval()
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output_gaussian = model_gaussian(data)
            output_xavier = model_xavier(data)
            output_kaiming = model_kaiming(data)
            loss_gaussian = criterion(output_gaussian, target)
            loss_xavier = criterion(output_xavier, target)
            loss_kaiming = criterion(output_kaiming, target)
            test_loss_gaussian += loss_gaussian.item() / target.size(0)
            test_loss_xavier += loss_xavier.item() / target.size(0)
            test_loss_kaiming += loss_kaiming.item() / target.size(0)
            _, predicted_gaussian = torch.max(output_gaussian, 1)
            _, predicted_xavier = torch.max(output_xavier, 1)
            _, predicted_kaiming = torch.max(output_kaiming, 1)
            total_test += target.size(0)
            correct_test_gaussian += (predicted_gaussian == target).sum().item()
            correct_test_xavier += (predicted_xavier == target).sum().item()
            correct_test_kaiming += (predicted_kaiming == target).sum().item()

    writer.add_scalars(
        "Test Loss",
        {
            "gaussian": test_loss_gaussian,
            "xavier": test_loss_xavier,
            "kaiming": test_loss_kaiming,
        },
        epoch,
    )
    writer.add_scalars(
        "Accuracy",
        {
            "train_gaussian": correct_train_gaussian / total_train,
            "train_xavier": correct_train_xavier / total_train,
            "train_kaiming": correct_train_kaiming / total_train,
            "test_gaussian": correct_test_gaussian / total_test,
            "test_xavier": correct_test_xavier / total_test,
            "test_kaiming": correct_test_kaiming / total_test,
        },
        epoch,
    )

writer.close()
