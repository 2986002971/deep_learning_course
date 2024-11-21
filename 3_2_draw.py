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


if __name__ == "__main__":
    seed_everything()

    log_dir = "runs/mnist_batch_size"
    if os.path.exists(log_dir):
        shutil.rmtree(log_dir)
    os.makedirs(log_dir)
    writer = SummaryWriter(log_dir)

    # 加载数据集
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

    # 创建不同batch_size的数据加载器
    batch_sizes = {
        "full_batch": len(training_data),  # 全量数据
        "batch_512": 512,  # 常规batch size
        "batch_64": 64,
        "batch_8": 8,
        "sgd": 1,  # 随机梯度下降
    }

    train_loaders = {
        name: DataLoader(training_data, batch_size=size, shuffle=True, num_workers=24)
        for name, size in batch_sizes.items()
    }
    test_loaders = {
        name: DataLoader(test_data, batch_size=size, shuffle=True, num_workers=24)
        for name, size in batch_sizes.items()
    }

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 为每个batch_size创建独立的模型和优化器
    models = {name: MLP().to(device) for name in batch_sizes.keys()}
    optimizers = {
        name: torch.optim.SGD(model.parameters(), lr=0.01)
        for name, model in models.items()
    }
    criterion = nn.CrossEntropyLoss()

    epochs = 50
    for epoch in tqdm(range(epochs)):
        # 训练阶段
        train_losses = {name: 0 for name in batch_sizes.keys()}
        train_steps = {name: 0 for name in batch_sizes.keys()}
        train_correct = {name: 0 for name in batch_sizes.keys()}
        train_total = {name: 0 for name in batch_sizes.keys()}

        for name in batch_sizes.keys():
            models[name].train()
            for data, target in train_loaders[name]:
                data, target = data.to(device), target.to(device)

                optimizers[name].zero_grad()
                output = models[name](data)
                loss = criterion(output, target)
                loss.backward()
                optimizers[name].step()

                train_losses[name] += loss.item()
                train_steps[name] += 1
                _, predicted = torch.max(output, 1)
                train_total[name] += target.size(0)
                train_correct[name] += (predicted == target).sum().item()

        # 对loss进行平均，由于criterion的reduction为mean，所以不需要除以batch_size，而是除以steps，越是小batch size，steps越多
        train_losses = {
            name: train_losses[name] / train_steps[name] for name in batch_sizes.keys()
        }

        writer.add_scalars(
            "Training Loss",
            train_losses,
            epoch,
        )

        # 测试阶段
        test_losses = {name: 0 for name in batch_sizes.keys()}
        test_steps = {name: 0 for name in batch_sizes.keys()}
        test_correct = {name: 0 for name in batch_sizes.keys()}
        test_total = {name: 0 for name in batch_sizes.keys()}

        for name in batch_sizes.keys():
            models[name].eval()
            with torch.no_grad():
                for data, target in test_loaders[name]:
                    data, target = data.to(device), target.to(device)
                    output = models[name](data)
                    loss = criterion(output, target)
                    test_losses[name] += loss.item()
                    test_steps[name] += 1
                    _, predicted = torch.max(output, 1)
                    test_total[name] += target.size(0)
                    test_correct[name] += (predicted == target).sum().item()

        # 对loss进行平均，由于criterion的reduction为mean，所以不需要除以batch_size，而是除以steps，越是小batch size，steps越多
        test_losses = {
            name: test_losses[name] / test_steps[name] for name in batch_sizes.keys()
        }

        writer.add_scalars(
            "Test Loss",
            test_losses,
            epoch,
        )

        # 记录训练和测试准确率
        accuracy_dict = {
            f"train_{name}": correct / total
            for name, correct, total in zip(
                batch_sizes.keys(), train_correct.values(), train_total.values()
            )
        }
        accuracy_dict.update(
            {
                f"test_{name}": correct / total
                for name, correct, total in zip(
                    batch_sizes.keys(), test_correct.values(), test_total.values()
                )
            }
        )

        writer.add_scalars("Accuracy", accuracy_dict, epoch)

    writer.close()
