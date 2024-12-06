import math
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

    log_dir = "runs/mnist_lr_schedulers"
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

    # 创建数据加载器
    batch_size = 128
    train_loader = DataLoader(
        training_data, batch_size=batch_size, shuffle=True, num_workers=24
    )
    test_loader = DataLoader(
        test_data, batch_size=batch_size, shuffle=True, num_workers=24
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 定义不同的学习率配置
    lr_configs = {
        "fixed": {"base_lr": 0.1},
        "exp": {"base_lr": 0.1, "gamma": 0.95},
        "step": {"base_lr": 0.1, "milestones": [20, 35], "gamma": 0.1},
        "poly": {"base_lr": 0.1, "power": 0.9},
        "linear": {"base_lr": 0.1},
        "cosine": {"base_lr": 0.1},
        "warmup": {"base_lr": 0.1, "warmup_epochs": 5},
    }

    # 为每种学习率策略创建独立的模型和优化器
    models = {name: MLP().to(device) for name in lr_configs.keys()}
    optimizers = {
        name: torch.optim.SGD(models[name].parameters(), lr=lr_configs[name]["base_lr"])
        for name in lr_configs.keys()
    }

    # 创建学习率调度器
    schedulers = {}
    epochs = 50
    for name, config in lr_configs.items():
        if name == "fixed":
            schedulers[name] = torch.optim.lr_scheduler.LambdaLR(
                optimizers[name], lambda epoch: 1
            )
        elif name == "exp":
            schedulers[name] = torch.optim.lr_scheduler.ExponentialLR(
                optimizers[name], gamma=config["gamma"]
            )
        elif name == "step":
            schedulers[name] = torch.optim.lr_scheduler.MultiStepLR(
                optimizers[name], milestones=config["milestones"], gamma=config["gamma"]
            )
        elif name == "poly":
            power = config["power"]
            schedulers[name] = torch.optim.lr_scheduler.LambdaLR(
                optimizers[name], lambda epoch, p=power: (1 - epoch / epochs) ** p
            )
        elif name == "linear":
            schedulers[name] = torch.optim.lr_scheduler.LambdaLR(
                optimizers[name], lambda epoch: 1 - epoch / epochs
            )
        elif name == "cosine":
            schedulers[name] = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizers[name], T_max=epochs
            )
        elif name == "warmup":
            warmup_epochs = config["warmup_epochs"]

            def warmup_lr(epoch, w_epochs=warmup_epochs):
                if epoch < w_epochs:
                    return epoch / w_epochs
                return 0.5 * (
                    1 + math.cos(math.pi * (epoch - w_epochs) / (epochs - w_epochs))
                )

            schedulers[name] = torch.optim.lr_scheduler.LambdaLR(
                optimizers[name], warmup_lr
            )

    criterion = nn.CrossEntropyLoss()

    for epoch in tqdm(range(epochs)):
        # 训练阶段
        train_losses = {name: 0 for name in lr_configs.keys()}
        train_steps = {name: 0 for name in lr_configs.keys()}
        train_correct = {name: 0 for name in lr_configs.keys()}
        train_total = {name: 0 for name in lr_configs.keys()}

        for name in lr_configs.keys():
            models[name].train()
            for data, target in train_loader:
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

            # 更新学习率
            schedulers[name].step()

            # 记录当前学习率
            writer.add_scalar(
                f"Learning Rate/{name}", schedulers[name].get_last_lr()[0], epoch
            )

        # 对loss进行平均，由于criterion的reduction为mean，所以不需要除以batch_size，而是除以steps，越是小batch size，steps越多
        train_losses = {
            name: train_losses[name] / train_steps[name] for name in lr_configs.keys()
        }

        writer.add_scalars(
            "Training Loss",
            train_losses,
            epoch,
        )

        # 测试阶段
        test_losses = {name: 0 for name in lr_configs.keys()}
        test_steps = {name: 0 for name in lr_configs.keys()}
        test_correct = {name: 0 for name in lr_configs.keys()}
        test_total = {name: 0 for name in lr_configs.keys()}

        for name in lr_configs.keys():
            models[name].eval()
            with torch.no_grad():
                for data, target in test_loader:
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
            name: test_losses[name] / test_steps[name] for name in lr_configs.keys()
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
                lr_configs.keys(), train_correct.values(), train_total.values()
            )
        }
        accuracy_dict.update(
            {
                f"test_{name}": correct / total
                for name, correct, total in zip(
                    lr_configs.keys(), test_correct.values(), test_total.values()
                )
            }
        )

        writer.add_scalars("Accuracy", accuracy_dict, epoch)

    writer.close()
