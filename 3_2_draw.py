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

    log_dir = "runs/mnist_weight_decay"
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

    # 定义不同的权重衰减率
    weight_decays = {
        "wd_0": 0.0,  # 无权重衰减
        "wd_0.0001": 0.0001,  # 较小的权重衰减
        "wd_0.001": 0.001,  # 中等权重衰减
        "wd_0.01": 0.01,  # 较大的权重衰减
        "wd_0.1": 0.1,  # 很大的权重衰减
    }

    # 创建数据加载器(使用固定的batch size)
    batch_size = 64
    train_loader = DataLoader(
        training_data, batch_size=batch_size, shuffle=True, num_workers=16
    )
    test_loader = DataLoader(
        test_data, batch_size=batch_size, shuffle=True, num_workers=16
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 为每个权重衰减率创建独立的模型和优化器
    models = {name: MLP().to(device) for name in weight_decays.keys()}
    optimizers = {
        name: torch.optim.SGD(
            model.parameters(),
            lr=0.01,
            momentum=0.9,  # 添加动量
            weight_decay=wd,  # 设置不同的权重衰减率
        )
        for name, (model, wd) in zip(
            models.keys(), zip(models.values(), weight_decays.values())
        )
    }
    criterion = nn.CrossEntropyLoss()

    # 训练循环
    epochs = 50
    for epoch in tqdm(range(epochs)):
        # 训练阶段
        train_losses = {name: 0 for name in weight_decays.keys()}
        train_steps = {name: 0 for name in weight_decays.keys()}
        train_correct = {name: 0 for name in weight_decays.keys()}
        train_total = {name: 0 for name in weight_decays.keys()}

        for name in weight_decays.keys():
            models[name].train()
            for data, target in train_loader:  # 使用相同的数据加载器
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
            name: train_losses[name] / train_steps[name]
            for name in weight_decays.keys()
        }

        writer.add_scalars(
            "Training Loss",
            train_losses,
            epoch,
        )

        # 测试阶段
        test_losses = {name: 0 for name in weight_decays.keys()}
        test_steps = {name: 0 for name in weight_decays.keys()}
        test_correct = {name: 0 for name in weight_decays.keys()}
        test_total = {name: 0 for name in weight_decays.keys()}

        for name in weight_decays.keys():
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
            name: test_losses[name] / test_steps[name] for name in weight_decays.keys()
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
                weight_decays.keys(), train_correct.values(), train_total.values()
            )
        }
        accuracy_dict.update(
            {
                f"test_{name}": correct / total
                for name, correct, total in zip(
                    weight_decays.keys(), test_correct.values(), test_total.values()
                )
            }
        )

        writer.add_scalars("Accuracy", accuracy_dict, epoch)

    writer.close()
