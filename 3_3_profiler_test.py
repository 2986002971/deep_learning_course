import os
import shutil

import torch
import torch.nn as nn
from torch.profiler import ProfilerActivity, record_function
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.models import resnet50
from torchvision.transforms import ToTensor


class ModifiedResNet50(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.resnet = resnet50()
        self.resnet.conv1 = nn.Conv2d(
            1, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, num_classes)

    def forward(self, x):
        return self.resnet(x)


def seed_everything(seed=42):
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_batch(data, target, model, optimizer, criterion, device):
    with record_function("train_batch_total"):
        with record_function("data_transfer"):
            data, target = data.to(device), target.to(device)

        with record_function("optimizer_zero_grad"):
            optimizer.zero_grad()

        with record_function("forward"):
            output = model(data)

        with record_function("loss"):
            loss = criterion(output, target)

        with record_function("backward"):
            loss.backward()

        with record_function("optimizer_step"):
            optimizer.step()

        return loss, output


def test_batch(data, target, model, criterion, device):
    with record_function("test_batch_total"):
        with record_function("data_transfer"):
            data, target = data.to(device), target.to(device)

        with record_function("forward"):
            output = model(data)

        with record_function("loss"):
            loss = criterion(output, target)

        return loss, output


if __name__ == "__main__":
    # 设置随机种子
    seed_everything()

    # 加载数据
    training_data = datasets.MNIST(
        root="data", train=True, download=True, transform=ToTensor()
    )
    test_data = datasets.MNIST(
        root="data", train=False, download=True, transform=ToTensor()
    )
    train_loader = DataLoader(
        training_data, batch_size=256, shuffle=True, num_workers=24
    )
    test_loader = DataLoader(test_data, batch_size=256, shuffle=True, num_workers=24)

    # 设置设备和模型
    assert (
        torch.cuda.is_available()
    ), "CUDA is not available. This script requires a GPU."
    device = "cuda"
    model = ModifiedResNet50().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    # 创建目录
    log_dir = "./runs/profiler"
    if os.path.exists(log_dir):
        shutil.rmtree(log_dir)
    os.makedirs(log_dir)

    # 修改 profiler 设置以支持 TensorBoard
    with torch.profiler.profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        schedule=torch.profiler.schedule(wait=1, warmup=3, active=3, repeat=2),
        on_trace_ready=torch.profiler.tensorboard_trace_handler(log_dir),
        record_shapes=True,
        profile_memory=True,
        with_modules=True,
    ) as prof:
        # 运行几个批次的训练
        for data, target in train_loader:
            prof.step()
            loss, output = train_batch(
                data, target, model, optimizer, criterion, device
            )
