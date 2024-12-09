import os

import requests
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from transformers import BertTokenizer


def get_time_machine_text():
    # 确保data目录存在
    data_dir = "./data"
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    file_path = os.path.join(data_dir, "time_machine.txt")

    # 如果文件已存在，直接读取
    if os.path.exists(file_path):
        print("发现本地文件，直接读取...")
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()

    # 如果文件不存在，从网络下载
    print("从Project Gutenberg下载文件...")
    url = "https://www.gutenberg.org/files/35/35-0.txt"
    try:
        response = requests.get(url)
        response.raise_for_status()
        text = response.text

        # 保存到本地
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(text)
        print("下载完成并保存到本地")

        return text
    except Exception as e:
        print(f"下载失败: {str(e)}")
        return None


def tokenize_text(text):
    """
    使用BERT tokenizer对文本进行分词

    Args:
        text (str): 输入文本

    Returns:
        tuple: (tokenizer, encoded_text)
    """
    # 使用BERT tokenizer
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    # 将文本分成行
    text_lines = [line.strip() for line in text.split("\n") if line.strip()]

    # 对文本进行编码
    encoded_text = []
    for line in text_lines:
        # 添加特殊标记并编码
        encoded = tokenizer.encode(
            line,
            add_special_tokens=True,  # 添加[CLS]和[SEP]标记
            return_tensors="pt",  # 返回PyTorch张量
        )
        encoded_text.append(encoded.squeeze(0))  # 移除batch维度

    return tokenizer, encoded_text


class TextDataset(Dataset):
    """文本数据集"""

    def __init__(self, encoded_text, seq_length=50):
        self.sequences = []

        # 将所有编码后的文本连接成一个长序列
        full_sequence = torch.cat(encoded_text)

        # 生成固定长度的序列
        for i in range(0, len(full_sequence) - seq_length):
            input_seq = full_sequence[i : i + seq_length]
            target_seq = full_sequence[i + 1 : i + seq_length + 1]
            self.sequences.append((input_seq, target_seq))

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx]


class LSTMModel(nn.Module):
    """LSTM语言模型"""

    def __init__(
        self, vocab_size, embedding_dim=256, hidden_dim=512, num_layers=2, dropout=0.5
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x, hidden=None):
        # x shape: (batch_size, seq_length)
        embeds = self.embedding(x)  # (batch_size, seq_length, embedding_dim)
        lstm_out, hidden = self.lstm(
            embeds, hidden
        )  # (batch_size, seq_length, hidden_dim)
        out = self.dropout(lstm_out)
        out = self.fc(out)  # (batch_size, seq_length, vocab_size)
        return out, hidden

    def init_hidden(self, batch_size, device):
        """初始化隐藏状态"""
        hidden_dim = self.lstm.hidden_size
        num_layers = self.lstm.num_layers
        return (
            torch.zeros(num_layers, batch_size, hidden_dim).to(device),
            torch.zeros(num_layers, batch_size, hidden_dim).to(device),
        )


if __name__ == "__main__":
    # 获取文本并分词
    text = get_time_machine_text()
    if text is not None:
        tokenizer, encoded_text = tokenize_text(text)

        # 创建数据集和数据加载器
        dataset = TextDataset(encoded_text, seq_length=50)
        dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=0)

        # 创建模型
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = LSTMModel(
            vocab_size=len(tokenizer), embedding_dim=256, hidden_dim=512, num_layers=2
        ).to(device)

        # 创建 TensorBoard 写入器
        log_dir = os.path.join("runs", "lstm_model")
        writer = SummaryWriter(log_dir)

        print(f"使用设备: {device}")
        print(f"词表大小: {len(tokenizer)}")
        print(f"数据集大小: {len(dataset)}")

        # 训练模型
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        num_epochs = 20
        total_loss = 0

        for epoch in range(num_epochs):
            model.train()

            for batch_idx, (inputs, targets) in enumerate(dataloader):
                inputs, targets = inputs.to(device), targets.to(device)

                # 初始化隐藏状态
                hidden = model.init_hidden(inputs.size(0), device)

                optimizer.zero_grad()
                output, hidden = model(inputs, hidden)
                loss = criterion(output.view(-1, output.size(-1)), targets.view(-1))
                loss.backward()

                # 梯度裁剪，防止梯度爆炸
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                optimizer.step()
                total_loss += loss.item()

                # 每100个批次打印一次信息
                if (batch_idx + 1) % 100 == 0:
                    avg_loss = total_loss / 100
                    print(
                        f"Epoch [{epoch+1}/{num_epochs}], "
                        f"Step [{batch_idx+1}/{len(dataloader)}], "
                        f"Loss: {avg_loss:.4f}"
                    )

                    writer.add_scalar(
                        "Training Loss", avg_loss, epoch * len(dataloader) + batch_idx
                    )
                    total_loss = 0

        # 生成文本
        model.eval()
        seed_text = "The time machine"
        tokens = tokenizer.encode(seed_text, return_tensors="pt").to(device)

        with torch.no_grad():
            for _ in range(100):
                # 获取输入序列，最多取最后50个token
                input_seq = tokens[:, -50:] if tokens.size(1) > 50 else tokens
                # 初始化隐藏状态，batch_size为1
                hidden = model.init_hidden(1, device)
                # 前向传播，获取输出和新的隐藏状态
                output, hidden = model(input_seq, hidden)
                # 获取下一个token的logits，并进行缩放
                next_token_logits = output[0, -1, :] / 1.0
                # 计算下一个token的概率分布
                probs = torch.softmax(next_token_logits, dim=-1)
                # 从概率分布中采样下一个token
                next_token = torch.multinomial(probs, num_samples=1)
                # 将新生成的token添加到tokens中
                tokens = torch.cat([tokens, next_token.unsqueeze(0)], dim=1)
                # 如果生成的token是结束标记，则停止生成
                if next_token.item() == tokenizer.eos_token_id:
                    break

        generated_text = tokenizer.decode(tokens[0], skip_special_tokens=True)
        print("\n生成的文本:")
        print(generated_text)
