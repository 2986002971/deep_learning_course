import torch
from torch import nn
from transformers import BertTokenizer


class LSTMModel(nn.Module):
    """LSTM语言模型 - 保持与训练时相同的结构"""

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
        embeds = self.embedding(x)
        lstm_out, hidden = self.lstm(embeds, hidden)
        out = self.dropout(lstm_out)
        out = self.fc(out)
        return out, hidden

    def init_hidden(self, batch_size, device):
        hidden_dim = self.lstm.hidden_size
        num_layers = self.lstm.num_layers
        return (
            torch.zeros(num_layers, batch_size, hidden_dim).to(device),
            torch.zeros(num_layers, batch_size, hidden_dim).to(device),
        )


def generate_text(
    model, tokenizer, seed_text, max_length=100, temperature=1.0, device="cuda"
):
    """生成文本"""
    model.eval()

    # 确保种子文本以句号结束
    if not seed_text.endswith("."):
        seed_text = seed_text + "."

    tokens = tokenizer.encode(seed_text, return_tensors="pt").to(device)

    with torch.no_grad():
        for _ in range(max_length):
            input_seq = tokens[:, -50:] if tokens.size(1) > 50 else tokens
            hidden = model.init_hidden(1, device)
            output, hidden = model(input_seq, hidden)
            next_token_logits = output[0, -1, :] / temperature
            probs = torch.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            # 将新token添加到序列中
            tokens = torch.cat([tokens, next_token.unsqueeze(0)], dim=1)

            # 检查是否生成了完整的句子
            generated_text = tokenizer.decode(tokens[0], skip_special_tokens=True)
            if (
                generated_text.endswith(".")
                or generated_text.endswith("!")
                or generated_text.endswith("?")
            ):
                if len(generated_text.split()) > 5:  # 确保句子足够长
                    break

            # 如果达到最大长度，强制在合适的位置结束
            if len(tokens[0]) >= max_length:
                generated_text = generated_text.rstrip() + "."
                break

    return generated_text


if __name__ == "__main__":
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # 加载tokenizer - 改为BERT tokenizer
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    # 创建模型
    model = LSTMModel(
        vocab_size=len(tokenizer), embedding_dim=256, hidden_dim=512, num_layers=2
    ).to(device)

    # 加载训练好的模型权重
    model.load_state_dict(torch.load("lstm_model.pth"))
    print("模型加载成功！")

    # 提示词列表
    prompts = [
        "time traveller",
        "traveller",
        "the time traveller says that",
        "when the time traveller returns to the garden",
        "the time traveller begins learning the language",
        "the time traveller determines that",
        "the time traveller knows he will have to stop",
        "when he wakes up",
        "the time traveller finds himself",
        "the time traveler tells the narrator to wait for him",
    ]

    # 为每个提示词生成文本
    print("\n开始生成文本...\n")
    for i, prompt in enumerate(prompts, 1):
        print(f"\n=== 提示词 {i}: '{prompt}' ===")
        # 使用不同的temperature值生成两个版本
        for temp in [0.7, 1.0]:
            generated = generate_text(
                model,
                tokenizer,
                prompt,
                max_length=200,
                temperature=temp,
                device=device,
            )
            print(f"\nTemperature {temp}:")
            print(generated)
            print("-" * 80)
