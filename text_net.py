import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence

# 定义神经网络
class TextCls(nn.Module):
    def __init__(self, vocab_size, embedding_dim, padding_idx, cls_num):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=padding_idx)
        self.fc = nn.Sequential(
            nn.Linear(embedding_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),  # 添加Dropout以防止过拟合

            nn.Linear(128, 64),
            nn.ReLU(),

            nn.Linear(64, 32),
            nn.ReLU(),

            nn.Linear(32, cls_num)
        )

    def forward(self, x):
        # 获取词嵌入向量：batch_size * seq_len * embedding_dim； 输入x为：[batch_size, seq_len]
        x_embed = self.embedding(x)  # [batch_size, seq_len, embedding_dim]

        # 创建mask: 1表示实际token，0表示padding
        mask = (x != 0).float()  # [batch_size, seq_len]
        mask = mask.unsqueeze(-1)  # [batch_size, seq_len, 1]

        # 使用mask处理embedding
        x_masked = x_embed * mask  # [batch_size, seq_len, embedding_dim]

        # 计算实际长度（非padding的token数量）
        lengths = torch.sum(mask, dim=1)  # [batch_size, 1, 1]

        # 求和后除以实际长度
        x_sum = torch.sum(x_masked, dim=1)  # [batch_size, embedding_dim]
        x_mean = x_sum / lengths

        out = self.fc(x_mean)
        return out

if __name__ == '__main__':
    vocab_size, embedding_dim, cls_num = 1000, 50, 4
    padding_idx = 0

    # 创建模型
    model = TextCls(vocab_size, embedding_dim, cls_num, padding_idx)

    # 创建输入数据（句子索引）
    sentences = [
        torch.tensor([1, 3, 4, 2]),  # 长度为4
        torch.tensor([5, 7, 9, 1, 5]),  # 长度为5
        torch.tensor([10, 4, 2])  # 长度为3
    ]

    # 使用pad_sequence填充句子
    inputs = pad_sequence(sentences, batch_first=True, padding_value=padding_idx)

    # 打印填充后的句子
    print(f"填充后的句子:\n{inputs}")

    # 前向传播
    outputs = model(inputs)
    print(f"模型的输出:\n{outputs}")