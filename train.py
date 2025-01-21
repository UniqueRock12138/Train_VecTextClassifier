import os

import torch
import torch.nn as nn
from torchtext.vocab import build_vocab_from_iterator
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchtext.datasets import AG_NEWS
from tqdm import tqdm

from text_net import TextCls
from text_dataset import NewDataset


# 基于dataset，创建一个词表
def build_vocab(dataset):
    # unk表示未知词，pad标识填充词
    special = ['<unk>', '<pad>']
    # 获取文本序列
    text_iter = map(lambda x: x[0], dataset)  # dataset->(text, label)
    # 建立文本词汇表
    text_vocab = build_vocab_from_iterator(text_iter, min_freq=2, specials=special)
    # '<unk>' 被设置为词汇表的默认索引。这意味着在将文本转换为索引时，任何不在词汇表中的词都会自动映射到 '<unk>' 的索引,而不是抛出错误或返回无效值。
    text_vocab.set_default_index(text_vocab['<unk>'])
    # 查看词汇表
    print(f"训练集词汇表的长度为：{len(text_vocab)}")
    print(f"{dataset}的词汇表是(前10个)".center(70, "-"))
    stoi = text_vocab.get_stoi()  # 获取词汇表中的所有词汇和对应的索引
    for idx, (word, index) in enumerate(stoi.items()):
        if idx == 10:  # 只打印前十个
            break
        print(f"{word}: {index}")
    print("".center(70, "-"))
    # 打印训练集词表的长度
    return text_vocab


# 将一个批次的数据转换为适合模型输入的格式
"""
主要功能：
    文本转换为索引：将每个文本中的词转换为词汇表中的索引。
    填充序列：将每个批次的文本序列填充到相同的长度。
    标签处理：将标签转换为张量，并调整标签的范围。
"""


def collate_batch(batch_data, text_vocab):
    text_list = list()
    labels = list()
    # 遍历每个批次的数据
    for text, label in batch_data:
        # 将文本中的每个词转换为词汇表中的索引
        text_tokens = [text_vocab[token] for token in text]
        # 将索引列表转换为张量
        text_tensor = torch.tensor(text_tokens, dtype=torch.long)
        # 将文本张量添加到列表中
        text_list.append(text_tensor)
        # 将标签转换为张量，并调整标签范围
        labels.append(torch.tensor(label - 1, dtype=torch.long))  # 标签：1,2,3,4 -> 0,1,2,3

    # 获取"<pad>"的索引
    padding_idx = text_vocab['<pad>']
    # 使用 pad_sequence 将每个句子填充成相同长度的句子
    text_padded = pad_sequence(text_list, batch_first=True, padding_value=padding_idx)
    # 将标签列表转换为张量
    labels_tensor = torch.tensor(labels)
    # 返回填充后的文本张量和标签张量
    return text_padded, labels_tensor


if __name__ == '__main__':
    writer = SummaryWriter("logs")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    train_data, test_data = AG_NEWS(root='./', split=("train", "test"))
    train_dataset = NewDataset(train_data)
    test_dataset = NewDataset(test_data)
    # 创建训练集词表
    trian_vocab = build_vocab(train_dataset)
    # 不需要为测试集单独构建词汇表，使用训练集的词汇表 trian_vocab 来处理测试数据。
    # test_vocab = build_vocab(test_dataset)

    # 对于不同长度的文本进行填充
    collate = lambda batch_data: collate_batch(batch_data, trian_vocab)

    # 创建数据加载器 (DataLoader) 来批量读取数据
    # collate_fn=collate：这是一个函数，用于将一个批次的数据转换为模型可以接受的格式。
    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate)
    test_dataloader = DataLoader(test_dataset, batch_size=32, collate_fn=collate)

    # 定义模型的参数
    vocab_size = len(trian_vocab)
    embed_dim = 128
    num_classes = 4
    padding_idx = trian_vocab['<pad>']
    model = TextCls(vocab_size, embed_dim, padding_idx, num_classes).to(device)
    optimizer = torch.optim.AdamW(model.parameters())
    loss_fn = nn.CrossEntropyLoss()

    best_test_acc = 0.0  # 初始化最佳准确率

    epochs = 100  # 训练轮次
    for epoch in range(epochs):
        # 训练阶段
        model.train()
        sum_loss = 0
        for text, label in tqdm(train_dataloader, desc="Training"):
            text = text.to(device)
            label = label.to(device)
            # 前向传播
            out = model(text)
            loss = loss_fn(out, label)

            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            sum_loss += loss.item()

        # 计算平均损失
        train_avg_loss = sum_loss / len(train_dataloader)

        # 测试阶段
        model.eval()
        test_sum_acc = 0
        with torch.no_grad():
            for text, label in tqdm(test_dataloader, desc="Testing"):
                text = text.to(device)
                label = label.to(device)
                # 前向传播
                out = model(text)
                preds = torch.argmax(out, dim=1)

                # 计算准确率
                acc = torch.mean((preds == label).float())
                test_sum_acc += acc.item()

        # 计算平均准确率
        test_avg_acc = test_sum_acc / len(test_dataloader)

        print(f"Epoch:{epoch},Train Loss:{train_avg_loss:.4f},Test Accuracy:{test_avg_acc:.4f}")
        writer.add_scalar("train_avg_loss", train_avg_loss, epoch)
        writer.add_scalar("test_avg_acc", test_avg_acc, epoch)

        # 保存最优模型和最后一轮的模型
        if test_avg_acc > best_test_acc:
            best_test_acc = test_avg_acc
            model_save_path = "best_model.pt"
            torch.save(model.state_dict(), model_save_path)
            print(f"保存第{epoch}轮的模型, 其测试准确率为{test_avg_acc:.4f}")
        if epoch == epochs - 1:
            model_save_path = "last_model.pt"
            torch.save(model.state_dict(), model_save_path)
            print(f"保存最后一轮的模型, 其测试准确率为{test_avg_acc:.4f}")
