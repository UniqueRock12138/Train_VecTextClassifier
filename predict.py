import torch
from torchtext.datasets import AG_NEWS

from torchtext.data.utils import get_tokenizer
from text_net import TextCls
from text_dataset import NewDataset
from train import build_vocab


# 加载模型
def load_model(model_path, vocab_size, embed_dim, padding_idx, num_classes, device):
    model = TextCls(vocab_size, embed_dim, padding_idx, num_classes).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()  # 设置模型为评估模式
    return model


# 进行预测
def predict(text, model, text_vocab, device):
    tokenizer = get_tokenizer("basic_english")  # 使用基本的英语分词器
    tokens = tokenizer(text)  # 将文本分词
    text_indices = [text_vocab[token] for token in tokens]  # 将词转换为索引
    text_tensor = torch.tensor(text_indices, dtype=torch.long).unsqueeze(0).to(device)  # 转换为张量并添加批次维度

    with torch.no_grad():  # 禁用梯度计算
        output = model(text_tensor)  # 通过模型进行前向传播
        predicted_label = torch.argmax(output, dim=1).item()  # 获取预测的标签

    return predicted_label


if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # 选择设备

    # 假设您已经有一个训练集来构建词汇表
    train_data, _ = AG_NEWS(root='./', split=("train", "test"))  # 加载AG_NEWS数据集
    train_dataset = NewDataset(train_data)  # 创建数据集对象
    text_vocab = build_vocab(train_dataset)  # 构建词汇表

    # 模型参数
    vocab_size = len(text_vocab)  # 词汇表大小
    embed_dim = 128  # 嵌入维度
    num_classes = 4  # 类别数量
    padding_idx = text_vocab['<pad>']  # 填充索引

    # 加载模型
    model_path = "best_model.pt"  # 模型文件路径
    model = load_model(model_path, vocab_size, embed_dim, padding_idx, num_classes, device)  # 加载模型

    # 输入文本进行预测
    input_text = "Your input text here"  # 输入文本
    predicted_label = predict(input_text, model, text_vocab, device)  # 进行预测

    print(f"预测的标签是: {predicted_label}")  # 输出预测结果
