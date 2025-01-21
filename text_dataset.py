from torch.utils.data import Dataset
from torchtext.data.utils import get_tokenizer


# 构造一个用于读取数据的dataset
class NewDataset(Dataset):
    # data对象保存数据集中的原始数据,label 和 text(标题和主体）
    def __init__(self, data):
        super().__init__()
        # 定义list
        self.data_content = []
        # 分词器: 将文本转成小写，分割空格同时分割标点符号
        tokenizer = get_tokenizer("basic_english")

        for label, text in data: # text包含标题和主体, 且无法解包
            tokenized_text = tokenizer(text)
            self.data_content.append((tokenized_text, label))  # 确保label是整数

    def __len__(self):
        return len(self.data_content)

    def __getitem__(self, item):
        return self.data_content[item]