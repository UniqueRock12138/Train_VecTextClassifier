# 基于深度学习和词向量训练文本分类模型
说明：  
`dataset.zip`: 数据集压缩包，要进行解压，解压后保持目录层级：dataset/AG_NEWS。因为数据集AG_NEWS可能无法通过代码下载，所以要提前准备。  
`train.py`: 训练模型  
`predict.py`: 模型预测  


### 1 数据集：AG_NEWS
<img src="https://github.com/UniqueRock12138/train_TextClassificationModel/blob/7960d3aa169514803555943f3e053f71131bdccd/README/image03.png" width="400px">
<img src="https://github.com/UniqueRock12138/train_TextClassificationModel/blob/7960d3aa169514803555943f3e053f71131bdccd/README/image02.png" width="400px">

### 2 环境搭建
torchtext 与 torch 存在兼容问题，建议新建虚拟环境，使用以下版本。  
`pip install torch==2.1.2 torchvision==0.16.2 torchtext==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu121` 
