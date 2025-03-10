# MalSeqNet

### 运行环境

- GPU: NVIDIA GeForce RTX 4090 (24G)
- OS: Ubuntu 22.04.4 LTS
- Python 3.12.3
- PyTorch 2.5.1 with CUDA 12.4

### 安装

除了PyTorch外，还需要安装如下依赖：

```bash
pip install pandas
pip install tqdm
pip install loguru
pip install scikit-learn
```

### 运行

运行前需要解压原始数据集，并进行预处理，使用如下命令：

```bash
bash uncompress-data.sh # 也可以自行解压
python preprocess.py
```

使用如下命令进行测试：

```bash
bash evaluate.sh
```

该脚本将依次输出：

- 在Android7.1下不使用关键子序列的准确率
- 在Android9下不使用关键子序列的准确率
- 在Android7.1下使用关键子序列的准确率
- 在Android9下使用关键子序列的准确率
