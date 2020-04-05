Knowledge-driven Dialogue
=============================
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This is a pytorch implementation of generative-based model for knowledge-driven dialogue

## Requirements

* cuda=9.0
* cudnn=7.0
* python>=3.6
* pytorch>=1.0
* tqdm
* numpy
* nltk
* scikit-learn

## Quickstart

### Step 1: Preprocess the data

运行tools下的[数据集名称].py即可得到.pt格式的数据集文件，例如

```
python tools/personachat.py
```

### Step 2: Train the model

运行train_[模型名称]_[数据集名称].sh即可训练，例如

```bash
sh train_seq2seq_personachat.sh
```

### Step 3: Test the Model

运行train_[模型名称]_[数据集名称].sh即可测试，例如

```bash
sh test_seq2seq_personachat.sh
```
