# CS336 大语言模型基础

本项目是CS336 Spring 2025课程的第一个作业，主要涉及大语言模型的基础组件，特别是**BPE（Byte Pair Encoding）**和**Token**的实现。

## 项目概述

本项目实现了一个完整的BPE分词器系统，包括：
- BPE训练算法
- Tokenizer编码/解码功能
- 特殊token处理
- 多进程优化

## BPE (Byte Pair Encoding) 简介

**BPE**是一种数据压缩算法，被广泛用于自然语言处理中的子词分词。其核心思想是：

1. **初始化**：从字节级别开始，每个字节作为一个基本token
2. **迭代合并**：统计文本中最频繁出现的字节对，将其合并为新的token
3. **重复过程**：持续合并直到达到预设的词汇表大小

### BPE的优势
- **处理未知词**：通过子词组合可以表示训练时未见过的词
- **平衡效率**：在词汇表大小和表示能力之间取得平衡
- **多语言支持**：基于字节级别，天然支持多语言文本

## Token 简介

**Token**是文本处理中的基本单位，在BPE分词器中：

### Token类型
- **字节token**：单个字节（0-255）
- **合并token**：通过BPE算法生成的子词
- **特殊token**：如`<|endoftext|>`等预定义的特殊标记

### Token处理流程
1. **预分词**：使用正则表达式将文本分割为词边界
2. **字节化**：将每个词转换为字节序列
3. **BPE合并**：应用训练好的合并规则
4. **ID映射**：将token映射为整数ID

## 项目结构

```
cs336/
├── assignment1-basics/          # 主要实现代码
│   ├── cs336_basics/
│   │   ├── bpe/                # BPE训练实现
│   │   └── tokenizers/         # Tokenizer实现
│   ├── learn_content/          # 学习材料和作业说明
│   └── tests/                  # 测试代码
└── readme.md                   # 本文件
```

## 核心功能

### BPE训练
- 支持多进程预分词优化
- 特殊token处理（如`<|endoftext|>`）
- 可配置的词汇表大小
- 内存高效的实现

### Tokenizer功能
- **编码**：将文本转换为token ID序列
- **解码**：将token ID序列还原为文本
- **流式处理**：支持大文件的流式编码
- **特殊token支持**：正确处理文档分隔符等

## 使用示例

```python
# 训练BPE模型
vocab, merges = train_bpe(
    input_path="data.txt",
    vocab_size=10000,
    special_tokens=["<|endoftext|>"]
)

# 创建tokenizer
tokenizer = Tokenizer(vocab, merges, special_tokens)

# 编码文本
tokens = tokenizer.encode("Hello world!")
print(tokens)  # [15496, 995, 0]

# 解码
text = tokenizer.decode(tokens)
print(text)  # "Hello world!"
```

## 技术特点

- **高效实现**：使用多进程优化预分词过程
- **内存友好**：支持大文件的流式处理
- **可扩展性**：易于添加新的特殊token
- **兼容性**：与GPT-2风格的BPE兼容

这个项目为理解现代大语言模型的分词机制提供了完整的实现参考。