


'''
将owt和tiny_story的文本全部变为IDs，然后保存为本地的numpy ，类型是uint16
'''
from typing import List
import os
import numpy as np
from cs336_basics.tokenizers.tokenizer import Tokenizer

def save_token_ids(token_ids, filename, dtype=np.uint16):
    """
    保存 token_ids（可任意可迭代）为 numpy 数组并写入文件。

    :param token_ids: 一个一维可迭代，包含所有 token ID（整数）
    :param filename: 字符串，输出文件名，比如 'train_ids.npy'
    :param dtype: NumPy 整型数据类型，默认 np.uint16
    """
    # 转换为 numpy 数组
    arr = np.array(token_ids, dtype=dtype)
    # 保存为 .npy 文件
    np.save(filename, arr)
    print(f"Saved {len(arr)} token IDs to {filename} with dtype {dtype}")

def load_token_ids(filename: str, mmap_mode: str = None) -> np.ndarray:
    """
    从 .npy 文件中读取 token ID 数组。

    :param filename: 文件路径（例如 "train_ids.npy"）。
    :param mmap_mode: 如果非 None，则使用 memory‐map 模式读取。常用值为 'r'（只读）、'r+'（读写）、'c'（拷贝写）等。默认为 None，表示一次性读入内存。
    :return: 一个 NumPy 数组，dtype 应为 uint16（假设保存时使用 uint16）。
    """
    arr = np.load(filename, mmap_mode=mmap_mode)
    # 可选：校验 dtype
    if arr.dtype != np.uint16:
        raise ValueError(f"数组 dtype 是 {arr.dtype}，而不是预期的 uint16")
    return arr


def main():
    
    # 文本文件 
    owt_train_data_path = "/home/niu/code/cs336/assignment1-basics/data/owt_train.txt"
    owt_test_data_path = "/home/niu/code/cs336/assignment1-basics/data/owt_valid.txt"
    
    tiny_story_train_data_path = "/home/niu/code/cs336/assignment1-basics/data/TinyStoriesV2-GPT4-train.txt"
    tiny_story_test_data_path = "/home/niu/code/cs336/assignment1-basics/data/TinyStoriesV2-GPT4-valid.txt"
    
    # 准备编码器
    tiny_story_vocab_path = "/home/niu/code/cs336/assignment1-basics/cs336_basics/bpe/output/TinyStoriesV2-GPT4-train_optim_vocab_10000.pkl"
    tiny_story_merge_path = "/home/niu/code/cs336/assignment1-basics/cs336_basics/bpe/output/TinyStoriesV2-GPT4-train_optim_merges_10000.pkl"
    
    tiny_story_tokenizer = Tokenizer.from_files(
        vocab_filepath=tiny_story_vocab_path,
        merges_filepath=tiny_story_merge_path,
        special_tokens=["<|endoftext|>"],
    )
    
    owt_vocab_path = "/home/niu/code/cs336/assignment1-basics/cs336_basics/bpe/output/owt-train_optim_vocab_32000.pkl"
    owt_merge_path = "/home/niu/code/cs336/assignment1-basics/cs336_basics/bpe/output/owt-train_optim_merges_32000.pkl"
    
    owt_tokenizer = Tokenizer.from_files(
        vocab_filepath=owt_vocab_path,
        merges_filepath=owt_merge_path,
        special_tokens=["<|endoftext|>"],
    )
    
    
    # 进行tokenize
    tiny_story_train_text  = open(tiny_story_train_data_path, "r", encoding="utf-8").read()
    
    print(f"begin to tokenize tiny_story_train_text...")
    tiny_story_train_ids : List[int] = tiny_story_tokenizer.encode_parallel(tiny_story_train_text, os.cpu_count())
    
    save_token_ids(tiny_story_train_ids,"tiny_story_train_ids.npy", np.uint16)
    
    loaded_tiny_story_train_ids = load_token_ids("tiny_story_train_ids.npy")
    
    assert np.array_equal(loaded_tiny_story_train_ids, tiny_story_train_ids)
    print(f"loaded tiny story ids is {loaded_tiny_story_train_ids}")
    print(f"tiny_story_train_ids ids is {tiny_story_train_ids}")
    
    


if __name__ == "__main__":
    main()