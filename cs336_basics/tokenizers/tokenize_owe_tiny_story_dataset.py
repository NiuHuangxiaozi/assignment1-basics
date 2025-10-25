


'''
将owt和tiny_story的文本全部变为IDs，然后保存为本地的numpy ，类型是uint16
'''
from typing import List
import os
import numpy as np
from cs336_basics.tokenizers.tokenizer import Tokenizer
from cs336_basics.bpe.pretokenization_example import find_chunk_boundaries
from tqdm import tqdm
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



def _chunk_generator(f, boundaries):
    for start, end in zip(boundaries[:-1], boundaries[1:]):
        f.seek(start)
        chunk = f.read(end - start).decode("utf-8", errors="ignore")
        yield chunk
  

def save_uint16_chunk(filename: str, ids_chunk: np.ndarray, mode: str = 'ab'):
    """
    将一个 uint16 类型的一维 numpy 数组追加保存到 .npy 文件里。
    说明：由于 .npy 格式头部只写一次，这里用 append 模式可能需注意： 
    如果文件不存在则使用 np.save 创建；如果已存在，则需要手动写入后续数据（见下面实现）。
    """
    if not os.path.exists(filename):
        # 文件不存在，直接用 np.save 写
        np.save(filename, ids_chunk.astype(np.uint16))
    else:
        # 文件已存在，直接追加二进制数据（跳过 .npy header）
        # 获取 header 长度
        with open(filename, 'rb+') as f:
            # 找到数据区起始位置：读取 header，再定位到数据开始位置
            # 简单方式：读取整个 existing array 然后 concat 再保存 —— 但可能内存大。
            # 这里一种较为廉价方案是借助 `np.lib.format` 模块了解 header size
            import numpy.lib.format as fmt
            f.seek(0, os.SEEK_END)
            pos = f.tell()
            # Append new data:
            f.write(ids_chunk.astype(np.uint16).tobytes())

def accumulate_and_save(outfile: str,
                        chunk_size: int = 10000):
    ids_list = []
    count = 0

    # 示例：模拟循环产生 int（你应替换为真实逻辑）
    for i in range(1, 1234567):  # 假设总共要产生 1,234,567 个 ints
        ids_list.append(i)
        count += 1
        if count >= chunk_size:
            # 达到一个 chunk，保存
            arr = np.array(ids_list, dtype=np.uint16)
            save_uint16_chunk(outfile, arr)
            # 重置
            ids_list.clear()
            count = 0

    # 循环结束后，如果还有剩余
    if ids_list:
        arr = np.array(ids_list, dtype=np.uint16)
        save_uint16_chunk(outfile, arr)
        
import argparse
import pprint
def main():
    
    
    # 准备实验参数
    parser = argparse.ArgumentParser(description="tokenize owt and tiny story dataset")
    parser.add_argument("--text_data_path", type=str, default="/home/niu/code/cs336/assignment1-basics/data/owt_train.txt")
    parser.add_argument("--tokenizer_vocab_path", type=str, default="/home/niu/code/cs336/assignment1-basics/cs336_basics/bpe/output/TinyStoriesV2-GPT4-train_optim_vocab_10000.pkl")
    parser.add_argument("--tokenizer_merge_path", type=str, default="/home/niu/code/cs336/assignment1-basics/cs336_basics/bpe/output/TinyStoriesV2-GPT4-train_optim_merges_10000.pkl")
    parser.add_argument("--id_save_path", type=str, default="owt_train_ids.npy")
    parser.add_argument("--save_ids_interval", type=int, default=100000)
    parser.add_argument("--num_chunks", type=int, default=100000)
    parser.add_argument("--special_tokens", type=list, default=["<|endoftext|>"])
    parser.add_argument("--is_chunk_save", action="store_true", default=False)
    args = parser.parse_args()
    pprint.pprint(args, indent = 2)
    
    
    
    if os.path.exists(args.id_save_path):
        try:
            os.remove(args.id_save_path)
            print(f"已删除文件：{args.id_save_path}")
        except Exception as e:
            print(f"删除文件时出错：{e}")
    else:
        print(f"文件不存在：{args.id_save_path}")
    
    # 定义tokenzier
    tokenizer = Tokenizer.from_files(
        vocab_filepath=args.tokenizer_vocab_path,
        merges_filepath=args.tokenizer_merge_path,
        special_tokens=["<|endoftext|>"],
    )
    
    
    # 进行tokenize
    print(f"begin to tokenize {args.text_data_path}...")
    ids_list = []
    check_ids_list = []
    
    # =======================
    with open(args.text_data_path, "rb") as f:
        boundaries = find_chunk_boundaries(
            f, args.num_chunks, args.special_tokens[0].encode("utf-8")
        )

        chunk_generator = _chunk_generator(f, boundaries)
        ids_list, check_ids_list = [], []

        # 外层进度条（整体进度）
        pbar = tqdm(total=len(boundaries) - 1, desc="Tokenizing chunks", position=0, leave=True)

        chunk_index = 0
        while chunk_index < len(boundaries) - 1:
            text_chunk = next(chunk_generator)

            # 内层 encode_parallel 使用自己的 tqdm，不干扰外层
            chunk_ids: List[int] = tokenizer.encode_parallel(
                text_chunk, os.cpu_count()  # 保证显示在下方
            )

            ids_list.extend(chunk_ids)
            check_ids_list.extend(chunk_ids)

            if args.is_chunk_save and len(ids_list) >= args.save_ids_interval:
                arr = np.array(ids_list, dtype=np.uint16)
                save_uint16_chunk(args.id_save_path, arr)
                ids_list.clear()

            chunk_index += 1
            pbar.update(1)  # 更新外层进度条

        pbar.close()

        if ids_list:
            arr = np.array(ids_list, dtype=np.uint16)
            save_uint16_chunk(args.id_save_path, arr)

    print(f"tokenize {args.text_data_path} end, save to {args.id_save_path}")
    
    
    # =======================
    loaded_data_ids = load_token_ids(args.id_save_path)
    # print(f"loaded data ids is {loaded_data_ids}")
    # print(f"check_ids_list ids is {check_ids_list}")
    
    
    print(f"loaded data ids length  is {len(loaded_data_ids)}")
    print(f"check_ids_list ids length is {len(check_ids_list)}")
    assert np.array_equal(check_ids_list, loaded_data_ids)
    
    


if __name__ == "__main__":
    main()