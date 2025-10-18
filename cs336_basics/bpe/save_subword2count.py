import logging
import regex
import json
import pickle
import multiprocessing as mp
from typing import Dict, Tuple, List
from tqdm import tqdm

from pretokenization_example import find_chunk_boundaries



INITIAL_VOCAB_SIZE = 256
PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
NUM_CHUNKS = 10000





#  copy from the optimize_bpe.py  辅助函数
def split_on_special_tokens(text: str, special_tokens: List[str]) -> List[str]:
    """
    把 text 按照 special_tokens 拆分，返回不包含 special token 的各个子段。
    空字符串片段会被过滤掉。
    """
    # 用 re.escape 把每个 special token 转义，确保正则安全
    escaped = [regex.escape(tok) for tok in special_tokens]
    # 构造拆分正则：任意一个 special token
    # 用捕获组把 matched token 保留下来（可选，看你要不要保留 token 本身）
    pattern = "(" + "|".join(escaped) + ")"
    parts = regex.split(pattern, text)
    # parts 中包含拆分出的文本片段 & 拆分符号本身（因为用了捕获组）
    docs: List[str] = []
    cur: List[str] = []
    for seg in parts:
        if seg in special_tokens:
            # 遇到 special token，把当前 accumulated 文本作为一个 doc 段
            docs.append("".join(cur))
            cur = []
        else:
            cur.append(seg)
    # 最后剩下的也算一个 doc
    if cur:
        docs.append("".join(cur))
    # 过滤空字符串
    docs = [d for d in docs if d != ""]
    return docs
def pretokenize_mp(input_path, special_tokens, PAT, num_chunks=None):
    
    if num_chunks is None:
        num_chunks = mp.cpu_count()
        logger.info(f"Split the file into {num_chunks} chunks")
    else:
        logger.info(f"Split the file into {num_chunks} chunks")
    
    
    
    logger.info(f"start counting subwords...")
    subword2count : dict[tuple[bytes], int] = {}
    with open(input_path, "rb") as f:
        boundaries = find_chunk_boundaries(f, num_chunks, b"<|endoftext|>")
        
        for start, end in tqdm(zip(boundaries[:-1], boundaries[1:]), total=len(boundaries)-1, desc="Pretokenizing chunks..."):
            logger.info(f"Pretokenizing chunk {start} to {end}")
            f.seek(start)
            chunk = f.read(end - start).decode("utf-8", errors="ignore")
            
            splits = split_on_special_tokens(text=chunk, special_tokens=special_tokens)
            for text in splits:
                for token in regex.finditer(PAT, text):
                    token_str = token.group(0)
                    token_bytes = token_str.encode("utf-8")
                    token_subwords = tuple([token_bytes[i:i+1] for i in range(len(token_bytes))])
                    subword2count[token_subwords] = subword2count.get(token_subwords, 0) + 1
                
    logger.info(f"end counting subwords")
    return subword2count





# 保存
def save_subword2count_to_pickle(subword2count: Dict[Tuple[bytes, ...], int],
                              filepath: str,
                              protocol: int = pickle.HIGHEST_PROTOCOL) -> None:
    """
    将 dict[tuple[bytes], int] 序列化为 pickle 文件。
    :param subword2count: 待保存的字典，键为 tuple of bytes，值为 int。
    :param filepath: 保存路径（含文件名）。
    :param protocol: pickle 协议版本（默认最高）。
    """
    with open(filepath, 'wb') as f:
        pickle.dump(subword2count, f, protocol=protocol)


# 加载
def load_subword2count_from_pickle(filepath: str) -> Dict[Tuple[bytes, ...], int]:
    """
    从 pickle 文件加载回 dict[tuple[bytes], int]。
    **警告：仅当你信任该文件来源时才使用。**
    :param filepath: pickle 文件路径。
    :return: 加载后的 dict。
    """
    with open(filepath, 'rb') as f:
        data = pickle.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"load data from pickle file error: {filepath}")
    return data 



# 主要的逻辑
def save_subword2count(
    input_path: str,
    special_tokens: list[str], 
    save_path: str,
) -> None:
    '''
        训练一个带有预分词的BPE模型
    '''
    
    # 初始化
    # build initial vocabulary
    vocab = {}
    for i in range(INITIAL_VOCAB_SIZE):
        vocab[i] = bytes([i])
    # add special tokens to vocabulary
    for token in special_tokens:
        if token not in vocab.values():
            vocab[len(vocab)] = token.encode("utf-8")
    
    merges: List[tuple[bytes, bytes]] = []
            
    logger.info(f"start pre-tokenization...")
    subword2count : dict[tuple[bytes], int] = pretokenize_mp(input_path=input_path, special_tokens=special_tokens, PAT=PAT, num_chunks=NUM_CHUNKS)
    logger.info(f"end pre-tokenization")
    
    
    logger.info(f"start saving subword2count to pickle...")
    save_subword2count_to_pickle(subword2count, save_path)
    logger.info(f"end saving subword2count to pickle success: {save_path}")
    
    logger.info(f"start loading subword2count from pickle for safe check...")
    subword2count_loaded = load_subword2count_from_pickle(save_path)
    logger.info(f"end loading subword2count from pickle for safe check...")
    assert subword2count_loaded == subword2count, "subword2count_loaded and subword2count are not the same"
    logger.info(f"subword2count_loaded and subword2count are the same")
    
    
    










#  测试代码================================================================================
import argparse
import time
from pprint import pformat
from datetime import datetime

def format_time(td_seconds: float) -> str:
    days, rem = divmod(td_seconds, 86400)
    hours, rem = divmod(rem, 3600)
    minutes, seconds = divmod(rem, 60)
    parts = []
    if days:
        parts.append(f"{int(days)}d")
    if hours:
        parts.append(f"{int(hours)}h")
    if minutes:
        parts.append(f"{int(minutes)}m")
    parts.append(f"{seconds:.2f}s")
    return " ".join(parts)

if __name__ == "__main__":
    
    # args
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus_path", type=str, default="/home/niu/code/cs336/assignment1-basics/data/owt_train.txt")
    parser.add_argument("--special_tokens", type=list, default=["<|endoftext|>"])
    parser.add_argument("--logs_save_path", type=str, default="./logs/save_owt_train_subword2count.log")
    parser.add_argument("--save_path", type=str, default="./output/owt_train_subword2count.json")
    args = parser.parse_args()

    # logger
    logging.basicConfig(
        filename=args.logs_save_path,
        level=logging.INFO,
        filemode='a',
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)

    # logging.disable(logging.INFO)
    logger.info(f"save subword2count arguments:\n{pformat(vars(args))}")

    start_time = time.time()
    logger.info(f"start time: {datetime.fromtimestamp(start_time).strftime('%Y-%m-%d %H:%M:%S')}")
    
    save_subword2count(args.corpus_path,
                       args.special_tokens,
                       args.save_path)
    end_time = time.time()
    logger.info(f"end time: {datetime.fromtimestamp(end_time).strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"time cost(s): {format_time(end_time - start_time)}")