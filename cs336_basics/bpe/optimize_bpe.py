
import regex
import logging  
from tqdm import tqdm
import os
import sys
import copy 
from typing import Dict, Tuple, List
import multiprocessing as mp
root = os.path.dirname(os.path.dirname(__file__))
if root not in sys.path:
    sys.path.insert(0, root)

from cs336_basics.bpe.utils import InvertIndex, BucketMaxSD
from cs336_basics.bpe.pretokenization_example import find_chunk_boundaries

from cs336_basics.utils import save_bytes_dict_to_pickle, save_merges_to_pickle
# logger
logging.basicConfig(
    filename='./logs/TinyStoriesV2-GPT4-train_optim_10000.log',
    level=logging.INFO,
    filemode='a',
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
# logging.disable(logging.INFO)


# constants
INITIAL_VOCAB_SIZE = 256   # number of initial tokens (byte values) ，do not contain special tokens
PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

NUM_CHUNKS = 1000
NUM_PRE_TOKEN_PROCESS = os.cpu_count()





# =============================================================================================================================
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

def print_split_docs(splited_text: list[str]):
    # 过滤空文档（如果有的话）
    docs = [doc for doc in splited_text if doc.strip() != ""]
    # 用两个换行分隔每个文档
    joined = "\n\n".join(docs)
    logger.info("Split docs:\n%s", joined)
    

# def get_most_appear_pair(d: Dict[Tuple[bytes, bytes], int]):
#     max_freq = max(d.values())
#     candidates = [pair for pair, freq in d.items() if freq == max_freq]
    
#     # 直接比较元组，Python的元组比较就是先比较第一个元素，再比较第二个元素
#     best_pair = max(candidates)
#     return (best_pair, max_freq)



def pre_tokenize_chunk(args: tuple[str, list[str], str])-> dict[tuple[bytes], int] :
    
    chunk, special_tokens, PAT = args
    # print(f" ======  \n my chunk length: {chunk} \n====== ")
    splits = split_on_special_tokens(text=chunk, special_tokens=special_tokens)
    # print(f"my splits length: {len(splits)}")
    local_subword2count : dict[tuple[bytes], int] = {}
    for text in splits:
        for token in regex.finditer(PAT, text):
            token_str = token.group(0)
            token_bytes = token_str.encode("utf-8")
            
            token_subwords = tuple([token_bytes[i:i+1] for i in range(len(token_bytes))])
            local_subword2count[token_subwords] = local_subword2count.get(token_subwords, 0) + 1
    return local_subword2count


def _chunk_generator(f, boundaries, special_tokens, PAT):
    for start, end in zip(boundaries[:-1], boundaries[1:]):
        f.seek(start)
        chunk = f.read(end - start).decode("utf-8", errors="ignore")
        yield (chunk, special_tokens, PAT)

def pretokenize_mp(input_path, special_tokens, PAT, num_chunks=None):
    
    if num_chunks is None:
        num_chunks = mp.cpu_count()
        logger.info(f"Split the file into {num_chunks} chunks")
    else:
        logger.info(f"Split the file into {num_chunks} chunks")

    logger.info(f"start pre-tokenization...")
    before_pretokenization_time = time.time()
    logger.info(f"pretokenization start time: {format_time(before_pretokenization_time)}")
    subword2count : dict[tuple[bytes], int] = {}
    
    with open(input_path, "rb") as f:
        
        boundaries = find_chunk_boundaries(f, num_chunks, b"<|endoftext|>")
        
        real_chunk_size = len(boundaries) - 1
        processes_to_use = min(mp.cpu_count(), NUM_PRE_TOKEN_PROCESS)
        processes_to_use = min(processes_to_use, real_chunk_size)     
        
        logger.info(f"processes_to_use: {processes_to_use} and real chunksize we use is :  {real_chunk_size}")
        
        chunk_generator = _chunk_generator(f, boundaries, special_tokens, PAT)
        pbar = tqdm(total=real_chunk_size, desc="Pre tokenize chunks...")
        for i in range(0, real_chunk_size, processes_to_use):
            start = i
            end = min(i + processes_to_use, real_chunk_size)
            chunk_args_i = []
            # print(f"start is {start}, end is {end}")
            for _ in range(start, end):
                chunk :Tuple = next(chunk_generator)
                chunk_args_i.append(chunk)
            
            real_processes_to_use = len(chunk_args_i)
            with mp.Pool(processes=real_processes_to_use) as pool:
                results_iterator = pool.imap(pre_tokenize_chunk, chunk_args_i)
                for local_subword2count in tqdm(results_iterator, total=len(chunk_args_i), desc=f"Processing {len(chunk_args_i)} mini chunks", leave=False):
                    for k, v in local_subword2count.items():
                        subword2count[k] = subword2count.get(k, 0) + v
            pbar.update(len(chunk_args_i))        
    # with mp.Pool(processes=processes_to_use) as pool:
    #     # print(f"Starting pre-tokenization with {processes_to_use} processes on {len(chunk_args)} chunks...")
        
    #     # Use imap_unordered for memory efficiency. It returns an iterator.
    #     results_iterator = pool.imap(pre_tokenize_chunk, chunk_args)
        
    #     # Iterate through results as they complete and merge them one by one.
    #     for local_subword2count in tqdm(results_iterator, total=len(chunk_args), desc="Processing chunks"):
    #         # print(f"local_subword2count size: {len(local_subword2count)}")
    #         for k, v in local_subword2count.items():
    #             subword2count[k] = subword2count.get(k, 0) + v
    
    
    # print(f"subword2count size: {len(subword2count)}")
    after_pretokenization_time = time.time()
    logger.info(f"pretokenization end time: {format_time(after_pretokenization_time)}")
    logger.info(f"pretokenization time cost: {format_time(after_pretokenization_time - before_pretokenization_time)}")
             
    return subword2count


def train_bpe(
    input_path: str,
    vocab_size: int,
    special_tokens: list[str],
    **kwargs    
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
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

    
    # read input text
    # with open(input_path, "r") as f:
    #     corpus = f.read()
    
    # # 按照<|endoftext|> 进行切分
    # splited_text = split_on_special_tokens(text=corpus, special_tokens=special_tokens)
    
    
    
    # subword2count : dict[tuple[bytes], int] = {}
    # for text in splited_text:
    #     # print(f"Processing text segment (length {len(text)}): {repr(text)}\n")
    #     token_iter = regex.finditer(PAT, text)
    #     for token in token_iter:
    #         token_str = token.group(0)
    #         token_bytes = token_str.encode("utf-8")
    #         token_subwords =  tuple([token_bytes[i:i+1] for i in range(len(token_bytes))])
    #         subword2count[token_subwords] = subword2count.get(token_subwords, 0) + 1
    
    
    # 这里需要构建一个倒排索引，（b1, b2） -> subword2count.key
    inverted_index = InvertIndex(subword2count)
    
    # 构建整体键值对的计数字典
    pair_count = BucketMaxSD(logger=logger)
    for sub_word_bytes, sub_word_count in subword2count.items():
        # logger.info(f" sub_word_bytes: {sub_word_bytes} count: {sub_word_count}")
        for b1, b2 in zip(sub_word_bytes, sub_word_bytes[1:]):
            pair_count.add((b1,b2), sub_word_count)
    num_merges = vocab_size - len(vocab)
    pbar = tqdm(total=num_merges, desc="BPE merge steps")
    while len(vocab) < vocab_size:  
        # 根据pair_count 找到出现频率最高的pair
        most_pairs = pair_count.get_most_appear_pair()
              
        # 更新 vocab，加入新合并的子词
        new_bytes =  most_pairs[0][0] + most_pairs[0][1]
        vocab[len(vocab)] = new_bytes
        
        # 记录合并操作
        merges.append(most_pairs[0])
        
        # 删除已经处理过的 pair_count，并且将新的 pair_count 也计算出来
        # if epoch ==0:
        #     print(f"most_pairs[0] is {most_pairs[0]} | inverted_index.get_sub_words_bytes_index(most_pairs[0]) is {inverted_index.get_sub_words_bytes_index(most_pairs[0])}")
        cor_index_list = copy.copy(inverted_index.get_sub_words_bytes_index(most_pairs[0]))
        for index in cor_index_list:
            # with open("123.txt", mode='a') as f:
            #         print(f'inverted_index {index} : {inverted_index.get_sub_words_bytes(index)}\n', file=f)

            # print(f"index is {index}")
            old_sub_word_bytes = inverted_index.get_sub_words_bytes(index)
            new_sub_word_bytes = []
            i = 0
            while i < len(old_sub_word_bytes):
                if i < len(old_sub_word_bytes) - 1 and (old_sub_word_bytes[i], old_sub_word_bytes[i+1]) == most_pairs[0]:
                    new_sub_word_bytes.append(most_pairs[0][0] + most_pairs[0][1])
                    i += 2
                else:
                    new_sub_word_bytes.append(old_sub_word_bytes[i])
                    i += 1
            
            # 得到了新的new_sub_word_bytes，然后进行善后处理。
            # print(f"== Old_sub_word_bytes is {old_sub_word_bytes} \n ==new sub word bytess {new_sub_word_bytes}")

            sub_word_bytes_count = subword2count[tuple(old_sub_word_bytes)]
            # 更新subword2count
        
            # print(f"new sub word bytes is {new_sub_word_bytes}  old sub word bytes i {old_sub_word_bytes}")
            if new_sub_word_bytes != old_sub_word_bytes:
                subword2count[tuple(new_sub_word_bytes)] = subword2count.get(tuple(new_sub_word_bytes), 0) + sub_word_bytes_count
                del subword2count[tuple(old_sub_word_bytes)]
            else:
                continue

            # 更新inverted_index
            j = 0
            # print(f"most_pairs is {most_pairs[0]}")
            # print(f"old_sub_word_bytes is {old_sub_word_bytes}")
            while j < len(old_sub_word_bytes):
                if j < len(old_sub_word_bytes) - 1 and (old_sub_word_bytes[j], old_sub_word_bytes[j+1]) == most_pairs[0]:   
                    if j - 1 >= 0:
                        pre_pairs = (old_sub_word_bytes[j - 1], old_sub_word_bytes[j])
                        # print(f"pre_pairs is {pre_pairs}")
                        inverted_index.delete_index(pre_pairs, index, new_sub_word_bytes) 
                    if j + 2 <= len(old_sub_word_bytes) - 1:
                        post_pairs = (old_sub_word_bytes[j + 1], old_sub_word_bytes[j + 2])
                        # print(f"delete post pair {post_pairs}")
                        inverted_index.delete_index(post_pairs, index, new_sub_word_bytes)
                    # 删除most_pairs
                    inverted_index.delete_index(most_pairs[0], index, new_sub_word_bytes)
                    j += 2
                else:
                    j += 1   
            # 将索引特替换为最新的
            inverted_index.subscribe_invertindex(index, new_sub_word_bytes)
            # with open("inverted_index_deleted.txt", "w", encoding="utf-8") as f:
            #     print(" delete inverted_index : \n", file=f)
            #     print(inverted_index, file=f)
            #     print("\n\n", file=f)
                
                
            n = 0
            while n < len(new_sub_word_bytes):
                if new_sub_word_bytes[n] == most_pairs[0][0] + most_pairs[0][1]:
                    if n - 1 >= 0:
                        pre_new_pairs = (new_sub_word_bytes[n-1], new_sub_word_bytes[n])
                        inverted_index.add_index(pre_new_pairs, index)
                    if n + 1 <= len(new_sub_word_bytes)-1:
                        post_new_pairs = (new_sub_word_bytes[n], new_sub_word_bytes[n+1])
                        inverted_index.add_index(post_new_pairs, index)
                    n += 1
                else:
                    n += 1
                    
                       
            # 更新pair_count
            # del pair_count[most_pairs[0]] 
            # 首先遍老的序列在pair_count中删除
            k= 0
            while k < len(old_sub_word_bytes):
                if k < len(old_sub_word_bytes) - 1 and (old_sub_word_bytes[k], old_sub_word_bytes[k+1]) == most_pairs[0]: 
                    if k - 1 >= 0:
                        pre_pairs = (old_sub_word_bytes[k - 1], old_sub_word_bytes[k])
                        # print(f"pre_pairs is {pre_pairs}")
                        pair_count.delete(pre_pairs, sub_word_bytes_count)
                            
                    if k + 2 <= len(old_sub_word_bytes) - 1:
                          
                        post_pairs = (old_sub_word_bytes[k + 1], old_sub_word_bytes[k + 2])
                        # print(f"post_pairs is {post_pairs}")
                        pair_count.delete(post_pairs, sub_word_bytes_count)
                            
                    pair_count.delete_all(most_pairs[0])    
                    k += 2
                else:
                    k += 1
            
            # 其次我们遍历新的序列，然后再次计算新加入的pair的count
            m = 0
            while m < len(new_sub_word_bytes):
                if new_sub_word_bytes[m] == most_pairs[0][0] + most_pairs[0][1]:
                    
                    # hehe的bug
                    if m - 1 >= 0:
                        pre_new_pairs = (new_sub_word_bytes[m-1], new_sub_word_bytes[m])
                        pair_count.add(pre_new_pairs, sub_word_bytes_count)
                    if m + 1 <= len(new_sub_word_bytes)-1:
                        post_new_pairs = (new_sub_word_bytes[m], new_sub_word_bytes[m+1])
                        pair_count.add(post_new_pairs, sub_word_bytes_count)
                    m += 1
                else:
                    m += 1
        if len(vocab) % 5 == 0:
            logger.info(f"vocab size {len(vocab)}/{vocab_size}, vocab size = {len(vocab)} merge size {len(merges)}")
        pbar.update(1)
    
    pbar.close()
    return vocab, merges    






# # 本地调试区域
# def bytes_to_safe_str(b: bytes) -> str:
#     try:
#         return b.decode('utf-8')
#     except UnicodeDecodeError:
#         # 不能utf-8 decode 的，用 repr 或 base64 表示
#         return repr(b)
# def save_merges_to_json(d: List[tuple[bytes, bytes]], filepath: str) -> None:
#     # 将 bytes 转为可写的 str
#     d_str = [ (bytes_to_safe_str(pairs[0]), bytes_to_safe_str(pairs[1])) for pairs in d]
#     # 把字典写入 JSON 文件
#     with open(filepath, "w", encoding="utf-8") as f:
#         json.dump(d_str, f, indent=2, ensure_ascii=False)
# def save_bytes_dict_to_json(d: Dict[int, bytes], filepath: str) -> None:
#     # 将 bytes 转为可写的 str
#     d_str = {k: bytes_to_safe_str(v) for k, v in d.items()}
#     # 把字典写入 JSON 文件
#     with open(filepath, "w", encoding="utf-8") as f:
#         json.dump(d_str, f, indent=2, ensure_ascii=False)

# import json
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
    mp.set_start_method('fork')
    # args
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus_path", type=str, default="/home/niu/code/cs336/assignment1-basics/data/TinyStoriesV2-GPT4-train.txt")
    parser.add_argument("--vocab_size", type=int, default=10000)
    parser.add_argument("--merges_save_path", type=str, default="./output/TinyStoriesV2-GPT4-train_optim_merges_10000.pkl")
    parser.add_argument("--vocab_save_path", type=str, default="./output/TinyStoriesV2-GPT4-train_optim_vocab_10000.pkl")
    parser.add_argument("--special_tokens", type=list, default=["<|endoftext|>"])
    args = parser.parse_args()
    

    logger.info(f"BPE training arguments:\n{pformat(vars(args))}")
    
    
    # exp
    start_time = time.time()
    logger.info(f"start time: {datetime.fromtimestamp(start_time).strftime('%Y-%m-%d %H:%M:%S')}")
    vocab, merges = train_bpe(args.corpus_path, args.vocab_size, args.special_tokens)
    end_time = time.time()
    logger.info(f"end time: {datetime.fromtimestamp(end_time).strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"time cost(s): {format_time(end_time - start_time)}")
    
    
    
    # save vocab to pickle
    save_bytes_dict_to_pickle(vocab, args.vocab_save_path)
    logger.info(f"save bytes dict to pickle success: {args.vocab_save_path}")
    
    # save merges to pickle
    save_merges_to_pickle(merges, args.merges_save_path)
    logger.info(f"save merges to pickle success: {args.merges_save_path}")