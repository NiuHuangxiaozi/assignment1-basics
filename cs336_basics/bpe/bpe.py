import regex
import logging  
from typing import Dict, Tuple, List
import os
import sys

root = os.path.dirname(os.path.dirname(__file__))
if root not in sys.path:
    sys.path.insert(0, root)
    
from tests.common import gpt2_bytes_to_unicode

INITIAL_VOCAB_SIZE = 256   # number of initial tokens (byte values) ，do not contain special tokens
PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

logging.basicConfig(
    filename='bpe_debug.log',
    level=logging.INFO,
    filemode='a',
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
logging.disable(logging.INFO)



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
    

def get_most_appear_pair(d: Dict[Tuple[bytes, bytes], int]):
    max_freq = max(d.values())
    candidates = [pair for pair, freq in d.items() if freq == max_freq]
    
    # 直接比较元组，Python的元组比较就是先比较第一个元素，再比较第二个元素
    best_pair = max(candidates)
    return (best_pair, max_freq)


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
            
    
    # read input text
    with open(input_path, "r") as f:
        corpus = f.read()
    
    # 按照<|endoftext|> 进行切分
    splited_text = split_on_special_tokens(text=corpus, special_tokens=special_tokens)
    
    
    
    subword2count : dict[tuple[bytes], int] = {}
    for text in splited_text:
        # print(f"Processing text segment (length {len(text)}): {repr(text)}\n")
        token_iter = regex.finditer(PAT, text)
        for token in token_iter:
            token_str = token.group(0)
            token_bytes = token_str.encode("utf-8")
            token_subwords =  tuple([token_bytes[i:i+1] for i in range(len(token_bytes))])
            subword2count[token_subwords] = subword2count.get(token_subwords, 0) + 1
    
    logger.info(f"Initial subword2count size: {len(subword2count)}")
    for k, v in list(subword2count.items()):
        logger.info(f"  {repr(k)}: {v}")
    logger.info(f"=====================================================================")
    # b"</w>" 表示的是字节串, 这个本来可以家的，但是我们使用了pre-tokenization
    
    # merge util down
    while len(vocab) < vocab_size:
        pair_count : Dict[Tuple[bytes, bytes], int] = {}
        for sub_word_bytes, sub_word_count in subword2count.items():
            logger.info(f" sub_word_bytes: {sub_word_bytes} count: {sub_word_count}")
            for b1, b2 in zip(sub_word_bytes, sub_word_bytes[1:]):
                pair_count[(b1,b2)] = pair_count.get((b1,b2), 0) + sub_word_count
        # logger.info(f" pair_count size: {pair_count}")
        most_pairs = get_most_appear_pair(pair_count)
        logger.info(f" most_pairs: {most_pairs}")

        vocab[len(vocab)] = most_pairs[0][0] + most_pairs[0][1]
        # logger.info(f"new vocab size: {vocab} added {repr(most_pairs[0][0] + most_pairs[0][1])}") 
         
        # print(f"merge {most_pairs[0]} to {most_pairs[0][0]+most_pairs[0][1]}")
        # print(f"merge {repr(most_pairs[0])} count {repr(most_pairs[1])}")
        merges.append(most_pairs[0])
        
        # merge pairs
        new_subword2count : dict[tuple[bytes], int] = {}
        for sub_word_bytes, sub_word_count in subword2count.items():
            new_sub_word_bytes = []
            i = 0
            while i < len(sub_word_bytes):
                if i < len(sub_word_bytes) - 1 and (sub_word_bytes[i], sub_word_bytes[i+1]) == most_pairs[0]:
                    new_sub_word_bytes.append(most_pairs[0][0] + most_pairs[0][1])
                    i += 2
                else:
                    new_sub_word_bytes.append(sub_word_bytes[i])
                    i += 1
            new_subword2count[tuple(new_sub_word_bytes)] = new_subword2count.get(tuple(new_sub_word_bytes), 0) + sub_word_count
        del subword2count
        subword2count = new_subword2count
    
    return vocab, merges    






# 本地调试区域
def bytes_to_safe_str(b: bytes) -> str:
    try:
        return b.decode('utf-8')
    except UnicodeDecodeError:
        # 不能utf-8 decode 的，用 repr 或 base64 表示
        return repr(b)
import re
import ast
import base64
from typing import Optional
def safe_str_to_bytes(s: str) -> bytes:
    """
    逆向 bytes_to_safe_str 的函数 —— 尝试把输入的 str 转换回 bytes。
    如果 s 是有效 UTF-8 文本，则直接 encode；
    如果 s 是 repr(b) 返回的格式，则用 ast.literal_eval 解析；
    如果两个都不行，尝试 base64 或报错。
    """
    # 第一种可能：它是一个合法的 UTF-8 字符串（即原来 b.decode('utf-8') 成功时的输出）
    try:
        # 这里 encode 会保留原 bytes
        return s.encode('utf-8')
    except UnicodeEncodeError:
        pass

    # 第二种可能：它是 repr(b) 的形式，比如 "b'...'"
    # 那么我们尝试用 ast.literal_eval 解析这个字符串为一个 bytes 对象
    # 这种方式比较安全，不直接用 eval。
    # 需要先判断字符串是否以 "b'" 或 'b"' 开头
    if (s.startswith("b'") and s.endswith("'")) or (s.startswith('b"') and s.endswith('"')):
        try:
            b = ast.literal_eval(s)
            if isinstance(b, (bytes, bytearray)):
                return bytes(b)
        except (ValueError, SyntaxError):
            pass

    # 第三种可能：你在 repr 时用了 base64（如果你改写了 bytes_to_safe_str，用 base64 表示）
    # 例如你可能会做 base64.b64encode(b).decode('ascii')；你也可以在这里做 base64 解码尝试
    # 这里是一个简单假设：如果字符串看起来像 base64 编码（由 A-Z a-z 0-9 + / = 组成）且长度适合，
    # 我们可以尝试 base64 解码
    try:
        b = base64.b64decode(s, validate=True)
        return b
    except (base64.binascii.Error, ValueError):
        pass

    # 如果都不能，还原失败，抛出异常或返回空 bytes
    raise ValueError(f"无法把这个 safe string 还原为 bytes: {s!r}")

def save_merges_to_json(d: List[tuple[bytes, bytes]], filepath: str) -> None:
    # 将 bytes 转为可写的 str
    d_str = [ (bytes_to_safe_str(pairs[0]), bytes_to_safe_str(pairs[1])) for pairs in d]
    # 把字典写入 JSON 文件
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(d_str, f, indent=2, ensure_ascii=False)
def save_bytes_dict_to_json(d: Dict[int, bytes], filepath: str) -> None:
    # 将 bytes 转为可写的 str
    d_str = {k: bytes_to_safe_str(v) for k, v in d.items()}
    # 把字典写入 JSON 文件
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(d_str, f, indent=2, ensure_ascii=False)

import json
if __name__ == "__main__":
    corpus_path  = "/home/niu/code/cs336/assignment1-basics/tests/fixtures/tinystories_sample_5M.txt"
    vocab_size =  1000
    special_tokens = ["<|endoftext|>"]
    vocab, merges = train_bpe(corpus_path, vocab_size, special_tokens)
    # logger.info(f"vocab: \n{vocab}")
    save_merges_to_json(merges, "./merges.json")
    save_bytes_dict_to_json(vocab, "./vocab.json")
    
    
    reference_vocab_path = "/home/niu/code/cs336/assignment1-basics/tests/fixtures/train-bpe-reference-vocab.json"
    gpt2_byte_decoder = {v: k for k, v in gpt2_bytes_to_unicode().items()}
    
    with open(reference_vocab_path, encoding="utf-8") as f:
        gpt2_reference_vocab = json.load(f)
        reference_vocab = {
            gpt2_vocab_index: bytes([gpt2_byte_decoder[token] for token in gpt2_vocab_item])
            for gpt2_vocab_item, gpt2_vocab_index in gpt2_reference_vocab.items()
        }
    # logger.info(f"reference_vocab: \n{reference_vocab}")
    save_bytes_dict_to_json(reference_vocab, "./reference_vocab.json")
    
    
    reference_merges_path = "/home/niu/code/cs336/assignment1-basics/tests/fixtures/train-bpe-reference-merges.txt"
    with open(reference_merges_path, encoding="utf-8") as f:
        gpt2_reference_merges = [tuple(line.rstrip().split(" ")) for line in f]
        reference_merges = [
            (
                bytes([gpt2_byte_decoder[token] for token in merge_token_1]),
                bytes([gpt2_byte_decoder[token] for token in merge_token_2]),
            )
            for merge_token_1, merge_token_2 in gpt2_reference_merges
        ]
    save_merges_to_json(reference_merges, "./reference_merges.json")