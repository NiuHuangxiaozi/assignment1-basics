import regex
import logging  
from typing import Dict, Tuple, List,Set
import os
import sys
from collections import defaultdict
from pprint import pprint
import copy 
root = os.path.dirname(os.path.dirname(__file__))
if root not in sys.path:
    sys.path.insert(0, root)
    
from tests.common import gpt2_bytes_to_unicode
from cs336_basics.invertindex import InvertIndex

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
    
    # logger.info(f"Initial subword2count size: {len(subword2count)}")
    # for k, v in list(subword2count.items()):
    #     logger.info(f"  {repr(k)}: {v}")
    # logger.info(f"=====================================================================")
    # b"</w>" 表示的是字节串, 这个本来可以家的，但是我们使用了pre-tokenization
    
    
    
    # 这里需要构建一个倒排索引，（b1, b2） -> subword2count.key
    inverted_index = InvertIndex(subword2count)
    
    # 构建整体键值对的计数字典
    pair_count : Dict[Tuple[bytes, bytes], int] = {}
    for sub_word_bytes, sub_word_count in subword2count.items():
            # logger.info(f" sub_word_bytes: {sub_word_bytes} count: {sub_word_count}")
            for b1, b2 in zip(sub_word_bytes, sub_word_bytes[1:]):
                pair_count[(b1,b2)] = pair_count.get((b1,b2), 0) + sub_word_count
    

    # with open("pair_count_original.txt", mode="w") as f:
    #     pprint(pair_count, stream=f)
    # with open("inverted_index_original.txt", mode="w") as f:
    #     print(inverted_index, file=f)
    qwe = 0
    epoch = 0
    while len(vocab) < vocab_size:  
        qwe +=1
        # 根据pair_count 找到出现频率最高的pair
        most_pairs = get_most_appear_pair(pair_count)
        
        # with open("subword2count_original.txt", "w", encoding="utf-8") as f:
        #     print("ORIGINL subword2count : \n", file=f)
        #     pprint(subword2count, stream=f)
        #     print("\n\n", file=f)
        
        # with open("original_paircount.txt", "w", encoding="utf-8") as f:
        #     print("original pair count : : \n", file=f)
        #     pprint(pair_count, stream=f)
        #     print("\n\n", file=f)
        
        # print(f"**** Most pair  is {most_pairs[0]} ****")
          
        # 更新 vocab，加入新合并的子词
        new_bytes =  most_pairs[0][0] + most_pairs[0][1]
        vocab[len(vocab)] = new_bytes
        
    
        # print(f"epoch is {epoch}")
        # with open(f"pair_count{epoch}.txt", mode="w") as f:
        #         pprint(pair_count, stream=f)
        # if new_bytes == b' the':
        #         print(f"pair count[ the] is {pair_count[(b' t', b'he')]}")
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
            # with open("subword2count_changed.txt", "w", encoding="utf-8") as f:
            #     print("cahnged subword2count : \n", file=f)
            #     pprint(subword2count, stream=f)
            #     print("\n\n", file=f)

            
            # with open("inverted_index.txt", "w", encoding="utf-8") as f:
            #     print(" inverted_index : \n", file=f)
            #     print(inverted_index, file=f)
            #     print("\n\n", file=f)
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
            # with open("inverted_index_add.txt", "w", encoding="utf-8") as f:
            #     print("add inverted_index : \n", file=f)
            #     print(inverted_index, file=f)
            #     print("\n\n", file=f)      
            
            
            # with open("pair_count.txt", "w", encoding="utf-8") as f:
            #     print("original_pair count : \n", file=f)
            #     pprint(pair_count, stream=f)
            #     print("\n\n", file=f)      
            # 更新pair_count
            # del pair_count[most_pairs[0]] 
            # 首先遍老的序列在pair_count中删除
            k= 0
            while k < len(old_sub_word_bytes):
                if k < len(old_sub_word_bytes) - 1 and (old_sub_word_bytes[k], old_sub_word_bytes[k+1]) == most_pairs[0]: 
                    if k - 1 >= 0:
                        pre_pairs = (old_sub_word_bytes[k - 1], old_sub_word_bytes[k])
                        # print(f"pre_pairs is {pre_pairs}")
                        pair_count[pre_pairs] = pair_count.get(pre_pairs, 0) - sub_word_bytes_count
                        if pair_count[pre_pairs] == 0:
                            del pair_count[pre_pairs]
                            
                    if k + 2 <= len(old_sub_word_bytes) - 1:
                          
                        post_pairs = (old_sub_word_bytes[k + 1], old_sub_word_bytes[k + 2])
                        # print(f"post_pairs is {post_pairs}")
                        pair_count[post_pairs] = pair_count.get(post_pairs, 0) - sub_word_bytes_count
                        if pair_count[post_pairs] == 0:
                            del pair_count[post_pairs]
                            
                    if most_pairs[0] in pair_count.keys():
                        del pair_count[most_pairs[0]]    
                    k += 2
                else:
                    k += 1
            
            # with open("pair_count_deleted.txt", "w", encoding="utf-8") as f:
            #     print("delete pair count : \n", file=f)
            #     pprint(pair_count, stream=f)
            #     print("\n\n", file=f)  
                
            # 其次我们遍历新的序列，然后再次计算新加入的pair的count
            m = 0
            # print(f"most_pairs is {most_pairs[0]}")
            # print(f"new_sub_word_bytes is {new_sub_word_bytes}")
            while m < len(new_sub_word_bytes):
                if new_sub_word_bytes[m] == most_pairs[0][0] + most_pairs[0][1]:
                    
                    # hehe的bug
                    if m - 1 >= 0:
                        pre_new_pairs = (new_sub_word_bytes[m-1], new_sub_word_bytes[m])
                        # print(f"pre_new_pairs is {pre_new_pairs}")
                        pair_count[pre_new_pairs] = pair_count.get(pre_new_pairs, 0) + sub_word_bytes_count
                    if m + 1 <= len(new_sub_word_bytes)-1:
                        post_new_pairs = (new_sub_word_bytes[m], new_sub_word_bytes[m+1])
                        # print(f"post_new_pairs is {post_new_pairs}")
                        pair_count[post_new_pairs] = pair_count.get(post_new_pairs, 0) + sub_word_bytes_count
                        # if new_sub_word_bytes[m] == b' t' and new_sub_word_bytes[m+1] == b'h':
                        #     with open("zxc.txt", mode='a') as f:
                        #         print(f"===============", file=f)
                        #         # pprint(pair_count, stream= f)
                        #         print(f"old sub word bytes is {old_sub_word_bytes}", file=f)
                        #         print(f"new sub word bytes is {new_sub_word_bytes}", file=f)
                        #         print(f"sub_word_bytes_count is {sub_word_bytes_count}", file=f)
                                
                        #         print(f"************", file=f)
                    m += 1
                else:
                    m += 1
            # with open("pair_count_add.txt", "w", encoding="utf-8") as f:
            #     print("add pair count : \n", file=f)
            #     pprint(pair_count, stream=f)
            #     print("\n\n", file=f)  
        epoch =epoch + 1
            # if epoch == 4 and qwe ==2:
            #     assert False
    return vocab, merges    






# 本地调试区域
def bytes_to_safe_str(b: bytes) -> str:
    try:
        return b.decode('utf-8')
    except UnicodeDecodeError:
        # 不能utf-8 decode 的，用 repr 或 base64 表示
        return repr(b)
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
    corpus_path  = "/home/niu/code/cs336/assignment1-basics/tests/fixtures/corpus.en"
    vocab_size =  500
    special_tokens = ["<|endoftext|>"]
    vocab, merges = train_bpe(corpus_path, vocab_size, special_tokens)
    # logger.info(f"vocab: \n{vocab}")
    save_merges_to_json(merges, "./optim_merges.json")
    save_bytes_dict_to_json(vocab, "./optim_vocab.json")