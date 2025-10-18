
import json
import pprint
import regex
from typing import Iterable, Iterator, List, Dict



# copy from bpe training, 有小的改动
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
    parts = [part for part in parts if part != ""]
    # parts 中包含拆分出的文本片段 & 拆分符号本身（因为用了捕获组）
    docs: List[str] = []
    cur: List[str] = []

  
    index = 0
    while index < len(parts):
        seg= parts[index]
        if seg in special_tokens:
            tmp = index
            all_special_tokens = []
            while tmp >= 0 and  tmp < len(parts) and parts[tmp] in special_tokens:
                all_special_tokens.append(parts[tmp])
                tmp += 1
            i = len(all_special_tokens)
            while i >= 1:
                if ''.join(all_special_tokens[:i]) in special_tokens:
                    docs.append("".join(cur))
                    docs.append(''.join(all_special_tokens[:i]))
                    cur = []
                    break
                i -= 1
            index = index + i
        else:
            cur.append(seg)
            index +=1
    # 最后剩下的也算一个 doc
    if cur:
        docs.append("".join(cur))
    
    # 过滤空字符串
    docs = [d for d in docs if d != ""]
    # print(f"docs is {docs}")
    return docs


class Tokenizer:
    def __init__(self, 
                 vocab: dict[int, bytes],
                 merges: list[tuple[bytes, bytes]],
                 special_tokens: list[str] | None = None):
        
        # 保存的关键数据结构
        self.id2token = {}
        self.token2id = {}
        
        self.vocab = vocab
        
        for i, token in vocab.items():
            self.id2token[i] = token
            self.token2id[token] = i
        assert self.id2token == self.vocab, "id2token and vocab are not the same"
        
        self.bpe_ranks = merges
        self.special_tokens = special_tokens
        self.PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        
        
        # pprint.pprint(self.token2id)
        # 处理special tokens
        if special_tokens:
            for special_token in special_tokens:
                if special_token.encode("utf-8") not in self.token2id.keys():
                    self.vocab[len(self.vocab)] = special_token.encode("utf-8")
                    self.token2id[special_token.encode("utf-8")] = len(self.token2id)
                    self.id2token[len(self.id2token)] = special_token.encode("utf-8")
        
        if special_tokens:
            self.special_tokens_bytes = [special_token.encode("utf-8") for special_token in special_tokens]
        else:
            self.special_tokens_bytes = None
        
    @classmethod
    def from_files(cls, vocab_filepath: str,
                   merges_filepath: str,
                   special_tokens: list[str] | None = None):
        '''
            Class method that constructs and return a Tokenizer from a serialized vocabulary and list of merges
            (in the same format that your BPE training code output) and (optionally) a list of special tokens
            
        Args:
            vocab_filepath: path to the vocabulary file
            merges_filepath: path to the merges file
            special_tokens: list of special tokens
        Returns:
            A Tokenizer object
        '''
        
         # 读取 vocab 文件
        with open(vocab_filepath, "r", encoding="utf-8") as f:
            vocab_dict = json.load(f)
        vocab : Dict[int, bytes] = {int(i): t.encode("utf-8") for i, t in vocab_dict.items()}


        # 读取 merges 文件
        with open(merges_filepath, "r", encoding="utf-8") as f:
            merges_list = json.load(f)
        # 把 merges 转为 rank 表
        merges : List[tuple[bytes, bytes]] = [(pair[0].encode("utf-8"), pair[1].encode("utf-8")) for _, pair in enumerate(merges_list)]
    

        return cls(vocab=vocab, merges=merges, special_tokens=special_tokens)
    
    
    def _get_token_byte_ids(self, token_subwords :List[bytes]) -> List[int]:
        
        token_byte_ids = []
        try:
            # 如果只有一个byte token
            if len(token_subwords) == 1:
                if self.token2id.get(token_subwords[0], None) is not None:
                    token_byte_ids.append(self.token2id[token_subwords[0]])
                    return token_byte_ids
                else:
                    raise ValueError(f"Token {token_subwords[0]} not found in vocabulary")
            
            has_merge_operations = True
            while has_merge_operations == True and len(token_subwords) > 1:
                
                has_merge_operations = False
                for pri_pairs in self.bpe_ranks:
                    
                    is_merged = False
                    for index, (b1, b2) in enumerate(zip(token_subwords, token_subwords[1:])):
                        if (b1, b2) == pri_pairs:
                            # 进行替换
                            token_subwords[index] = pri_pairs[0] + pri_pairs[1]
                            del token_subwords[index + 1]
                            is_merged = True
                            break
                    if is_merged == True:
                        has_merge_operations = True
                        break 
                    
            for byte_token in token_subwords:
                token_byte_ids.append(self.token2id[byte_token])
            return token_byte_ids
        except ValueError as e:
            raise ValueError(f"Error getting token byte ids: {e}")
    def encode(self, text: str) -> list[int]:
        '''
                Encode an input text into a sequence of token IDs
        '''
        tokenized_text :List[int] = []
        
        # splited_text 里面包含特殊的字符
        if self.special_tokens is not None:
            splited_text = split_on_special_tokens(text=text, special_tokens=self.special_tokens)
        else:
            splited_text = [text]
        
        for text in splited_text:
            # print(f"text: {text}")
            # special_tokens已经在self.token2id中，所以直接append
            if self.special_tokens and text in self.special_tokens:
                tokenized_text.append(self.token2id[text.encode("utf-8")])
            else:
                for token in regex.finditer(self.PAT, text):
                    token_str = token.group(0)
                    token_bytes = token_str.encode("utf-8")
                    token_subwords = [token_bytes[i:i+1] for i in range(len(token_bytes))]
                    token_byte_ids = self._get_token_byte_ids(token_subwords)
                    tokenized_text.extend(token_byte_ids)
        return tokenized_text
    
    
    def _iter_encode(self, iterable: Iterable[str]) -> Iterator[int]:
        for index, text in enumerate(iterable):
            encode_list = self.encode(text)
            for id in encode_list:
                yield id
        
    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        '''
        Encode an iterable of input texts into a sequence of token IDsGiven an iterable of
        strings (e.g., a Python file handle), return a generator that lazily yields token IDs. This is
        required for memory-eﬀicient tokenization of large files that we cannot directly load into
        memory.
        '''
        iter_func = self._iter_encode(iterable=iterable)
        return iter_func
            
    def decode(self, ids: list[int]) -> str:
        '''
            Decode a sequence of token IDs into a string
        '''
        result_component = []
        error_bytes = b'\x80'
        
        bytes_content = b''
         
        for id in ids:
            if id in self.id2token:
                token_bytes = self.id2token[id]
                if self.special_tokens_bytes is not None and token_bytes in self.special_tokens_bytes:
                    
                    if bytes_content != b'':
                        str_content = bytes_content.decode("utf-8", errors="replace")
                        result_component.append(str_content)
                        
                    result_component.append(token_bytes.decode("utf-8", errors="replace"))
                    bytes_content = b''
                else:
                    bytes_content += token_bytes
            else:
                # 畸形字节 \x80
                bytes_content += error_bytes
        

        if bytes_content != b'':
            tail_str_content = bytes_content.decode("utf-8", errors="replace")
            result_component.append(tail_str_content)
        
        return ''.join(result_component)



# ================================
import os
from functools import lru_cache



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
        
        
def get_tokenizer_from_vocab_merges_path(
    vocab_path: str | os.PathLike,
    merges_path: str | os.PathLike,
    special_tokens: list[str] | None = None,
):
    gpt2_byte_decoder = {v: k for k, v in gpt2_bytes_to_unicode().items()}
    with open(vocab_path) as vocab_f:
        gpt2_vocab = json.load(vocab_f)
    gpt2_bpe_merges = []
    with open(merges_path) as f:
        for line in f:
            cleaned_line = line.rstrip()
            if cleaned_line and len(cleaned_line.split(" ")) == 2:
                gpt2_bpe_merges.append(tuple(cleaned_line.split(" ")))
    # The GPT-2 tokenizer uses a remapped unicode encoding for bytes. Let's
    # just return the original bytes, so we don't force students to use
    # any particular encoding scheme.
    vocab = {
        gpt2_vocab_index: bytes([gpt2_byte_decoder[token] for token in gpt2_vocab_item])
        for gpt2_vocab_item, gpt2_vocab_index in gpt2_vocab.items()
    }
    # If any of the special tokens don't exist in the vocab, append them to the vocab.
    if special_tokens:
        for special_token in special_tokens:
            byte_encoded_special_token = special_token.encode("utf-8")
            if byte_encoded_special_token not in set(vocab.values()):
                vocab[len(vocab)] = byte_encoded_special_token

    merges = [
        (
            bytes([gpt2_byte_decoder[token] for token in merge_token_1]),
            bytes([gpt2_byte_decoder[token] for token in merge_token_2]),
        )
        for merge_token_1, merge_token_2 in gpt2_bpe_merges
    ]
    return get_tokenizer(vocab, merges, special_tokens)

@lru_cache
def gpt2_bytes_to_unicode() -> dict[int, str]:
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    # now get the representations of the other 68 integers that do need shifting
    # each will get mapped chr(256 + n), where n will grow from 0...67 in the loop
    # Get printable representations of the remaining integers 68 integers.
    n = 0
    for b in range(2**8):
        if b not in bs:
            # If this integer isn't in our list of visually-representable
            # charcters, then map it to the next nice character (offset by 256)
            bs.append(b)
            cs.append(2**8 + n)
            n += 1
    characters = [chr(n) for n in cs]
    d = dict(zip(bs, characters))
    return d


def get_tokenizer(
    vocab: dict[int, bytes],
    merges: list[tuple[bytes, bytes]],
    special_tokens: list[str] | None = None,
):
    tokenizer = Tokenizer(vocab=vocab, merges=merges, special_tokens=special_tokens)
    return tokenizer


VOCAB_PATH = "/home/niu/code/cs336/assignment1-basics/tests/fixtures/gpt2_vocab.json"
MERGES_PATH = "/home/niu/code/cs336/assignment1-basics/tests/fixtures/gpt2_merges.txt"

if __name__ == "__main__":
    merge_file = "/home/niu/code/cs336/assignment1-basics/cs336_basics/bpe/TinyStoriesV2-GPT4-train_optim_merges_10000.json"
    vocab_file = "/home/niu/code/cs336/assignment1-basics/cs336_basics/bpe/TinyStoriesV2-GPT4-train_optim_vocab_10000.json"
    
    
    tokenizer = get_tokenizer_from_vocab_merges_path(
        vocab_path=VOCAB_PATH,
        merges_path=MERGES_PATH,
        special_tokens=["<|endoftext|>"],
    )
    # tokenizer = Tokenizer.from_files(vocab_filepath=vocab_file, merges_filepath=merge_file)
    test_string = "Héllò hôw <|endoftext|><|endoftext|> are ü? 🙃<|endoftext|>"
    target_string = tokenizer.decode(tokenizer.encode(test_string))
    print(f"target_string: {target_string}")
    assert target_string == test_string