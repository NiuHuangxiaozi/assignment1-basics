

from typing import List, Tuple, Dict
from collections import defaultdict
import json
class InvertIndex:
    
    def __init__(self, subword2count : dict[tuple[bytes], int]):
        self.subword2count = subword2count
        self.index: Dict[Tuple[bytes, bytes], List[int]] = defaultdict(list)
        self.sub_words_bytes_list : List[List[bytes]]= []
        
        self._init()
    
    
    def __str__(self):
         # index 部分：一行一个键值对
        index_lines = []
        for k, v in self.index.items():
            key_repr = "(" + ", ".join(repr(b) for b in k) + ")"
            val_repr = "[" + ", ".join(str(i) for i in v) + "]"
            index_lines.append(f"    {key_repr} : {val_repr}")
        index_block = "{\n" + "\n".join(index_lines) + "\n  }"

        # sub_words_bytes_list 部分：每个子列表一行，前面带索引号
        swbl_lines = []
        swbl_lines.append("  sub_words_bytes_list = [")
        for idx, sublist in enumerate(self.sub_words_bytes_list):
            # sublist 是 List[bytes]，用 repr 表示整个列表
            sublist_repr = "[" + ", ".join(repr(b) for b in sublist) + "]"
            swbl_lines.append(f"    {idx}: {sublist_repr}")
        swbl_lines.append("  ]")
        swbl_block = "\n".join(swbl_lines)

        return (
            f"InvertIndex(\n"
            f"  index size: {len(self.index)}\n"
            f"  sub_words_bytes_list size: {len(self.sub_words_bytes_list)}\n"
            f"  index = {index_block}\n"
            f"{swbl_block}\n"
            f")"
        )
        
    def __len__(self):
        return len(self.sub_words_bytes_list)
    
    def _init(self):
        sub_words_bytes_list_index = 0
        for sub_word_bytes in self.subword2count.keys():
            self.sub_words_bytes_list.append(list(sub_word_bytes))
            for b1, b2 in zip(sub_word_bytes, sub_word_bytes[1:]):
                self.index[(b1,b2)].append(sub_words_bytes_list_index)
            sub_words_bytes_list_index += 1

    
    # 增
    def subscribe_invertindex(self, 
            index:int,
            new_sub_word_bytes: Tuple[bytes],
        ):
        assert index >=0
        assert index <len(self.sub_words_bytes_list)
        self.sub_words_bytes_list[index] = list(new_sub_word_bytes)
    
    def get_sub_words_bytes_index(self, pairs: Tuple[bytes, bytes]):
        return self.index[pairs]

    def get_sub_words_bytes(self, index):
        return self.sub_words_bytes_list[index]
    
        
    # 判断一个新的子串里面还存不存在这个pair
    def _pairs_in_sub_words_bytes(self, pairs: Tuple[bytes, bytes], sub_word_bytes: List[bytes]) -> bool:
            for b1, b2 in zip(sub_word_bytes, sub_word_bytes[1:]):
                if (b1, b2) == pairs:
                    return True
            return False
    
    def delete_index(self, pairs: Tuple[bytes, bytes], index: int, new_sub_word_bytes: List[bytes]):
        '''
            删除对应的指针，如果新的new_sub_word_bytes 不再包含pairs，那么就删除，某则保留
        '''
        if self._pairs_in_sub_words_bytes(pairs, new_sub_word_bytes):
            return
        else:
            if index in self.index[pairs]:
                while index in self.index[pairs]:
                    self.index[pairs].remove(index)
            if len(self.index[pairs]) == 0:
                del self.index[pairs]
    
    def add_index(self, pairs :Tuple[bytes, bytes], index: int):
        '''
            一行就把pairs在不在的情况全部包括了
        '''
        if index not in self.index[pairs]:
            self.index[pairs].append(index)
    