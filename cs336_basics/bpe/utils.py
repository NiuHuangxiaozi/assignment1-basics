

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
            



from typing import Tuple, Dict  
from sortedcontainers import SortedDict
class BucketMaxSD:
    def __init__(self):
        self.counts: Dict[Tuple[bytes, bytes], int] = {}
        self.buckets = SortedDict() # Num->Set(Pair)
    
    
    # 获取最大的pair
    def get_most_appear_pair(self) -> Tuple[Tuple[bytes, bytes],int]:
        if not self.buckets:
            raise ValueError("No pairs present")
        # 获取最大 count 的 bucket
        max_number, pair_set = self.buckets.peekitem(-1)
        # 如果多个 pair，选择字典序最大的 tuple
        best_pair = max(pair_set)
        return (best_pair, max_number)
    
    def add(self, pair: Tuple[bytes, bytes], count: int):
        
        
        # 1、修改count
        old_count = self.counts.get(pair, 0)
        # 创建新的或者更新
        self.counts[pair] = old_count + count
        new_count = self.counts[pair]
        
        
        # 2、修改buckets
        # 将原来old_count对应的pair删去
        if old_count != 0:
            assert old_count in self.buckets.keys()
            self.buckets[old_count].remove(pair)
            if len(self.buckets[old_count]) == 0:
                del self.buckets[old_count]
        
        # 将new_count对应的pair添加进去
        if new_count in self.buckets.keys():
            self.buckets[new_count].add(pair)
        else:
            self.buckets[new_count] = set()
            self.buckets[new_count].add(pair)
        
    
    
    # 将这个pair彻底的从数据结构中删除
    def delete_all(self, pair: Tuple[bytes, bytes]):
        if pair not in self.counts.keys():
            return
        old_count = self.counts[pair]
        del self.counts[pair]
        self.buckets[old_count].remove(pair)
        if len(self.buckets[old_count]) == 0:
            del self.buckets[old_count]
    
    
    
    
    def delete(self, pair: Tuple[bytes, bytes], count: int):
        
        # 0、不用删除
        if pair not in self.counts.keys():
            return
        
        # 1、修改count
        original_count = self.counts[pair]
        self.counts[pair] = original_count - count
        
        
        # 如果原来的count在buckets中存在，那么bucket需要更新
        if original_count in self.buckets.keys():
            self.buckets[original_count].remove(pair)
            if len(self.buckets[original_count]) == 0:
                del self.buckets[original_count]
                
        # 如果这个pair的count小于0，那么就删除
        if self.counts[pair] <= 0:
            del self.counts[pair]    
        else:                    
            if self.counts[pair] in self.buckets.keys():
                self.buckets[self.counts[pair]].add(pair)
            else:
                self.buckets[self.counts[pair]] = set()
                self.buckets[self.counts[pair]].add(pair)
           
                