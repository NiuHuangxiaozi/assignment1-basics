


import pickle
from typing import List, Tuple, Dict

def save_merges_to_pickle(
    d: List[Tuple[bytes, bytes]],
    filepath: str,
    protocol: int = pickle.HIGHEST_PROTOCOL
) -> None:
    """
    保存 merges（bytes 对）到 pickle 文件。
    d: 列表，每个元素是一个 tuple(bytes, bytes)
    filepath: 保存路径
    protocol: pickle 协议版本（默认最高）
    """
    with open(filepath, "wb") as f:
        pickle.dump(d, f, protocol=protocol)

def load_merges_from_pickle(
    filepath: str
) -> List[Tuple[bytes, bytes]]:
    """
    从 pickle 文件加载 merges，返回 List[tuple(bytes, bytes)]。
    filepath: pickle 文件路径
    """
    with open(filepath, "rb") as f:
        obj = pickle.load(f)
    # optionally check type
    if not isinstance(obj, list):
        raise ValueError(f"Expected a list from {filepath}, got {type(obj)}")
    # further check elements are tuple of bytes
    for i, item in enumerate(obj):
        if (not isinstance(item, tuple) or len(item) != 2
                or not isinstance(item[0], (bytes, bytearray))
                or not isinstance(item[1], (bytes, bytearray))):
            raise ValueError(f"Item at index {i} is not a tuple(bytes, bytes): {item!r}")
    # convert bytearray to bytes if needed
    result = [(bytes(a), bytes(b)) for (a, b) in obj]
    return result

def save_bytes_dict_to_pickle(
    d: Dict[int, bytes],
    filepath: str,
    protocol: int = pickle.HIGHEST_PROTOCOL
) -> None:
    """
    保存一个 int->bytes 的字典到 pickle 文件。
    """
    with open(filepath, "wb") as f:
        pickle.dump(d, f, protocol=protocol)

def load_bytes_dict_from_pickle(
    filepath: str
) -> Dict[int, bytes]:
    """
    从 pickle 文件加载 int->bytes 字典，返回 Dict[int, bytes]。
    """
    with open(filepath, "rb") as f:
        obj = pickle.load(f)
    if not isinstance(obj, dict):
        raise ValueError(f"Expected a dict from {filepath}, got {type(obj)}")
    # check keys and values
    result: Dict[int, bytes] = {}
    for k, v in obj.items():
        if not isinstance(k, int):
            raise ValueError(f"Key {k!r} is not int")
        if not isinstance(v, (bytes, bytearray)):
            raise ValueError(f"Value for key {k!r} is not bytes: {v!r}")
        result[k] = bytes(v)
    return result
