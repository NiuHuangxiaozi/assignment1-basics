

from re import T
import torch
from einops import rearrange
from jaxtyping import Int,Float
from cs336_basics.modules.TransformerLM import NiuTransformerLM
from cs336_basics.tokenizers.tokenizer import Tokenizer


import torch
import torch.nn.functional as F

def sample_next_token(
    logits: torch.Tensor,
    temperature: float = 1.0,
    top_p: float = 0.9
) -> torch.Tensor:
    """
    从 logits 中采样下一个 token，步骤是：
      1. 温度缩放： logits / temperature → softmax 得到 probs
      2. 核采样 (top-p)：选出累积概率 ≥ top_p 的最小 token 集合，再从中重归一化采样

    Args:
        logits: Tensor，形状 [vocab_size] 或 [batch_size, vocab_size]
        temperature: 温度系数 τ (>0)
        top_p: 核采样阈值 p (0 < p ≤ 1)

    Returns:
        next_token: Tensor，选中的下一个 token 的 index （如果 batch>1，返回每 batch 一个）
    """
    # 如果是 batch 维度
    if logits.dim() == 2:
        batch_size, vocab_size = logits.size()
    else:
        logits = logits.unsqueeze(0)
        batch_size, vocab_size = logits.size()

    # 1. 温度缩放
    scaled_logits = logits / temperature

    # 2. 计算 softmax 概率
    probs = F.softmax(scaled_logits, dim=-1)

    # 3. 对每 batch 做 top-p 过滤 & 采样
    next_tokens = []
    for i in range(batch_size):
        prob = probs[i]

        # 排序，降序
        sorted_probs, sorted_indices = torch.sort(prob, descending=True)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

        # 找到最小 k 使得累积 ≥ top_p
        cutoff_idx = torch.searchsorted(cumulative_probs, top_p)

        # 选出候选集
        candidate_indices = sorted_indices[: cutoff_idx + 1]
        candidate_probs = sorted_probs[: cutoff_idx + 1]

        # 归一化这些候选的概率
        candidate_probs = candidate_probs / candidate_probs.sum()

        # 从候选中采样一个
        next_token_local = candidate_indices[torch.multinomial(candidate_probs, num_samples=1)]
        next_tokens.append(next_token_local)

    next_tokens = torch.stack(next_tokens)  # shape [batch_size]
    # 如果原来是单 batch，返回 scalar
    if next_tokens.size(0) == 1:
        return next_tokens[0]
    return next_tokens






def one_sample_model_generation(
                    input_ids: Int[torch.Tensor, "seq_len"],
                    model: NiuTransformerLM,
                    tokenizer: Tokenizer,
                    max_length: int,
                    temperature: float = 1.0,
                    p: float = None,
                    device: torch.device = None) -> str:
    '''
        Generate text using the model, with temperature and nucleus sampling
        Args:
            input_ids: the input ids of the prompt shape(1, seq_len)
            model: the model to generate text
            tokenizer: the tokenizer to use
            max_length: the maximum length of the generated text
            temperature: the temperature for softmax
            p: the probability for nucleus sampling
        Returns:
            the generated text
    '''
    
    model = model.to(device)
    input_ids = input_ids.to(device)
    
    prompt_length = input_ids.shape[-1]
    with torch.no_grad():
        for _ in range(max_length-prompt_length+1):
            
            input_data: Int[torch.Tensor, "seq_len"] = rearrange(input_ids, "seq_len -> 1 seq_len")
            logits: Float[torch.Tensor, "1 seq_len vocab_size"] = model(input_data)
            # 推理过程中我们每次只选择最后面那个值
            last_logits: Float[torch.Tensor, "batch vocab_size"] = logits[:,-1,:]
            last_logits: Float[torch.Tensor, "vocab_size"] = rearrange(last_logits, "1 vocab_size -> vocab_size")
            
            tokens: Int[torch.Tensor, "one_token"] = sample_next_token(last_logits , temperature, p)
            
            input_ids: Int[torch.Tensor, "seq_len"] = torch.cat([input_ids, tokens], dim=0)
            
    return tokenizer.decode(input_ids.tolist())


from cs336_basics.train.train_utils import load_model_config
def test():
    
    # 定义设备
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model_cfg = load_model_config("/home/niu/code/cs336/assignment1-basics/cs336_basics/train/configs/model_configs.yaml")
    # 定义模型
    model = NiuTransformerLM(model_cfg.vocab_size,
                             model_cfg.context_length,
                             model_cfg.d_model,
                             model_cfg.num_layers,
                             model_cfg.num_heads,
                             model_cfg.d_ff,
                             model_cfg.rope_theta,
                             device=device)
    
    model_state_dict = torch.load("/home/niu/code/cs336/assignment1-basics/cs336_basics/train/LM/models_checkpoints/iter_4150/model_iter_4150.pth")["model"]
    model.load_state_dict(model_state_dict)
    
    # 定义tokenizer
    tokenizer = Tokenizer.from_files(
        vocab_filepath="/home/niu/code/cs336/assignment1-basics/cs336_basics/bpe/output/TinyStoriesV2-GPT4-train_optim_vocab_10000.pkl",
        merges_filepath="/home/niu/code/cs336/assignment1-basics/cs336_basics/bpe/output/TinyStoriesV2-GPT4-train_optim_merges_10000.pkl")
    
    # 定义输入文本
    input_text = "Once upon a time, there was a boy who loved a girl, and her name was"
    input_ids: Int[torch.Tensor, "seq_len"] = torch.tensor(tokenizer.encode(input_text)).to(device)
    
    # 定义生成文本
    generated_text = one_sample_model_generation(input_ids=input_ids,
                                                 model=model,
                                                 tokenizer=tokenizer,
                                                 max_length=100,
                                                 temperature=1.0,
                                                 p=1,
                                                 device=device
                                                 )
    # 打印生成文本
    print(generated_text)

if __name__ == "__main__":
    test()




