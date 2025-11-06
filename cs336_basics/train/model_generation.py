

import torch
from jaxtyping import Int,Float
from cs336_basics.modules.TransformerLM import NiuTransformerLM
from cs336_basics.tokenizers.tokenizer import Tokenizer





def nucleus_sampling(probs: Float[torch.Tensor, "batch vocab_size"], p: float) -> Int[torch.Tensor, "batch vocab_size"]:
    '''
        Nucleus sampling
        Args:
            probs: the probabilities of the tokens shape(batch, vocab_size)
            p: the probability for nucleus sampling
        Returns:
            the tokens
    '''
    # 排序后的概率值是sorted_probs，排序后的对应于原来probs索引是sorted_indices
    sorted_probs, sorted_indices = torch.sort(probs, dim=-1, descending=True)
    
    # 沿着概率的维度进行累积求和
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
    # 如果累积概率大于p，则将概率设置为0
    mask = cumulative_probs > p
    
    # -1代表的是不保留
    sorted_indices = sorted_indices.masked_fill(mask, -1)
    new_logits = sorted_probs.masked_fill(mask, -float('inf'))
    new_probs = torch.softmax(new_logits, dim=-1)
    return new_probs, sorted_indices




def one_sample_model_generation(
                    input_ids: Int[torch.Tensor, "batch seq_len"],
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
            logits: Float[torch.Tensor, "batch seq_len vocab_size"] = model(input_ids)
            # d带有正常温度的softmax
            
            # 推理过程中我们每次只选择最后面那个值
            last_logits = logits[:, -1, :]
            
            # 全部除以温度
            last_logits = last_logits / temperature
            # 求出每一个token的概率
            probs = torch.softmax(last_logits, dim=-1)
            
            # 如果用户设置了 p，则进行 nucleus sampling
            if p is not None:
                new_probs, new_indices = nucleus_sampling(probs, p)
                probs = new_probs
                indices = new_indices
            else:
                indices = torch.argmax(probs, dim=-1)
            
            slected_index = torch.multinomial(probs, num_samples=1)

            tokens = indices.gather(dim=-1, index=slected_index)
            
            # 如果是结束的token，则直接跳出循环
            if tokens == tokenizer.get_end_token_id():
                break
            input_ids = torch.cat([input_ids, tokens], dim=-1)
            

    return tokenizer.decode(input_ids.tolist()[0])


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
    
    model_state_dict = torch.load("/home/niu/code/cs336/assignment1-basics/cs336_basics/train/debug/models_checkpoints/epoch_10/model_epoch_10.pth")["model"]
    model.load_state_dict(model_state_dict)
    
    # 定义tokenizer
    tokenizer = Tokenizer.from_files(
        vocab_filepath="/home/niu/code/cs336/assignment1-basics/cs336_basics/bpe/output/TinyStoriesV2-GPT4-train_optim_vocab_10000.pkl",
        merges_filepath="/home/niu/code/cs336/assignment1-basics/cs336_basics/bpe/output/TinyStoriesV2-GPT4-train_optim_merges_10000.pkl")
    
    # 定义输入文本
    input_text = "Once upon a time, there was a boy who loved a girl, and her name was"
    input_ids = torch.tensor(tokenizer.encode(input_text)).unsqueeze(0).to(device)
    
    # 定义生成文本
    generated_text = one_sample_model_generation(input_ids=input_ids,
                                                 model=model,
                                                 tokenizer=tokenizer,
                                                 max_length=100,
                                                 temperature=1.0,
                                                 p=0.8,
                                                 device=device
                                                 )
    # 打印生成文本
    print(generated_text)

if __name__ == "__main__":
    test()




