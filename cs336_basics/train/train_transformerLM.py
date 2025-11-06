

import argparse
from re import L
import torch
import time
import os
import numpy as np
from tqdm import tqdm
from typing import Tuple, Iterator, Optional
from cs336_basics.train.train_utils import load_model_config, load_training_config
from cs336_basics.tools.tools import data_loading
from cs336_basics.modules.TransformerLM import NiuTransformerLM
from cs336_basics.optimizer.optimizer import NIUAdam
from cs336_basics.loss.loss import NIUCrossEntropyLoss
from cs336_basics.tools.tools import cosine_scheduling
from cs336_basics.tools.tools import save_checkpoint
from cs336_basics.tools.tools import gradient_clipping


class TrainDataGenerator(Iterator[Tuple[torch.Tensor, torch.Tensor]]):
    def __init__(self, train_ids, batch_size, context_length, device):
        self.train_ids = train_ids
        self.batch_size = batch_size
        self.context_length = context_length
        self.device = device
        self._batches_yielded = 0      # 内部计数器
        self.max_iter_num = min(len(train_ids) - context_length, 100) # 最大迭代次数
        print(f"max_iter_num is {min(self.max_iter_num,10)}")
    def __iter__(self):
      return self
    
    def __next__(self):
      self._batches_yielded += 1
      if self._batches_yielded > self.max_iter_num:
        self._batches_yielded = 0
        raise StopIteration("All batches have been yielded")
      else:
        x, y = data_loading(self.train_ids, self.batch_size, self.context_length, self.device)
        return x, y
     
class Scheduler:
    def __init__(self, optimizer, learning_rate, learning_rate_min, warmup_steps, cosine_cycle_steps):
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.learning_rate_min = learning_rate_min
        self.warmup_steps = warmup_steps
        self.cosine_cycle_steps = cosine_cycle_steps
    def step(self, t:Optional[int]=None):
        if t is None:
            raise ValueError("t is required")
        lr = cosine_scheduling(t, self.learning_rate, self.learning_rate_min, self.warmup_steps, self.cosine_cycle_steps)
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

# 训练模型
def train(model, train_data_generator: TrainDataGenerator, optimizer, criterion, scheduler, training_cfg):
    tqdm.write(f"Training started with {training_cfg.epochs} epochs")
    
    epoch_avg_loss = 0
    iter_avg_loss = 0
    
    model.to(training_cfg.device)
    model.train()
    for epoch in tqdm(range(training_cfg.epochs)):
        for i, data_batch in enumerate(train_data_generator):
            x, y = data_batch
            optimizer.zero_grad()
            x = x.to(training_cfg.device)
            y = y.to(training_cfg.device)
            logits = model(x)
            
            loss = criterion(logits, y)
            loss.backward()
            gradient_clipping(model.parameters(), training_cfg.max_norm)
            optimizer.step()
            iter_avg_loss += loss.item()
            epoch_avg_loss += loss.item()
            if i % training_cfg.iter_print_freq == 0:
                tqdm.write(f"Epoch {epoch+1}, Iter {i+1}, Average Loss {iter_avg_loss/training_cfg.iter_print_freq:.4f}")
                iter_avg_loss = 0
        if epoch % training_cfg.epoch_print_freq == 0:
            tqdm.write(f"Epoch {epoch+1}, Average Loss {epoch_avg_loss/training_cfg.epoch_print_freq:.4f}")
            epoch_avg_loss = 0
            os.makedirs(os.path.join(training_cfg.exp_name, training_cfg.save_path, f"epoch_{epoch+1}"), exist_ok=True)
            save_checkpoint(model, optimizer, epoch+1, os.path.join(training_cfg.exp_name, training_cfg.save_path, f"epoch_{epoch+1}", f"model_epoch_{epoch+1}.pth"))
        scheduler.step(t=epoch)
    tqdm.write(f"Training finished")
    return model

def train_transformerLM(args):
    model_cfg = load_model_config(args.model_cfg_path)
    training_cfg = load_training_config(args.exp_cfg_path)
    # 创建保存路径
    os.makedirs(os.path.join(training_cfg.exp_name, training_cfg.save_path), exist_ok=True)
    
    # 读入数据集
    train_ids: np.ndarray = np.load(training_cfg.train_ids_path, mmap_mode='r')
    train_data_generator: TrainDataGenerator = TrainDataGenerator(train_ids, training_cfg.batch_size, model_cfg.context_length, training_cfg.device)
    
    # 定义模型
    model = NiuTransformerLM(model_cfg.vocab_size,
                             model_cfg.context_length,
                             model_cfg.d_model,
                             model_cfg.num_layers,
                             model_cfg.num_heads,
                             model_cfg.d_ff,
                             model_cfg.rope_theta,
                             device=training_cfg.device)
    
    # 定义优化器
    optimizer = NIUAdam(model.parameters(),
                        lr= training_cfg.learning_rate,
                        betas=(training_cfg.optimizer.beta1, training_cfg.optimizer.beta2),
                        eps=training_cfg.optimizer.eps,
                        weight_decay=training_cfg.optimizer.weight_decay
                        )
    
    # 定义损失函数
    criterion = NIUCrossEntropyLoss()
    
    # 定义学习率调度器
    scheduler = Scheduler(optimizer,
                          learning_rate=training_cfg.learning_rate,
                          learning_rate_min=training_cfg.learning_rate_min,
                          warmup_steps=training_cfg.warmup_steps,
                          cosine_cycle_steps=training_cfg.cosine_cycle_steps)
    

    # 定义训练器
    trained_model = train(model,
                          train_data_generator,
                          optimizer,
                          criterion,
                          scheduler,
                          training_cfg)
    
    return trained_model



if __name__ == "__main__":
    parser = argparse.ArgumentParser("train transformerLM on TinyStoriesV2-GPT4 dataset")
    parser.add_argument("--model_cfg_path", type=str, default="configs/model_configs.yaml")
    parser.add_argument("--exp_cfg_path", type=str, default="configs/exp_configs.yaml")
    args = parser.parse_args()
    train_transformerLM(args)