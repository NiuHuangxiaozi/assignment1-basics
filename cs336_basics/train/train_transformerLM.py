

import argparse
from re import L
import wandb
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
from cs336_basics.tools.tools import load_checkpoint
import logging

logging.basicConfig(filename='transformerLM_tinystory_GPT4_train_dataset.log', level=logging.INFO)
logger = logging.getLogger(__name__)

class TrainDataGenerator(Iterator[Tuple[torch.Tensor, torch.Tensor]]):
    def __init__(self, train_ids, batch_size, context_length, device):
        self.train_ids = train_ids
        self.batch_size = batch_size
        self.context_length = context_length
        self.device = device
        self._batches_yielded = 0      # 内部计数器
        self.max_iter_num = min(len(train_ids) - context_length, 7000) # 最大迭代次数
        logger.info(f"max_iter_num is {min(self.max_iter_num, 7000)}")
    def __iter__(self):
      return self
    def __len__(self):
      return self.max_iter_num
    def reset(self, iter_number):
        self._batches_yielded = iter_number % len(self)
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
def train(model, train_data_generator: TrainDataGenerator, optimizer, criterion, scheduler, training_cfg, iter_number):
    logger.info(f"Training started with {training_cfg.epochs} epochs")

    start_time = time.time()
    epoch_loss = 0
    iter_loss_list = []
    
    train_data_generator.reset(iter_number)
    model.to(training_cfg.device)
    model.train()
    
    
    total_iters_pbar = tqdm(total = training_cfg.epochs * len(train_data_generator), initial=iter_number, desc="Training Small Language Model...")
    
    cur_epoch = iter_number // len(train_data_generator)
    epoch_pbar = tqdm(total=training_cfg.epochs, initial=cur_epoch, desc="Epochs...")
    
    for epoch in epoch_pbar:
        for _, data_batch in enumerate(train_data_generator):
            
            x, y = data_batch
            optimizer.zero_grad()
            x = x.to(training_cfg.device)
            y = y.to(training_cfg.device)
            logits = model(x)

            loss = criterion(logits, y)
            loss.backward()
            gradient_clipping(model.parameters(), training_cfg.max_norm)
            optimizer.step()

            loss_val = loss.item()
            iter_loss_list.append(loss_val)
            wandb.log(
                {
                    "each_iter_loss": iter_loss_list[-1],
                    "avg_iter_loss": np.mean(iter_loss_list),
                    "iter_number": iter_number,
                },
                step=iter_number
                )
            epoch_loss += loss_val

            # 每 iter 记录
            if iter_number % training_cfg.iter_print_freq == 0:
                avg_iter_loss = np.mean(iter_loss_list)
                logger.info(f"Epoch {epoch+1}, Iter {iter_number}, Average Iter Loss {avg_iter_loss:.4f}")
                # 保存 checkpoint
                os.makedirs(os.path.join(training_cfg.exp_name, training_cfg.save_path, f"iter_{iter_number}"), exist_ok=True)
                save_checkpoint(model, optimizer, iter_number, os.path.join(training_cfg.exp_name, training_cfg.save_path, f"iter_{iter_number}", f"model_iter_{iter_number}.pth"))
                # 同时记录 avg_iter_loss
                wandb.log({
                    "avg_iter_loss": avg_iter_loss,
                }, step=iter_number)
                iter_loss_list = []  # 可选：清空一下统计列表

            
            # 训练完了一个iter
            total_iters_pbar.update(1)
            iter_number += 1
            
            
        if epoch % training_cfg.epoch_print_freq == 0:
            avg_epoch_loss = epoch_loss / training_cfg.epoch_print_freq
            logger.info(f"Epoch {epoch+1}, Average Loss {avg_epoch_loss:.4f}")
            # 记录 epoch 损失
            wandb.log({
                "epoch": epoch+1,
                "epoch_loss": avg_epoch_loss,
            }, step=iter_number)
            epoch_loss = 0
            
        epoch_pbar.update(1)
        
        scheduler.step(t=epoch)

    logger.info(f"Training finished, total iterations {iter_number}, total time {time.time() - start_time:.2f} seconds")
    wandb.finish()
    return model


def train_transformerLM(args):
    model_cfg = load_model_config(args.model_cfg_path)
    training_cfg = load_training_config(args.exp_cfg_path)
        # 初始化 W&B
    wandb.init(project=training_cfg.exp_name, config={
        "learning_rate": training_cfg.learning_rate,
        "batch_size": training_cfg.batch_size,
        "epochs": training_cfg.epochs,
        "context_length": model_cfg.context_length,
        "max_norm": training_cfg.max_norm,
        "warmup_steps": training_cfg.warmup_steps,
        "cosine_cycle_steps": training_cfg.cosine_cycle_steps,
        "learning_rate_min": training_cfg.learning_rate_min,
        "optimizer": training_cfg.optimizer.type,
        "beta1": training_cfg.optimizer.beta1,
        "beta2": training_cfg.optimizer.beta2,
        "eps": training_cfg.optimizer.eps,
        "weight_decay": training_cfg.optimizer.weight_decay,
    })

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
    

    
    # 上面全部是初始化参数，这里判断时都需要从checkpoint加载
    if training_cfg.resume_from_checkpoint:
        iter_number = load_checkpoint(training_cfg.checkpoint_path, model, optimizer)
        logger.info(f"Resuming training from checkpoint at iteration {iter_number}")
    else:
        iter_number = 0
        logger.info("Starting training from scratch")
    
    # 定义训练器
    trained_model = train(model,
                          train_data_generator,
                          optimizer,
                          criterion,
                          scheduler,
                          training_cfg,
                          iter_number)
    
    return trained_model



if __name__ == "__main__":
    parser = argparse.ArgumentParser("train transformerLM on TinyStoriesV2-GPT4 dataset")
    parser.add_argument("--model_cfg_path", type=str, default="configs/model_configs.yaml")
    parser.add_argument("--exp_cfg_path", type=str, default="configs/exp_configs.yaml")
    args = parser.parse_args()
    train_transformerLM(args)