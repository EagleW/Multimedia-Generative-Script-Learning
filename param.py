# coding=utf-8
# Copyleft 2019 project LXRT.

import argparse
import random

import numpy as np
import torch



def parse_args():
    parser = argparse.ArgumentParser()
    # Data Splits
    parser.add_argument('--test_only', action='store_true')

    # Quick experiments
    parser.add_argument('--train_topk', type=int, default=-1)
    parser.add_argument('--valid_topk', type=int, default=-1)

    # Training Hyper-parameters
    parser.add_argument('--batch_size', type=int, default=10)
    parser.add_argument('--valid_batch_size', type=int, default=None)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--patient', type=int, default=4)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--seed', type=int, default=42, help='random seed')


    # Contrastive
    parser.add_argument('--neg_num_total', type=int, default=4)
    parser.add_argument('--neg_num', type=int, default=2)
    parser.add_argument('--wandb', action='store_true')

    # Model Loading
    parser.add_argument('--load', type=str, default=None,
                        help='Load the model (usually the fine-tuned model).')
    parser.add_argument('--model', type=str, default="bart",
                        help='BART or T5')
                        
    # CPU/GPU
    parser.add_argument('--fp16', action='store_true')
    parser.add_argument("--distributed", action='store_true')


    # Optimization
    parser.add_argument('--min_lr', type=float, default=1e-6)
    parser.add_argument('--lr_mul', type=int, default=1)
    parser.add_argument("--warmup_steps", default=2000, type=int)

    # Pre-training Config
    parser.add_argument("--dataset_dir", default='data', type=str)
    

    # Inference
    parser.add_argument('--num_beams', type=int, default=5)
    parser.add_argument('--output', type=str, default='wikihow_stp')


    # Training configuration

    parser.add_argument('--clip_grad_norm', type=float, default=1.0)
    
    parser.add_argument("--weight_decay", default=0.01, type=float, help="Weight decay if we apply some.")
    parser.add_argument("--adam_eps", default=1e-6, type=float, help="Epsilon for Adam optimizer.")



    parser.add_argument("--start_epoch", default=0, type=int)

    # Parse the arguments.
    args = parser.parse_args()

    # Set seeds
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    return args


if __name__ == '__main__':
    args = parse_args()