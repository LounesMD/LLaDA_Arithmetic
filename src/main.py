import argparse
import math
import random

import torch
import torch.optim as optim

from method.arm import ARM
from method.llada import Llada
from method.utils import TransformerModel
from tokenizer.tokenizer import naive_tokenizer, naive_pad_tokenizer, group_pad_tokenizer
from utils import parse_arguments, prepare_data
from train import train


def main():
    args = parse_arguments()

    if args.tokenizer == "naive":
        tokenizer = naive_tokenizer(args.number_bits)
    elif args.tokenizer == "naive_pad":
        tokenizer = naive_pad_tokenizer(args.number_bits)
    elif args.tokenizer == "group_pad":
        tokenizer = group_pad_tokenizer(args.number_bits)
    else:
        raise ValueError("Invalid tokenizer.")

    device = torch.device(args.device)
    learning_rate = 1e-4
    num_epochs = args.num_epochs
    batch_size = 32
    seq_len = 2 * args.number_bits + 1

    # Prepare data
    data_train, data_test = prepare_data(args)


    vocab_size = len(tokenizer.vocab)

    model = TransformerModel(
        ntoken=tokenizer.ntokens, ninp=128, nhead=16, nhid=64, device=device, nlayers=8
    ).to(device)

    print("Initializing model...")
    if args.method == "arm":
        method = ARM(
            model=model,
            vocab_size=vocab_size,
            mask_token_id=tokenizer.token_to_id["[MASK]"],
            device=device,
        )
    elif args.method == "llada":
        method = Llada(
            model=model,
            vocab_size=vocab_size,
            mask_token_id=tokenizer.token_to_id["[MASK]"],
            device=device,
        )
    else:
        raise ValueError("Invalid method")
    optimizer = optim.AdamW(method.model.parameters(), lr=learning_rate)

    print("Training model on toy addition dataset...")
    train(method,optimizer, num_epochs, data_train, data_test, tokenizer,batch_size,args.number_bits,seq_len)


if __name__ == "__main__":
    main()
