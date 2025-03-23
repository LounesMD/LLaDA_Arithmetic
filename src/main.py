import argparse
import math
import random

import torch
import torch.optim as optim

from method.arm import ARM
from method.llada import Llada
from method.utils import TransformerModel
from tokenizer.tokenizer import naive_tokenizer, naive_pad_tokenizer, group_pad_tokenizer
from utils import sample_datapoint
from train import train
from utils import parse_arguments


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

    ### Fixed hyperparameters:
    dataset_size = 64_000  # Hardcoded dataset size.
    train_proportion = 0.9
    # Simple tokenizer with 0, ..., 9, "+", "=", "[PAD]", "[EOS]", "[MASK]".
    seq_len = args.number_bits + 1  # e.g. "12+345="'s result should fit in 7 tokens
    batch_size = 32
    num_epochs = args.num_epochs
    learning_rate = 5e-4
    device = args.device

    data = []
    for i in range(dataset_size):
        data.append(sample_datapoint(args.number_bits))

    data_train = data[: int(train_proportion * dataset_size)]
    data_test = data[int(train_proportion * dataset_size) :]


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
    train(method,optimizer,num_epochs, data_train, data_test, tokenizer,batch_size,args.number_bits,seq_len)


if __name__ == "__main__":
    main()
