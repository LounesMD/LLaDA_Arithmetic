import argparse
import math
import random

import torch
import torch.optim as optim

from llada.llada import Llada
from llada.utils import TransformerModel
from tokenizer.tokenizer import character_level_tokenizer
from utils import train_epoch, evaluate, sample_from_model, parse_arguments
from utils import prepare_data, train_and_evaluate


def main():
    # Parse arguments
    args = parse_arguments()

    # Prepare data
    data_train, data_test = prepare_data(args)

    # Initialize tokenizer
    tokenizer = character_level_tokenizer(args.number_bits)

    # Create model
    model = TransformerModel(
            ntoken=tokenizer.ntokens,
            ninp=128,
            nhead=16,
            nhid=64,
            device=args.device,
            nlayers=8).to(args.device)

    # Initialize LLaDA process
    llada_process = Llada(
                        model=model,
                        vocab_size=None,
                        mask_token_id=tokenizer.token_to_id["[MASK]"],
                        device=args.device
                    )

    optimizer = torch.optim.AdamW(llada_process.model.parameters(), lr=1e-4)

    print("Training model...")
    train_and_evaluate(llada_process, tokenizer, data_train, data_test, args, optimizer)

    print("\nSampling from the trained model...")
    sample_from_model(llada_process, tokenizer, data_train, data_test, batch_size=32, seq_len=args.number_bits+1)

if __name__ == "__main__":
    main()
