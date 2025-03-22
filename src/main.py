import argparse
import math
import random

import torch
import torch.optim as optim

from method.arm import ARM
from method.llada import Llada
from method.utils import TransformerModel
from tokenizer.tokenizer import naive_tokenizer, naive_pad_tokenizer, group_pad_tokenizer
from utils import get_batch, sample_datapoint


def train_epoch(
    method,
    optimizer,
    data_train,
    tokenizer,
    batch_size,
    number_bits,
    step,
    freq,
    device,
):
    ### Train the model for one epoch.
    total_loss = 0.0
    for batch, i in enumerate(range(0, len(data_train) - batch_size - 1, batch_size)):
        prompts, target_answers, prompt_length, _ = get_batch(
            "train", i, data_train, None, tokenizer, batch_size
        )
        prompts = prompts.to(device)
        target_answers = target_answers.to(device)

        input_tensor = torch.cat((prompts, target_answers), 0)
        method.model.zero_grad()

        loss = method.train_batch(
            optimizer=optimizer,
            number_bits=number_bits,
            tokens=input_tensor,
            prompt_length=prompt_length,
        )
        total_loss += loss if loss is not None else 0

        if batch % freq == 0 and batch > 0:
            cur_loss = total_loss / freq
            print(
                "| {:5d}/{:5d} batches | loss {:5.2f}".format(
                    batch, len(data_train) // batch_size, cur_loss
                )
            )
            total_loss = 0.0
    # return total_loss / (batch + 1)

def train(method,optimizer,num_epochs, data_train, data_test, tokenizer,batch_size,number_bits,seq_len):
    device = method.device
    for e in range(num_epochs):
        method.model.train()
        train_epoch(
            method=method,
            optimizer=optimizer,
            data_train=data_train,
            tokenizer=tokenizer,
            batch_size=batch_size,
            number_bits=number_bits,
            step=e,
            freq=100,
            device=device,
        )

        method.model.eval()
        test_accuracy = method.evaluate(data_test, batch_size, tokenizer)
        print("-" * 89)
        print(
            "| end of epoch {:3d} | test accuracy {:5.2f}".format(
                e + 1, test_accuracy
            )
        )
        print("-" * 89)

    print("\nSampling from the trained model...")
    # Generate a few examples:
    for j in range(5):
        prompts, target_answers, _, _ = get_batch(
            "test", j, data_train, data_test, tokenizer, batch_size
        )

        sampled_tokens = method.sample(
            input_tokens=prompts, seq_len=seq_len
        )
        for i in range(batch_size):
            print(
                "Sampled tokens:",
                tokenizer.decode(sampled_tokens[:, i].cpu().numpy().tolist()),
            )
            print(
                "Target tokens:",
                tokenizer.decode(target_answers[:, i].cpu().numpy().tolist()),
            )
            print()




def main():
    parser = argparse.ArgumentParser(
        description="""A script to train and evaluate a LLaDA-style model.""",
    )

    parser.add_argument(
        "--number_bits",
        type=int,
        default=15,
        help="Number of bits for the addition dataset. If `number_bits` is n, the numbers a and b will each have at most n bits.",
    )

    parser.add_argument(
        "--method",
        type=str,
        default="llada",
        help="Method between 'llada' and 'arm'.",
    )

    parser.add_argument(
        "--tokenizer",
        type=str,
        default="naive",
        help="Tokenizer between 'naive', 'naive_pad' and 'group_pad'.",
    )
 
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=5,
        help="Number of training steps.",
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use",
    )

    args = parser.parse_args()

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

    if args.tokenizer == "naive":
        tokenizer = naive_tokenizer(args.number_bits)
    elif args.tokenizer == "naive_pad":
        tokenizer = naive_pad_tokenizer(args.number_bits)
    elif args.tokenizer == "group_pad":
        tokenizer = group_pad_tokenizer(args.number_bits)
    else:
        raise ValueError("Invalid tokenizer.")

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
