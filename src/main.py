import argparse
import math
import random

import torch
import torch.optim as optim

from llada.llada import Llada
from llada.utils import TransformerModel
from tokenizer.tokenizer import character_level_tokenizer
from utils import get_batch, sample_datapoint


def train_epoch(
    llada_process,
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
        prompts, target_answers, _, _ = get_batch(
            "train", i, data_train, None, tokenizer, batch_size
        )
        prompts = prompts.to(device)  # (length_prompts, batch_size)
        target_answers = target_answers.to(device)  # (length_answers, batch_size)

        input_tensor = torch.cat(
            (prompts, target_answers), 0
        )  # (length_prompts + length_answers, batch_size)
        llada_process.model.zero_grad()
        mask_ratio = random.random()
        loss = llada_process.train_batch(
            optimizer=optimizer,
            number_bits=number_bits,
            tokens=input_tensor,
            mask_ratio=mask_ratio,
        )
        total_loss += loss if loss is not None else 0

        if batch % freq == 0 and batch > 0:
            cur_loss = total_loss / freq
            print(
                "| {:5d}/{:5d} batches | loss {:5.2f} | perplexity {:8.2f}".format(
                    batch, len(data_train) // batch_size, cur_loss, math.exp(cur_loss)
                )
            )
            total_loss = 0.0
    # return total_loss / (batch + 1)


def evaluate(llada_process, data_test, batch_size, tokenizer, device):
    # Turn on evaluation mode disables dropout.
    correct = 0.0
    with torch.no_grad():
        for batch, i in enumerate(range(0, len(data_test) - 1, batch_size)):
            prompts, target_answers, length_prompts, length_answers = get_batch(
                "test", i, None, data_test, tokenizer, batch_size
            )
            prompts = prompts.to(device)  # (length_prompts, batch_size)
            target_answers = target_answers.to(
                device
            )  # (length_answers + 1, batch_size)
            output = llada_process.sample(
                input_tokens=prompts, seq_len=length_answers + 1, steps=5
            )  # TODO: Check why it should be length_answers + 1
            answers_tokens = output[
                length_prompts:, :
            ]  # (length_answers + 1, batch_size), contains tokens
            equality_test = (
                answers_tokens == target_answers.cpu()
            )  # (length_answers + 1, batch_size), contains boolean values
            correct += torch.all(equality_test, axis=0).float().sum()
        accuracy = correct / len(data_test)
    return accuracy.item()


def main():
    parser = argparse.ArgumentParser(
        description="""A script to train and evaluate a LLaDA-style model.""",
    )

    parser.add_argument(
        "--number_bits",
        type=int,
        default=40,
        help="Number of bits for the addition dataset. If `number_bits` is n, the numbers a and b will each have at most n bits.",
    )

    args = parser.parse_args()

    ### Fixed hyperparameters:
    dataset_size = 64_000  # Hardcoded dataset size.
    train_proportion = 0.9
    # Simple tokenizer with 0, ..., 9, "+", "=", "[PAD]", "[EOS]", "[MASK]".
    vocab_size = None
    seq_len = args.number_bits + 1  # e.g. "12+345="'s result should fit in 7 tokens
    batch_size = 32
    num_steps = 5  # 2000
    learning_rate = 5e-4
    device = "mps"

    data = []
    for i in range(dataset_size):
        data.append(sample_datapoint(args.number_bits))

    data_train = data[: int(train_proportion * dataset_size)]
    data_test = data[int(train_proportion * dataset_size) :]

    tokenizer = character_level_tokenizer(args.number_bits)
    model = TransformerModel(
        ntoken=tokenizer.ntokens, ninp=128, nhead=16, nhid=64, device=device, nlayers=8
    ).to(device)

    print("Initializing model...")
    llada_process = Llada(
        model=model,
        vocab_size=vocab_size,
        mask_token_id=tokenizer.token_to_id["[MASK]"],
        device=device,
    )
    optimizer = optim.AdamW(llada_process.model.parameters(), lr=learning_rate)

    print("Training model on toy addition dataset...")
    llada_process.model.train()

    for step in range(num_steps):
        llada_process.model.train()
        train_epoch(
            llada_process=llada_process,
            optimizer=optimizer,
            data_train=data_train,
            tokenizer=tokenizer,
            batch_size=batch_size,
            number_bits=args.number_bits,
            step=step,
            freq=100,
            device=device,
        )
        # print(f"Final step {step}, total loss: {total_loss}")

        llada_process.model.eval()
        test_accuracy = evaluate(
            llada_process, data_test, batch_size, tokenizer, device
        )
        print("-" * 89)
        print(
            "| end of epoch {:3d} | test accuracy {:5.2f}".format(step, test_accuracy)
        )
        print("-" * 89)

    print("\nSampling from the trained model...")
    # Generate a few examples:
    for j in range(5):
        prompts, target_answers, _, _ = get_batch(
            "test", j, data_train, data_test, tokenizer, batch_size
        )
        sampled_tokens = llada_process.sample(
            input_tokens=prompts, seq_len=seq_len, steps=5
        )
        for i in range(batch_size):
            print(
                "Sampled tokens:",
                tokenizer.decode(sampled_tokens[:, i].numpy().tolist()),
            )
            print(
                "Target tokens:",
                tokenizer.decode(target_answers[:, i].numpy().tolist()),
            )
            print()
        # breakpoint()


if __name__ == "__main__":
    main()
