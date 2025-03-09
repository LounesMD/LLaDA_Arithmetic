import argparse
import random

import torch
import torch.optim as optim

from llada.llada import Llada
from llada.utils import TransformerModel
from tokenizer.tokenizer import character_level_tokenizer
from utils import get_batch, sample_datapoint


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
    num_steps = 4  # 2000
    learning_rate = 5e-4
    device = "mps"

    data = []
    for _ in range(dataset_size):
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
        total_loss = 0.0
        for batch, i in enumerate(
            range(0, len(data_train) - batch_size - 1, batch_size)
        ):
            prompts, target_answers, length_prompts, length_answers = get_batch(
                "train", i, data_train, data_test, tokenizer, batch_size
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
                number_bits=args.number_bits,
                tokens=input_tensor,
                mask_ratio=mask_ratio,
            )
            total_loss += loss if loss is not None else 0

            if batch % 10 == 0 and batch > 0:
                print(f"Step {step}, batch {batch}, loss: {total_loss / batch}")

        print(f"Step {step}, total loss: {total_loss / (batch+ 1)}")

    print("\nSampling from the trained model...")
    # Generate a few examples:
    for j in range(1):
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
