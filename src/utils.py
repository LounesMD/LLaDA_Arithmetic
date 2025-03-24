import argparse
from method.arm import ARM
from method.llada import Llada
from tokenizer.tokenizer import naive_tokenizer, naive_pad_tokenizer, group_pad_tokenizer
import torch
from data import AdditionDataset
import random

def sample_datapoint(number_bits=3):
    """
    returns a string containing two random numbers on `number_bits` many bits and their sum.
    """
    a_list = [random.randint(0, 9) for _ in range(number_bits)]
    b_list = [random.randint(0, 9) for _ in range(number_bits)]
    a_int = int("".join([str(x) for x in a_list]))
    b_int = int("".join([str(x) for x in b_list]))
    sum_int = a_int + b_int
    return (str(a_int) + "+" + str(b_int) + "=", str(sum_int))

def prepare_data(args, tokenizer):
    """Prepare the training and testing datasets."""
    data = [sample_datapoint(args.number_bits) for _ in range(args.data_size)]
    X, Y, length_prompts, length_answers = process_data(data, tokenizer)

    dataset = AdditionDataset(X, Y, length_prompts, length_answers)
    return dataset

def parse_arguments():
    parser = argparse.ArgumentParser(
        description="A script to train and evaluate a LLaDA-style model."
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
        choices=["llada", "arm"],
        help="Method between 'llada' and 'arm'.",
    )

    parser.add_argument(
        "--tokenizer",
        type=str,
        default="naive",
        choices=["naive", "naive_pad", "group_pad", "gpt2"],
        help="Tokenizer between 'naive', 'naive_pad' and 'group_pad'.",
    )

    parser.add_argument(
        "--num_epochs",
        type=int,
        default=5,
        help="Number of training steps."
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "mps", "cpu"],
        help="Device to use",
    )

    parser.add_argument("--data_size", type=int, default=64_000, help="Dataset size.")

    parser.add_argument("--batch_size", type=int, default=32, help="Batch size.")

    parser.add_argument(
        "--learning_rate", type=float, default=5e-4, help="Learning rate."
    )

    parser.add_argument(
        "--seq_length",
        type=int,
        default=21,
        help="Defines the length of the sequence to be sampled by Llada.",
    )

    return parser.parse_args()


def initialize_method(method_name, model, vocab_size, tokenizer, device):
    """Initialize the method (ARM or Llada) based on the user input."""
    if method_name == "arm":
        return ARM(
            model=model,
            vocab_size=vocab_size,
            mask_token_id=tokenizer.token_to_id["[MASK]"],
            device=device,
        )
    elif method_name == "llada":
        return Llada(
            model=model,
            vocab_size=vocab_size,
            mask_token_id=tokenizer.token_to_id["[MASK]"],
            device=device,
        )
    else:
        raise ValueError("Invalid method.")


def initialize_tokenizer(tokenizer, number_bits):
    """Initialize tokenizer based on user input."""
    if tokenizer == "naive":
        return naive_tokenizer(number_bits)
    elif tokenizer == "naive_pad":
        return naive_pad_tokenizer(number_bits)
    elif tokenizer == "group_pad":
        return group_pad_tokenizer(number_bits)
    else:
        raise ValueError("Invalid tokenizer.")


def pad(token_list, tokenizer, type_list="prompts"):
    max_length = max([len(x) for x in token_list])
    out = []
    for x in token_list:
        if type_list == "prompts":
            out.append(
                [tokenizer.token_to_id[tokenizer.pad_token]] * (max_length - len(x)) + x
            )
        if type_list == "answers":
            out.append(
                x
                + [tokenizer.token_to_id[tokenizer.eos_token]]
                + [tokenizer.token_to_id[tokenizer.pad_token]] * (max_length - len(x))
            )
    return out, max_length


def process_data(data, tokenizer):
    data = data
    prompts = [tokenizer.encode(data[i][0]) for i in range(len(data))]
    padded_prompts, length_prompts = pad(prompts, tokenizer, "prompts")
    answers = [tokenizer.encode(data[i][1]) for i in range(len(data))]
    padded_answers, length_answers = pad(answers, tokenizer, "answers")
    X = torch.stack([torch.tensor(x) for x in padded_prompts], 1)
    Y = torch.stack([torch.tensor(x) for x in padded_answers], 1)
    return X.permute(1,0), Y.permute(1,0), length_prompts, length_answers
