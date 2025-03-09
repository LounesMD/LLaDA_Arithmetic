import random

import torch


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


def get_batch(split, i, data_train, data_test, tokenizer, batch_size):
    data = data_train if split == "train" else data_test
    prompts = [tokenizer.encode(data[i][0]) for i in range(i, i + batch_size)]
    padded_prompts, length_prompts = pad(prompts, tokenizer, "prompts")
    answers = [tokenizer.encode(data[i][1]) for i in range(i, i + batch_size)]
    padded_answers, length_answers = pad(answers, tokenizer, "answers")
    X = torch.stack([torch.tensor(x) for x in padded_prompts], 1)
    Y = torch.stack([torch.tensor(x) for x in padded_answers], 1)
    return X, Y, length_prompts, length_answers
