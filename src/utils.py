import random
import argparse
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
    n_batch = len(data_train)/batch_size
    for batch, i in enumerate(range(0, len(data_train) - batch_size - 1, batch_size)):
        prompts, target_answers, _, _ = get_batch(
            "train", i, data_train, None, tokenizer, batch_size
        )
        prompts = prompts.to(device)
        target_answers = target_answers.to(device)

        input_tensor = torch.cat((prompts, target_answers), 0)
        llada_process.model.zero_grad()

        # As many mask ratios as the batch_size
        mask_ratio = torch.rand((1, input_tensor.size(1))).to(device=device)

        loss = llada_process.train_batch(
            optimizer=optimizer,
            number_bits=number_bits,
            tokens=input_tensor,
            mask_ratio=mask_ratio,
        )
        total_loss += loss if loss is not None else 0

        if batch % 3 == 0 and batch > 0:
            cur_loss = total_loss / freq
            print(
                "| {:5d}/{:5d} batches | loss {:5.2f}".format(
                    batch, len(data_train) // batch_size, cur_loss
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
            prompts = prompts.to(device)

            target_answers = target_answers.to(device)

            # TODO: Check why it should be length_answers + 1
            output = llada_process.sample(
                input_tokens=prompts, seq_len=length_answers + 1, steps=5
            )

            answers_tokens = output[length_prompts:, :]

            equality_test = answers_tokens == target_answers.cpu()
            correct += torch.all(equality_test, axis=0).float().sum()

        accuracy = correct / len(data_test)
    return accuracy.item()

def sample_from_model(llada_process, tokenizer, data_train, data_test, batch_size, seq_len):
    """Sample from the trained model."""
    for j in range(5):
        prompts, target_answers, _, _ = get_batch("test", j, data_train, data_test, tokenizer, batch_size)
        sampled_tokens = llada_process.sample(input_tokens=prompts, seq_len=seq_len, steps=5)
        for i in range(batch_size):
            print("Sampled tokens:", tokenizer.decode(sampled_tokens[:, i].numpy().tolist()))
            print("Target tokens:", tokenizer.decode(target_answers[:, i].numpy().tolist()))
            print()


def parse_arguments():
    """Parse the command-line arguments."""
    parser = argparse.ArgumentParser(description="Train and evaluate a LLaDA-style model.")
    parser.add_argument("--number_bits", type=int, default=2, help="Number of bits for the addition dataset.")
    parser.add_argument("--number_steps", type=int, default=10, help="Number of training steps.")
    parser.add_argument("--data_size", type=int, default=64_000, help="Dataset size.")
    parser.add_argument("--device", type=str, default="mps", help="Device to use.")
    parser.add_argument("--model_type", type=str, default="transformer", help="Type of model to use (e.g., 'transformer').")
    return parser.parse_args()


def prepare_data(args):
    """Prepare the training and testing datasets."""
    data = [sample_datapoint(args.number_bits) for _ in range(args.data_size)]
    train_proportion = 0.9
    data_train = data[: int(train_proportion * args.data_size)]
    data_test = data[int(train_proportion * args.data_size):]
    return data_train, data_test


def train_and_evaluate(llada_process, tokenizer, data_train, data_test, args, optimizer):
    """Train and evaluate the model."""
    num_steps = args.number_steps
    batch_size = 32
    seq_len = args.number_bits + 1

    for step in range(num_steps):
        # Train epoch
        llada_process.model.train()
        train_epoch(
            llada_process=llada_process,
            optimizer=optimizer,
            data_train=data_train,
            tokenizer=tokenizer,
            batch_size=batch_size,
            number_bits=args.number_bits,
            step=step,
            freq=5,
            device=args.device
        )

        # Evaluate
        llada_process.model.eval()
        test_accuracy = evaluate(llada_process, data_test, batch_size, tokenizer, args.device)
        print(f"-" * 89)
        print(f"| end of epoch {step + 1:3d} | test accuracy {test_accuracy:5.2f}")
        print(f"-" * 89)
