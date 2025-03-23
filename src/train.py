import os

import torch

from method.utils import get_batch


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


def train(
    method,
    optimizer,
    num_epochs,
    data_train,
    data_test,
    tokenizer,
    batch_size,
    number_bits,
):
    device = method.device
    seq_len = 2 * number_bits + 1

    best_acc = 0
    save_dir = "./weights"
    os.makedirs(save_dir, exist_ok=True)

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
            "| end of epoch {:3d} | test accuracy {:5.2f}".format(e + 1, test_accuracy)
        )
        print("-" * 89)
        # Save the model if it has a better accuracy than the previous best.
        if test_accuracy > best_acc:
            best_acc = test_accuracy
            save_path = os.path.join(save_dir, "{}_best.pth".format(method.name))
            method.save(save_path)
        # Save a copy of the latest model.
        save_path = os.path.join(save_dir, "{}_last.pth".format(method.name))
        method.save(save_path)

    print("\nSampling from the trained model...")
    # Generate a few examples:
    for j in range(5):
        prompts, target_answers, _, _ = get_batch(
            "test", j, data_train, data_test, tokenizer, batch_size
        )

        sampled_tokens = method.sample(input_tokens=prompts, seq_len=seq_len)
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
