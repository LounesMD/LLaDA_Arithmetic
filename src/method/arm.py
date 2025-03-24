import random
import time
from typing import Literal

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from method.utils import add_gumbel_noise, get_num_transfer_tokens


class ARM:
    """
    Autoregressive Model
    """

    def __init__(
        self,
        model: nn.Module,
        vocab_size: int,
        mask_token_id: int,
        device: Literal["mps", "cuda", "cpu"] = "cpu",
        name: str = "ARM",
    ):
        """
        Args:
            model: A Transformer-based mask predictor (xxx).
            vocab_size: Size of the vocabulary (used in cross-entropy).
            mask_token_id: ID for the [MASK] token in the vocabulary.
            device: ["cuda", "mps", "cpu"].

        """
        self.model = model  # .to(device)
        self.vocab_size = vocab_size
        self.device = device
        self.criterion = nn.CrossEntropyLoss()
        self.name = name

    def train_batch(self, optimizer, number_bits, tokens, prompt_length):
        self.model.zero_grad()
        prompt_length = prompt_length[0]
        output, _ = self.model(
            tokens
        )  # (prompt_length + answers_length + 1, batch_size, ntokens)
        output_answers = output[prompt_length - 1 : -1, :, :].reshape(
            -1, self.vocab_size
        )  # ((answers_length + 1) * batch_size, ntokens)
        target_answers = tokens[prompt_length:, :].reshape(
            -1
        )  # ((answers_length + 1) * batch_size)
        loss = self.criterion(output_answers, target_answers)
        loss.backward()
        optimizer.step()
        return loss.item()

    @torch.no_grad()
    def sample(
        self, input_tokens, seq_len=5, mode="greedy", num_samples=1, temperature=0.8
    ):
        input_tensor = torch.repeat_interleave(
            input_tokens, repeats=num_samples, dim=1
        ).to(self.device)
        # (prompt_length, batch_size * num_samples)
        for _ in range(seq_len):
            output, _ = self.model(
                input_tensor
            )  # (prompt_length, batch_size * num_samples, ntokens)
            logits = output[-1, :, :]  # (batch_size * num_samples, ntokens)
            if mode == "greedy":
                tokens = torch.argmax(logits, -1).view(
                    (1, -1)
                )  # (1, batch_size * num_samples)
            else:  # mode == "sampling"
                logits /= temperature
                probs = torch.softmax(logits, dim=-1)
                tokens = torch.multinomial(probs, num_samples=1).view(
                    (1, -1)
                )  # (1, batch_size * num_samples)
            input_tensor = torch.cat((input_tensor, tokens), 0)
        return input_tensor

    @torch.no_grad()
    def evaluate(self,test_loader, batch_size,tokenizer):
        # Turn on evaluation mode disables dropout.
        self.model.eval()
        correct = 0.
        for batch, (prompts, target_answers, prompt_length, length_answers) in enumerate(test_loader):
            prompts = prompts.to(self.device).permute(1, 0) # (prompt_length, batch_size)
            target_answers = target_answers.to(self.device).permute(1, 0) # (answers_length + 1, batch_size)

            # TODO: Improve this hardcoded part
            length_answers = length_answers[0]
            prompt_length = prompt_length[0]

            output = self.sample(prompts, length_answers + 1) # (prompt_length + answers_length + 1, batch_size)
            answers_tokens = output[prompt_length:, :] # (answers_length + 1, batch_size), contains tokens
            equality_test = answers_tokens == target_answers # (answers_length + 1, batch_size), contains boolean values
            correct += torch.all(equality_test, axis=0).float().sum()
        accuracy = correct / len(test_loader.dataset)
        return accuracy.item()

    def save(self, path):
        torch.save(self.model.state_dict(), path)

    def load(self, path):
        self.model.load_state_dict(torch.load(path))
