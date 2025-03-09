import random
from typing import Literal

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from llada.utils import add_gumbel_noise


class Llada:
    """
    Naive implementation of the LLaDA process: https://arxiv.org/pdf/2502.09992
    """

    def __init__(
        self,
        model: nn.Module,
        vocab_size: int,
        mask_token_id: int,
        device: Literal["mps", "cuda", "cpu"] = "cpu",
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
        self.mask_token_id = mask_token_id
        self.device = device
        self.criterion = nn.CrossEntropyLoss()

    def random_mask(self, tokens: torch.Tensor, t: float, k: int):
        """
        Randomly mask each token with a probability 't'.
        Returns (masked_tokens, mask_positions).
        """
        # tokens shape: [batch_size, seq_len]
        mask_positions = torch.rand_like(tokens.float()) < t
        mask_positions[:k, :] = False
        masked_tokens = tokens.clone()
        masked_tokens[mask_positions] = self.mask_token_id
        return masked_tokens, mask_positions

    def train_batch(
        self, optimizer, number_bits: int, tokens: torch.Tensor, mask_ratio: float
    ):
        """
        Train on a single batch using a specified mask_ratio in [0,1].
        This represents one “step” or iteration of gradient descent.
        Returns the loss value.
        """
        # mask_ratio = 0.7
        tokens = tokens.to(self.device)
        ### /!\ I mask only the result of the operation /!\ ###
        ### Not the best implementation, but it's a start ###
        k = 2 * number_bits + 1  # Index of the equal sign
        masked_tokens, mask_positions = self.random_mask(tokens, mask_ratio, k)
        # breakpoint()
        output, _ = self.model(
            masked_tokens
        )  # shape: (seq_len, batch_size, vocab_size)
        T, B, C = output.size()

        # Flatten
        logits_flat = output.view(-1, C)  # (B*seq_len, vocab_size)
        targets_flat = tokens.view(-1)  # (B*seq_len,)
        mask_positions_flat = mask_positions.view(-1)  # (B*seq_len,)
        mask_indices = mask_positions_flat.nonzero(as_tuple=True)[0]

        # Select only masked positions for the cross-entropy loss
        logits_masked = logits_flat[mask_indices]
        targets_masked = targets_flat[mask_indices]
        if len(mask_indices) == 0:
            # If no tokens were masked, return None to indicate no update done
            return None

        # loss = self.criterion(logits_masked, targets_masked)
        loss = self.criterion(logits_masked, targets_masked)  # * (1/(1-mask_ratio))
        # print(logits_masked.argmax(-1))
        # print(targets_masked)
        print(loss)
        # breakpoint()
        # print(targets_masked)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        for name, param in self.model.named_parameters():
            if param.grad is None:
                print(f"Warning: {name} has no gradient!")

        return loss.item()

    @torch.no_grad()
    def sample(
        self,
        seq_len: int,
        input_tokens: torch.Tensor,
        steps: int = 5,
        schedule: str = "uniform",
        re_mask_mode: str = "low_confidence",
        temperature: float = 1.0,
        cfg_scale: float = 0.0,
        block_size: int = None,
    ):
        """
        TODO: This version of generate is not autoregressive, and can be improved.
        Use: https://github.com/ML-GSAI/LLaDA/blob/main/generate.py
        V0 sample function for LLaDA model.

        Args:
            seq_len: Length of the sequence to generate.
            input_tokens: Initial input tensor of shape (seq_len, batch_size).
            steps: Number of discrete diffusion steps from t=1 to t=0.
            schedule: "uniform" => linearly spaced t-values.
            re_mask_mode: "random" or "low_confidence".
            temperature: Controls randomness in sampling (lower = deterministic).
            cfg_scale: Scale factor for classifier-free guidance.
            block_size: If provided, enables semi-autoregressive generation.

        Returns:
            A LongTensor of shape (seq_len, batch_size) with final sampled IDs.
        """
        self.model.eval()

        input_tokens = input_tokens.to(self.device)
        batch_size = input_tokens.shape[1]

        # Create a fully masked response sequence
        current_responses = torch.full(
            (seq_len, batch_size),
            self.mask_token_id,
            dtype=torch.long,
            device=self.device,
        )
        current_tokens = torch.cat((input_tokens, current_responses), dim=0)
        # Create time steps
        if schedule == "uniform":
            t_values = torch.linspace(1.0, 0.0, steps + 1).tolist()
        else:
            raise NotImplementedError("Only 'uniform' schedule is implemented.")

        # Step-wise diffusion process
        for i in range(steps):
            t_current = t_values[i]
            t_next = t_values[i + 1]

            # Model forward pass
            if cfg_scale > 0.0:
                # Classifier-Free Guidance: Run model with and without condition
                un_x = current_tokens.clone()
                un_x[
                    input_tokens != self.mask_token_id
                ] = self.mask_token_id  # Remove context
                x_combined = torch.cat(
                    [current_tokens, un_x], dim=1
                )  # Double batch size
                logits = self.model(x_combined)[0]
                logits, un_logits = torch.chunk(
                    logits, 2, dim=1
                )  # Split guided & unconditioned
                logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
            else:
                logits = self.model(current_tokens)[0]

            # Apply temperature scaling with Gumbel noise
            logits_with_noise = add_gumbel_noise(logits, temperature)

            mask_positions = current_tokens == self.mask_token_id

            # We replace the masked tokens by the most predicted ones
            if mask_positions.any():
                predicted_ids = logits_with_noise.argmax(dim=-1)

                # Update masked tokens
                current_tokens[mask_positions] = predicted_ids[mask_positions]

            # Re-mask for diffusion process
            if t_next > 0:
                fraction_to_remask = t_next / max(t_current, 1e-5)

                mask_indices = mask_positions.nonzero(as_tuple=True)[0]
                re_mask_count = int(len(mask_indices) * fraction_to_remask)

                if re_mask_count > 0:
                    if re_mask_mode == "low_confidence":
                        # Remask least confident predictions
                        prob_dist = F.softmax(logits, dim=-1)
                        max_probs, _ = prob_dist.max(dim=-1)
                        sorted_indices = mask_indices[max_probs[mask_indices].argsort()]
                        chosen = sorted_indices[:re_mask_count]
                    else:  # Default: Random remasking
                        chosen = np.random.choice(
                            mask_indices.cpu().numpy(),
                            size=re_mask_count,
                            replace=False,
                        )

                    current_tokens[chosen] = self.mask_token_id

        return current_tokens.cpu()
