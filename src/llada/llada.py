import random
import time
from typing import Literal

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from llada.utils import add_gumbel_noise, get_num_transfer_tokens


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
        self.criterion = nn.CrossEntropyLoss(reduction="sum")

    def random_mask(self, tokens: torch.Tensor, t: torch.tensor, k: int):
        """
        Randomly mask each token with a probability 't'.
        't' is the mask ratio across all batches.
        Returns (masked_tokens, mask_positions).
        """
        # tokens shape: [batch_size, seq_len]
        mask_positions = torch.rand_like(tokens, dtype=torch.float32) < t
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
        tokens = tokens.to(self.device)
        ### /!\ I mask only the result of the operation /!\ ###
        ### Not the best implementation, but it's a start ###
        k = 2 * number_bits + 2  # Index of the equal sign
        masked_tokens, mask_positions = self.random_mask(tokens, mask_ratio, k)
        output, _ = self.model(
            masked_tokens
        )  # shape: (seq_len, batch_size, vocab_size)
        _, B, _ = output.size()

        if True:
            # Unvectorized loss. To use it, change the reduction of the criterion to "sum".
            final_loss = 0
            cpt = 0
            for i in range(B):
                if sum(mask_positions[:, i]).item() > 0:
                    final_loss += (
                        self.criterion(
                            output[:, i, :][mask_positions[:, i]],
                            tokens[:, i][mask_positions[:, i]],
                        )
                        / mask_ratio.squeeze(0)[i]
                    )
                    cpt += 1
            if cpt == 0:
                return None
            final_loss /= cpt
        else:
            # Vectorized loss (approx. 4x)
            # Compute loss.
            # 1) Get the masked tokens and predictions (loss elements)
            masked_output = output[mask_positions]
            masked_tokens = tokens[mask_positions]
            # For each masked position, which batch element it belongs to:
            batch_indices = torch.where(mask_positions)[1]

            # 2) Compute per-token loss
            per_token_loss = self.criterion(
                masked_output,
                masked_tokens,
            )
            # 3) Per-batch mask_ratio weighting
            # we have two steps:
            #    - sum up per-token losses for each batch item
            #    - multiply by 1/mask_ratio[i]
            # and mean over batches.

            # Number of masked tokens in each batch item
            counts_per_batch = mask_positions.sum(dim=0)

            # Scatter-add per_token_loss into a length-B tensor that sums losses by batch index.
            # Initialize a (B,) zero Tensor and index_add/scatter_add with batch_indices.
            loss_sum_per_batch = torch.zeros_like(
                counts_per_batch, dtype=per_token_loss.dtype
            )
            loss_sum_per_batch.index_add_(0, batch_indices, per_token_loss)
            # We'll just be careful about dividing by zero if some item i has no masked tokens.
            mask_ratio_squeezed = mask_ratio.squeeze(0)  # ensure shape (B,)
            # Convert counts to float to avoid integer division
            counts_per_batch = counts_per_batch.float()

            # Avoid division-by-zero for batches that have no masked tokens:
            nonzero = counts_per_batch > 0
            if sum(nonzero).item() == 0:
                return None
            loss_sum_per_batch[nonzero] = (
                loss_sum_per_batch[nonzero] / counts_per_batch[nonzero]
            )
            loss_sum_per_batch[nonzero] = (
                loss_sum_per_batch[nonzero] / mask_ratio_squeezed[nonzero]
            )
            # Finally, we take the mean over all individual CE
            final_loss = loss_sum_per_batch.mean()

        optimizer.zero_grad()
        final_loss.backward()
        optimizer.step()

        return final_loss.item()

    @torch.no_grad()
    def sample(
        self,
        seq_len: int,
        input_tokens: torch.Tensor,
        steps: int = 5,
        re_mask_mode: str = "low_confidence",
        temperature: float = 1.0,
        cfg_scale: float = 0.0,
        block_length: int = None,
    ):
        """
        Use: https://github.com/ML-GSAI/LLaDA/blob/main/generate.py

        Input: [p1,...,pn]
        Inner representation: [p1,...,pn, [MASK], [MASK], ...]
        Output: [p1,...,pn, p_{n+1}, ..., p_{n+m}]

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
        x = torch.cat((input_tokens, current_responses), dim=0)
        prompt_index = x != self.mask_token_id

        # TODO: Try to scale the output generation to make generation by blocks
        # assert seq_len % block_length == 0
        # num_blocks = seq_len // block_length

        # assert steps % num_blocks == 0
        # steps = steps // num_blocks
        num_blocks = 1
        block_length = seq_len
        for num_block in range(num_blocks):
            block_mask_index = (
                x[
                    input_tokens.shape[0]
                    + num_block * block_length : input_tokens.shape[0]
                    + (num_block + 1) * block_length :
                ]
                == self.mask_token_id
            )
            num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps)
            # Step-wise diffusion process
            for i in range(steps):
                mask_index = x == self.mask_token_id
                # Model forward pass
                if cfg_scale > 0.0:
                    # TODO: Check why x_ = torch.cat([x, un_x], dim=0) ?
                    un_x = x.clone()
                    un_x[prompt_index] = self.mask_token_id
                    x_ = torch.cat([x, un_x], dim=0)
                    logits = self.model(x_)[0]
                    logits, un_logits = torch.chunk(logits, 2, dim=0)
                    logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
                else:
                    logits = self.model(x)[0]

                # Apply temperature scaling with Gumbel noise
                logits_with_noise = add_gumbel_noise(logits, temperature)
                x0 = torch.argmax(logits_with_noise, dim=-1)

                if re_mask_mode == "low_confidence":
                    p = F.softmax(logits.to(torch.float32), dim=-1)
                    x0_p = torch.squeeze(
                        torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1
                    )  # b, l
                elif re_mask_mode == "random":
                    x0_p = torch.rand((x0.shape[0], x0.shape[1]), device=x0.device)
                else:
                    raise NotImplementedError(re_mask_mode)
                x0_p[
                    input_tokens.shape[0] + (num_block + 1) * block_length :, :
                ] = -np.inf

                x0 = torch.where(mask_index, x0, x)
                confidence = torch.where(mask_index, x0_p, -np.inf)

                transfer_index = torch.zeros_like(
                    x0, dtype=torch.bool, device=x0.device
                )
                # breakpoint()
                for j in range(confidence.shape[1]):
                    _, select_index = torch.topk(
                        confidence[:, j], k=num_transfer_tokens[i, j]
                    )
                    transfer_index[select_index, j] = True
                x[transfer_index] = x0[transfer_index]
        return x.cpu()
