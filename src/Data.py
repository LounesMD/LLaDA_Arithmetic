from torch.utils.data import Dataset
import torch

class AdditionDataset(Dataset):
    def __init__(self, data, tokenizer):
        self.data = data
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Tokenize the prompt and answer
        prompt = self.tokenizer.encode(self.data[idx][0])
        answer = self.tokenizer.encode(self.data[idx][1])

        # Pad the sequences
        padded_prompt, length_prompt = self.pad([prompt], self.tokenizer, "prompts")
        padded_answer, length_answer = self.pad([answer], self.tokenizer, "answers")

        # Convert to tensors and return
        X = torch.tensor(padded_prompt[0])
        Y = torch.tensor(padded_answer[0])

        return X, Y, length_prompt, length_answer

    def pad(self, token_list, tokenizer, type_list="prompts"):
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


    import torch
from torch.utils.data import Dataset, DataLoader

class PaddedTextDataset(Dataset):
    def __init__(self, X, Y, length_prompts, length_answers):
        """
        PyTorch Dataset for padded text data.

        Args:
            X (torch.Tensor): Padded input prompts.
            Y (torch.Tensor): Padded target answers.
            length_prompts (list): Lengths of prompts before padding.
            length_answers (list): Lengths of answers before padding.
        """
        self.X = X
        self.Y = Y
        self.length_prompts = length_prompts
        self.length_answers = length_answers

    def __len__(self):
        return len(self.X.T)  # Since X is [seq_len, batch_size], transpose to get correct length

    def __getitem__(self, idx):
        return self.X[:, idx], self.Y[:, idx], self.length_prompts[idx], self.length_answers[idx]
