import torch
from torch.utils.data import Dataset, DataLoader

class AdditionDataset(Dataset):
    def __init__(self, data, tokenizer):
        self.data = data
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        prompt, answer = self.data[idx]
        prompt_encoded = self.tokenizer.encode(prompt)
        answer_encoded = self.tokenizer.encode(answer)
        return prompt_encoded, answer_encoded

def pad(sequences, tokenizer, seq_type):
    max_length = max(len(seq) for seq in sequences)
    padded_sequences = [seq + [tokenizer.pad_token_id] * (max_length - len(seq)) for seq in sequences]
    lengths = [len(seq) for seq in sequences]
    return padded_sequences, lengths
