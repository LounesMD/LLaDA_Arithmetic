from torch.utils.data import Dataset
import torch

class AdditionDataset(Dataset):
    def __init__(self, X, Y, length_prompts, length_answers):
        self.X = X
        self.Y = Y
        self.length_prompts = length_prompts
        self.length_answers = length_answers

    def __len__(self):
        # Return the number of samples in the dataset
        return self.X.shape[0]  # Assuming X and Y have the same number of samples

    def __getitem__(self, idx):
        # Return the sample at index idx
        prompt = self.X[idx]  # Get the prompt (input)
        answer = self.Y[idx]  # Get the answer (output)
        return prompt, answer, self.length_prompts, self.length_answers

class ShakespeareDataset(Dataset):
    def __init__(self, X):
        self.X = X
        self.Y = torch.empty(0)
        self.length_prompts = X.shape[0]
        self.length_answers = X.shape[0]
    def __len__(self):
        # Return the number of samples in the dataset
        return self.X.shape[0]  # Assuming X and Y have the same number of samples

    def __getitem__(self, idx):
        # Return the sample at index idx
        prompt = self.X[idx]  # Get the prompt (input)
        answer = self.Y[idx]  # Get the answer (output)
        return prompt, answer, self.length_prompts, self.length_answers