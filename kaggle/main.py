import os
import subprocess

# Clone the repository
subprocess.run(
    ["git", "clone", "--branch", "dataloader", "--single-branch", "https://github.com/LounesMD/LLaDA_Arithmetic.git"],
    check=True,
)
os.chdir("LLaDA_Arithmetic")


# Define hyperparameters
method = "llada"
tokenizer = "group_pad"
learning_rate = 5e-4
num_epochs = 5
number_bits = 20
device = "cuda"

# Run the training script
subprocess.run(
    [
        "python",
        "src/main.py",
        "--method",
        method,
        "--tokenizer",
        tokenizer,
        "--learning_rate",
        f"{learning_rate}",
        "--num_epochs",
        f"{num_epochs}",
        "--number_bits",
        f"{number_bits}",
        "--seq_length",
        f"{number_bits+1}",
        "--device",
        device,
    ],
    check=True,
)
