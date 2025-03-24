import subprocess
import os


# Clone the repository
subprocess.run(
    ["git", "clone", "--branch", "dataloader", "--single-branch", "https://github.com/LounesMD/LLaDA_Arithmetic.git"],
    check=True,
)
os.chdir("LLaDA_Arithmetic")


# Define hyperparameters
num_epochs = 5
number_bits = 20
device = "cuda"

# Run the training script
subprocess.run(
    ["python", "src/main.py",
     "--num_epochs", f"{num_epochs}",
     "--number_bits", f"{number_bits}",
     "--device", device,
     "--method", "llada",
     "--tokenizer", "gpt2",
    ],
    check=True,
)
