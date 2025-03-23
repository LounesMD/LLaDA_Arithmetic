import subprocess
import os


# Clone a specific branch of the repository
subprocess.run(
    ["git", "clone", "--branch", "your-branch-name", "--single-branch", "https://github.com/LounesMD/LLaDA_Arithmetic.git"],
    check=True,
)

os.chdir("LLaDA_Arithmetic")

# Define hyperparameters
number_steps = 5
number_bits = 20
device = "cuda"

# Run the training script
subprocess.run(
    ["python", "src/main.py",
     "--number_steps", f"{number_steps}",
     "--number_bits", f"{number_bits}",
     "--device", device,
    ],
    check=True,
)
