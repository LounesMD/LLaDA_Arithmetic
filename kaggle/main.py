import subprocess
import os

subprocess.run(["git", "clone", "https://github.com/LounesMD/LLaDA_Arithmetic.git"], check=True)
os.chdir("LLaDA_Arithmetic")

subprocess.run(["python", "src/main.py", "--number_steps", "5", "--number_bits", "20", "--device", "cuda"], check=True)
