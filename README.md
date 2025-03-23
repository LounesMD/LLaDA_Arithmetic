# MVA Large Language Models Course - Project

This repository contains the code for the project of the LLM course of the MVA master at ENS Paris-Saclay.

## Installation

Install the dependencies:

```bash
conda env create -f environment.yaml
conda activate llm_project
```

### Run the project
To run train and run LLaDA:
```bash
python src/main.py --method llada --tokenizer group_pad --number_epochs 5 --number_bits 20
```

To train the model using Kaggle's GPU, ensure you have a Kaggle account and API key, adapt the `kaggle/kernel-metadata.json` file to your Kaggle username, and run:
```bash
kaggle kernels push -p kaggle/
```


## Acknowledgements

We would like to thank:
* The authors of the paper [Large Language Diffusion Models](https://ml-gsai.github.io/LLaDA-demo/) from which we are basing this project.

## Contact and Contributors

This project is conducted by: [Nicolas Sereyjol-Garros](), [Tom Ravaud](), [Christopher Marouani](), and [Lounès Meddahi]().