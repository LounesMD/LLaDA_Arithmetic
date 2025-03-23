import torch.optim as optim

from method.utils import TransformerModel
from utils import parse_arguments, prepare_data, initialize_method, initialize_tokenizer
from train import train

def main():
    # Parse arguments
    args = parse_arguments()

    # Prepare training and test data
    data_train, data_test = prepare_data(args)

    # Initialize tokenizer
    tokenizer = initialize_tokenizer(args.tokenizer, args.number_bits)

    # Initialize model
    model = TransformerModel(
        ntoken=tokenizer.ntokens, ninp=128, nhead=16, nhid=64, device=args.device, nlayers=8
    ).to(args.device)

    # Initialize method (ARM or Llada)
    print("Initializing model...")
    method = initialize_method(args.method, model, len(tokenizer.vocab), tokenizer, args.device)

    # Set up optimizer
    learning_rate = 1e-4
    optimizer = optim.AdamW(method.model.parameters(), lr=learning_rate)

    # Train the model
    print("Training model on toy addition dataset...")
    train(method, optimizer, args.num_epochs, data_train, data_test, tokenizer, args.batch_size, args.number_bits)


if __name__ == "__main__":
    main()
