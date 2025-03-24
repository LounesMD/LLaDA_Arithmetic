import torch.optim as optim

from method.utils import TransformerModel
from train import train
from utils import initialize_method, initialize_tokenizer, parse_arguments, prepare_data


def main():
    # Parse arguments
    args = parse_arguments()

    # Prepare training and test data
    data_train, data_test = prepare_data(args)

    # Initialize tokenizer
    tokenizer = initialize_tokenizer(args.tokenizer, args.number_bits)

    # Initialize model
    model = TransformerModel(
        ntoken=tokenizer.ntokens,
        ninp=128,
        nhead=16,
        nhid=64,
        device=args.device,
        nlayers=8,
    ).to(args.device)

    # Initialize method (ARM or Llada)
    print("Initializing model...")
    method = initialize_method(
        args.method, model, len(tokenizer.vocab), tokenizer, args.device
    )

    # Set up optimizer
    optimizer = optim.AdamW(method.model.parameters(), lr=args.learning_rate)

    # Train the model
    print("Training model on toy addition dataset...")
    train(
        method=method,
        optimizer=optimizer,
        num_epochs=args.num_epochs,
        data_train=data_train,
        data_test=data_test,
        tokenizer=tokenizer,
        batch_size=args.batch_size,
        number_bits=args.number_bits,
        seq_len=args.seq_length,
    )


if __name__ == "__main__":
    main()
