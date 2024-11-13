import transformers
import torch
import os
import json
import random
import argparse
import numpy as np
from datetime import datetime
from torch.nn import DataParallel
from tqdm import tqdm

# This function processes the raw training data and splits it into tokenized segments
def build_files(raw_data_path, tokenized_data_path, tokenizer, num_pieces):
    # Read and preprocess raw data
    with open(raw_data_path, 'r', encoding='utf8') as f:
        print('Reading data from raw file...')
        lines = json.load(f)
        lines = [line.replace('\n', ' [SEP] ') for line in lines]  # Replace newline characters with [SEP] tokens
    
    # Concatenate all lines into a single string
    full_text = ''.join(lines)
    len_full_text = len(full_text)

    # Create tokenized data path if it doesn't exist
    if not os.path.exists(tokenized_data_path):
        os.makedirs(tokenized_data_path)

    # Split data into multiple pieces and tokenize
    print('Tokenizing data and saving to files...')
    for i in tqdm(range(num_pieces)):
        start_idx = len_full_text // num_pieces * i
        end_idx = len_full_text // num_pieces * (i + 1)
        tokenized_ids = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(full_text[start_idx:end_idx]))

        # Write tokenized IDs to a file
        with open(f'{tokenized_data_path}/tokenized_train_{i}.txt', 'w') as tokenized_file:
            tokenized_file.write(' '.join(map(str, tokenized_ids)))

    print('Tokenization complete.')


def main():
    # Argument parsing
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='0', type=str, help='Specify GPU devices to use, e.g., "0,1,2,3"')
    parser.add_argument('--model_config', default='config/model_config_small.json', type=str, help='Path to model config file')
    parser.add_argument('--tokenizer_path', default='cache/vocab_small.txt', type=str, help='Path to vocabulary file')
    parser.add_argument('--raw_data_path', default='data/train.json', type=str, help='Path to raw training data')
    parser.add_argument('--tokenized_data_path', default='data/tokenized/', type=str, help='Directory to save tokenized data')
    parser.add_argument('--raw', action='store_true', help='Flag to indicate if data needs to be tokenized first')
    parser.add_argument('--epochs', default=5, type=int, help='Number of training epochs')
    parser.add_argument('--batch_size', default=8, type=int, help='Batch size for training')
    parser.add_argument('--lr', default=1.5e-4, type=float, help='Learning rate')
    parser.add_argument('--warmup_steps', default=2000, type=int, help='Number of warmup steps for learning rate scheduler')
    parser.add_argument('--log_step', default=1, type=int, help='Log training loss every specified number of steps')
    parser.add_argument('--stride', default=768, type=int, help='Window stride for splitting training data')
    parser.add_argument('--gradient_accumulation', default=1, type=int, help='Number of gradient accumulation steps')
    parser.add_argument('--fp16', action='store_true', help='Use mixed precision training if set')
    parser.add_argument('--fp16_opt_level', default='O1', type=str, help='Optimization level for mixed precision training')
    parser.add_argument('--max_grad_norm', default=1.0, type=float, help='Maximum gradient norm for clipping')
    parser.add_argument('--num_pieces', default=100, type=int, help='Number of pieces to split data into')
    parser.add_argument('--output_dir', default='model/', type=str, help='Directory to save trained models')
    parser.add_argument('--pretrained_model', default='', type=str, help='Path to a pre-trained model')
    parser.add_argument('--segment', action='store_true', help='Use word-level tokenization if set')

    args = parser.parse_args()
    print(f'Arguments: {args}')

    # Import appropriate tokenizer based on the argument
    if args.segment:
        from tokenizations import tokenization_bert_word_level as tokenization_bert
    else:
        from tokenizations import tokenization_bert

    # Set GPU device visibility
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Using device: {device}')

    # Load model configuration and tokenizer
    model_config = transformers.GPT2Config.from_json_file(args.model_config)
    tokenizer = tokenization_bert.BertTokenizer(vocab_file=args.tokenizer_path)
    tokenizer.max_len = 999999

    # Tokenize data if raw flag is set
    if args.raw:
        print('Starting tokenization of raw data...')
        build_files(raw_data_path=args.raw_data_path, tokenized_data_path=args.tokenized_data_path, tokenizer=tokenizer, num_pieces=args.num_pieces)
        print('Data tokenization complete.')

    # Load or initialize model
    if not args.pretrained_model:
        model = transformers.GPT2LMHeadModel(config=model_config)
    else:
        model = transformers.GPT2LMHeadModel.from_pretrained(args.pretrained_model)
    model.to(device)
    model.train()

    # Calculate total steps for training
    print('Calculating total training steps...')
    total_tokens = 0
    for i in tqdm(range(args.num_pieces)):
        with open(f'{args.tokenized_data_path}/tokenized_train_{i}.txt', 'r') as f:
            total_tokens += len(f.read().strip().split())
    total_steps = (total_tokens // args.stride * args.epochs) // (args.batch_size * args.gradient_accumulation)
    print(f'Total steps: {total_steps}')

    # Optimizer and learning rate scheduler
    optimizer = transformers.AdamW(model.parameters(), lr=args.lr, correct_bias=True)
    scheduler = transformers.get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=total_steps)

    # Mixed precision setup
    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)

    # Multi-GPU setup
    if torch.cuda.device_count() > 1:
        model = DataParallel(model)
        print(f'Using {torch.cuda.device_count()} GPUs for training')

    # Training loop
    print('Starting training...')
    for epoch in range(args.epochs):
        print(f'Epoch {epoch + 1}/{args.epochs}')
        epoch_start_time = datetime.now()
        piece_indices = list(range(args.num_pieces))
        random.shuffle(piece_indices)

        for piece_idx in piece_indices:
            with open(f'{args.tokenized_data_path}/tokenized_train_{piece_idx}.txt', 'r') as f:
                tokens = list(map(int, f.read().strip().split()))

            start = 0
            samples = []
            while start < len(tokens) - model_config.n_ctx:
                samples.append(tokens[start:start + model_config.n_ctx])
                start += args.stride
            if start < len(tokens):
                samples.append(tokens[-model_config.n_ctx:])

            random.shuffle(samples)

            for step in range(0, len(samples), args.batch_size):
                batch_samples = samples[step:step + args.batch_size]
                batch_labels = torch.tensor(batch_samples).long().to(device)
                batch_inputs = batch_labels.clone()

                outputs = model(input_ids=batch_inputs, labels=batch_labels)
                loss = outputs.loss

                if torch.cuda.device_count() > 1:
                    loss = loss.mean()
                if args.gradient_accumulation > 1:
                    loss /= args.gradient_accumulation

                if args.fp16:
                    with amp.scale_loss(loss, optimizer) as scaled_loss:
                        scaled_loss.backward()
                    torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                if (step // args.batch_size + 1) % args.gradient_accumulation == 0:
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()

                if (step // args.batch_size + 1) % args.log_step == 0:
                    print(f'[{datetime.now().strftime("%H:%M:%S")}] Step {step // args.batch_size + 1} of epoch {epoch + 1}, Loss: {loss.item()}')

        # Save model at the end of each epoch
        epoch_dir = f'{args.output_dir}/model_epoch{epoch + 1}'
        if not os.path.exists(epoch_dir):
            os.makedirs(epoch_dir)
        model_to_save = model.module if hasattr(model, 'module') else model
        model_to_save.save_pretrained(epoch_dir)
        print(f'Epoch {epoch + 1} complete. Model saved to {epoch_dir}')
        print(f'Epoch time: {datetime.now() - epoch_start_time}')

    # Save final model
    final_model_dir = f'{args.output_dir}/final_model'
    if not os.path.exists(final_model_dir):
        os.makedirs(final_model_dir)
    model_to_save.save_pretrained(final_model_dir)
    print(f'Training complete. Final model saved to {final_model_dir}')


if __name__ == '__main__':
    main()
