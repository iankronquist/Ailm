#!/usr/bin/env python3

'''Command line tool to run validation against different models and datasets.'''

import yaml
from train import  DataLoaderConfig, BatchConfig, TokenizingConfig, create_data_loader, create_tokenizer, create_encoder_worker, create_model, loss_fn, validate

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--sequence-length', type=int, default=1024)
    parser.add_argument('--ubatch-size', type=int, default=4)
    parser.add_argument('--batch-count', type=int, default=4)
    parser.add_argument('--data-set-kind', type=str, default=None)
    parser.add_argument('--data-set-directory', type=str, default=None)
    parser.add_argument('--data-set-shuffle', type=int, default=None)
    parser.add_argument('--model-config', type=str)
    parser.add_argument('model', type=str)
    args = parser.parse_args()

    # Batch config does *not* come from original training config file because we might want to optimize for a different memory footprint since we don't need to load the optimizer.
    batch_size = args.batch_count * args.ubatch_size * args.sequence_length
    batch_config = BatchConfig(batch_size, args.ubatch_size, args.sequence_length)

    # Load the training config file. 
    with open(args.model_config) as config_file:
        training_config = yaml.safe_load(config_file)
    # Tokenizer config comes straight from the training file. Mixing and matching tokenizers can only get you into trouble.
    tokenizer_config = TokenizingConfig(**training_config['tokenizing'])
    # Data loader config starts with the configuration file and applies additional overrides from the command line.
    data_loader_config_from_file = training_config['validation_data_loader']
    if args.data_set_shuffle is not None:
        data_loader_config_from_file['shuffle'] = args.data_set_shuffle
    if args.data_set_kind is not None:
        data_loader_config_from_file['kind'] = args.data_set_kind
    if args.data_set_directory is not None:
        data_loader_config_from_file['directory'] = args.data_set_directory
    validation_loader_config = DataLoaderConfig(**data_loader_config_from_file)

    model, _model, _model_config = create_model(training_config, args.model, None,)
    tokenizer = create_tokenizer(tokenizer_config)
    validation_loader = create_data_loader(validation_loader_config, batch_config, tokenizer, tokenizer_config, resume_config=None)
    encoding_worker_queue = create_encoder_worker(tokenizer_config, validation_loader, 'val')
    try:
        result = validate(model, validation_loader, encoding_worker_queue, args.batch_count, True, 500, loss_fn)
        print(result)
    finally:
        # The Python interpreter will not exit until the encoder worker thread shuts down.
        encoding_worker_queue.shutdown()
