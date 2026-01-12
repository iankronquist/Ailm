import random
import queue
import threading
import time
import typing
import mlx.core as mx
import tiktoken
from model import AilmV1, AilmV1Config
import yaml

# MARK: Sampling

@mx.compile
def sample_topk(logits: mx.array, temperature: float=0.5, k: int = 5) -> mx.array:
    B, V = logits.shape
    logits /= temperature
    logits = mx.softmax(logits, axis=-1)
    args = mx.argpartition(logits, k, axis=-1)[..., -k:]

    random = mx.random.randint(0, k, shape=(B,))
    batch_idx = mx.arange(B)
    selected = args[batch_idx, random]
    return selected

@mx.compile
def sample_greedy(logits: mx.array, temperature: float=0.5, k: int = 5) -> mx.array:
    token_ids = logits.argmax(axis=-1)
    return token_ids

# Translate names for strategies to our MLX implementations.
SAMPLE_STRATEGIES = {
        'topk': sample_topk,
        'greedy': sample_greedy,
        }

def decode_and_print_worker(tokenizer: tiktoken.Encoding, work_queue: queue.Queue):
    '''A worker thread so we can decode tokens and print them off of the main thread while training is happening.'''
    while True:
        try:
            batched_tokens = work_queue.get()
            tokens_list  = batched_tokens
            tokens_list = typing.cast(list[list[int]], tokens_list)
            decoded = tokenizer.decode_batch(tokens_list)
            print('\n----\n'.join(decoded))
        except KeyError as e:
            print(e)
            decoded = ''
        except queue.ShutDown:
            break

class InferenceManager:
    '''In the spirit of removing as much work as possible from the training thread, this inference manager only encodes the prompt once, and performs decoding and printing on a separate thread.'''
    def __init__(self, model: AilmV1, tokenizer: tiktoken.Encoding, max_tokens: int, max_batches: int, prompt: str, k: int, temperature: float, sample_strategy: str='topk', enable_kv_cache: bool = True):
        self.model = mx.compile(model)
        self.tokenizer = tokenizer
        self.max_tokens = max_tokens
        self.max_batches = max_batches
        self.k = k
        self.prompt = prompt
        self.temperature = temperature
        self.enable_kv_cache = enable_kv_cache

        tokenized_prompt = mx.array(tokenizer.encode(self.prompt))

        self.prompt_tokens = len(tokenized_prompt)
        max_length = max_tokens + self.prompt_tokens
        self.batched_tokens = mx.zeros((max_batches, max_length), dtype=mx.uint16)
        self.batched_tokenized_prompt = mx.zeros((max_batches, self.prompt_tokens), dtype=mx.uint16)
        self.batched_tokens[:, 0:self.prompt_tokens] = tokenized_prompt
        self.batched_tokenized_prompt[:, 0:self.prompt_tokens] = tokenized_prompt

        sample_function = SAMPLE_STRATEGIES[sample_strategy]

        def do_infer(batch: mx.array):

            logits = model(batch)
            logits = logits[:, -1]
            chosen = sample_function(logits, temperature, k)
            return chosen

        self.do_infer = do_infer

        self.decoder_queue: queue.Queue[mx.array] = queue.Queue(1)
        self.worker = threading.Thread(name='inference decoder', target=decode_and_print_worker, args=(tokenizer, self.decoder_queue,))
        self.worker.start()

    def finish(self):
        '''Shutdown the work queue if we are ever deleted. This will cause the worker thread to finish.'''
        self.decoder_queue.shutdown()
        self.worker.join()

    def infer(self):
        self.model.train(False)
        self.model.reset_key_value_cache(self.enable_kv_cache)
        
        # Start with the full prompt
        todos = self.batched_tokenized_prompt
        
        for step in range(self.max_tokens):
            if self.enable_kv_cache:
                if step == 0:
                    batch = todos
                else:
                    batch = todos[:, -1:]
            else:
                batch = todos

            chosen = self.do_infer(batch)

            # Append to our running sequence
            todos = mx.concatenate([todos, chosen[:, None]], axis=-1)
        self.model.reset_key_value_cache(False)
        self.model.train(True)
        self.decoder_queue.put(todos.tolist())


# MARK: Inference   

def infer_tokenize(prompt: str, max_tokens, max_batches) -> mx.array:
    # Tokenize the prompt and batch it up.
    tokens = tokenizer.encode(prompt)
    tokens = mx.array(tokens)
    prompt_tokens = len(tokens)
    max_length = max_tokens + prompt_tokens
    batched_tokens = mx.zeros((max_batches, max_length), dtype=mx.uint16)
    batched_tokens[:, 0:prompt_tokens] = tokens
    return batched_tokens


def infer(model: AilmV1, tokenizer: tiktoken.Encoding, max_tokens: int, max_batches: int, prompt_tokens: int, batched_tokens: mx.array, k: int, temperature: float):

    model.train(False)
    model.reset_key_value_cache(True)

    for i in (range(max_tokens)):

        batch_view = batched_tokens[:, 0:(i+prompt_tokens)]
        result = model(batch_view)

        result = result[:,-1]
        chosen = sample_topk(result, temperature, k)
        mx.eval(chosen)

        batched_tokens[:, i+prompt_tokens] = chosen

    try:
        tokens_list  = batched_tokens.tolist()
        tokens_list = typing.cast(list[list[int]], tokens_list)
        decoded = tokenizer.decode_batch(tokens_list)
    except KeyError as e:
        print(e)
        decoded = ''

    model.reset_key_value_cache(False)
    model.train(True)

    return decoded

# Setting type=bool in argparse.add_argument doesn't work the way you expect. This function borrowed from trusty old-fashioned stack overflow, can be passed to the type kwarg to create a boolean and get behavior similar to what you would expect if type was int or str.
# https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse
def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

if __name__ == '__main__':
    import argparse
    supported_models = ['AilmV1']

    # Define the command line argument parser and parse any arguments provided.
    argparser = argparse.ArgumentParser(description='Run inference with the model')
    argparser.add_argument('models', type=str, nargs='+', help='Safetensors or npz file with the model weights')
    argparser.add_argument('--model-arch', type=str, help='The architecture of the model', choices=supported_models)
    argparser.add_argument('--config-file', type=str, help='The config file from which to get the architecture of the model', default=None)
    argparser.add_argument('--token-count', type=int, default=50, help='The number of tokens to generate. Does not include the prompt.')
    argparser.add_argument('--batch-count', type=int, default=4, help='The number of batches of tokens to generate in parallel.')
    argparser.add_argument('--tokenizer', type=str, default='gpt2', help='The tokenizer to use.')
    argparser.add_argument('--temperature', type=float, default=0.5, help='The temperature for sampling.')
    argparser.add_argument('--top-k', type=int, default=10, help='The k cutoff value for sampling.')
    argparser.add_argument('--sample-strategy', type=str, default='topk', choices=SAMPLE_STRATEGIES.keys(), help='Sample strategy.')
    argparser.add_argument('--enable-kv-cache', type=str2bool, default=True, help='Enable the Key Value cache.')
    argparser.add_argument('--prompt', type=str, default='To be or not to be,', help='The prompt with which to start generation.')
    argparser.add_argument('--enable-attention-sinks', type=bool, default=True, help='Forcibly enable or disable attention sinks.')
    argparser.add_argument('--rng-seed', type=int, help='A seed for the random number generators.')
    args = argparser.parse_args()
    print(f'Inference prompt is: "{args.prompt}"')

    if args.rng_seed is not None:
        random.seed(args.rng_seed)
        mx.random.seed(args.rng_seed)

    for model_name in args.models:
        print('model_name', model_name)
        try:
            # Initialize the model and tokenizer.
            if args.config_file:
                with open(args.config_file) as config_file:
                    config_file = yaml.safe_load(config_file)
                config = AilmV1Config(**config_file['model_config'])
            else:
                config = AilmV1Config(no_rms_norm_weight=False)
            model = AilmV1(config)
            model.load_weights(model_name )

            tokenizer = tiktoken.get_encoding(args.tokenizer)

            inference_manager = InferenceManager(model, tokenizer, args.token_count, args.batch_count, args.prompt, args.top_k, args.temperature, sample_strategy=args.sample_strategy, enable_kv_cache=args.enable_kv_cache)
            try:
                inference_manager.infer()
            finally:
                inference_manager.finish()
        except Exception as e:
            print(e)
            continue
