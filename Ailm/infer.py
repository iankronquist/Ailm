import queue
import threading
import time
import typing
import mlx.core as mx
import tiktoken
from model import AilmV1, AilmV1Config

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

def decode_and_print_worker(tokenizer: tiktoken.Encoding, work_queue: queue.Queue):
    print("Starting decoder thread")
    while True:
        try:
            batched_tokens = work_queue.get()
            d0 = time.time()
            print('got batch')
            tokens_list  = batched_tokens.tolist()
            tokens_list = typing.cast(list[list[int]], tokens_list)
            decoded = tokenizer.decode_batch(tokens_list)
            d1 = time.time()
            print('\n----\n'.join(decoded), '\ndecoding duration', d1-d0)
        except KeyError as e:
            print(e)
            decoded = ''
        except queue.ShutDown:
            print('shutting down decoder thread')
            break

class InferenceManager:
    '''In the spirit of removing as much work as possible from the training thread, this inference manager only encodes the prompt once, and performs decoding and printing on a separate thread.'''
    def __init__(self, model: AilmV1, tokenizer: tiktoken.Encoding, max_tokens: int, max_batches: int, prompt: str, k: int, temperature: float):
        self.model = model
        self.tokenizer = tokenizer
        self.max_tokens = max_tokens
        self.max_batches = max_batches
        self.k = k
        self.prompt = prompt
        self.temperature = temperature

        tokenized_prompt = mx.array(tokenizer.encode(self.prompt))

        self.prompt_tokens = len(tokenized_prompt)
        max_length = max_tokens + self.prompt_tokens
        self.batched_tokens = mx.zeros((max_batches, max_length), dtype=mx.uint16)
        self.batched_tokenized_prompt = mx.zeros((max_batches, self.prompt_tokens), dtype=mx.uint16)
        self.batched_tokens[:, 0:self.prompt_tokens] = tokenized_prompt
        self.batched_tokenized_prompt[:, 0:self.prompt_tokens] = tokenized_prompt
        # def do_infer(batched_prompt: mx.array, i: int):
        #     #print('do infer', i, batched_prompt[0, 0:(i+prompt_tokens)])
        #     batch_view = batched_prompt[:, 0:(i+self.prompt_tokens)]
        #     logits = model(batch_view)
        #     logits = logits[:,-1]
        #     chosen = sample_topk(logits, self.temperature, self.k)
        #     #batched_prompt[:, i+prompt_tokens] = chosen
        #     #return batched_prompt
        #     return mx.concatenate([batch_view, chosen[:, None]], axis=-1)
        
        def do_infer(batch: mx.array):

            logits = model(batch)
            logits = logits[:, -1]
            chosen = sample_topk(logits, temperature, k)
            new_batch = mx.concatenate(
                [batch, chosen[:, None]],
                axis=1
            )
            return new_batch

        #self.do_infer = mx.compile(do_infer, inputs=[self.model.state, self.batched_tokens])
        #self.do_infer = mx.compile(do_infer, inputs=[self.model.state])
        self.do_infer = do_infer

        self.decoder_queue = queue.Queue(1)
        self.worker = threading.Thread(name='inference decoder', target=decode_and_print_worker, args=(tokenizer, self.decoder_queue,))
        self.worker.start()

    def finish(self):
        '''Shutdown the work queue if we are ever deleted. This will cause the worker thread to finish.'''
        self.decoder_queue.shutdown()
        self.worker.join()
    def infer(self):
        i0 = time.time()
        self.model.train(False)
        self.model.reset_key_value_cache(True)
        batch = self.batched_tokenized_prompt
        i05 = time.time()
        for _ in range(self.max_tokens):
            batch = self.do_infer(batch)
        i1 = time.time()
        self.model.reset_key_value_cache(False)
        self.model.train(True)
        self.decoder_queue.put(batch)
        i2 = time.time()
        print('>>>>>>>>>inference duration', i1-i05, i2-i0)


# MARK: Inference   

def infer(model: AilmV1, tokenizer: tiktoken.Encoding, max_tokens: int, max_batches: int, prompt: str, k: int, temperature: float):

    model.train(False)
    model.reset_key_value_cache(True)

    # Tokenize the prompt and batch it up.
    tokens = tokenizer.encode(prompt)
    tokens = mx.array(tokens)
    prompt_tokens = len(tokens)
    max_length = max_tokens + prompt_tokens
    batched_tokens = mx.zeros((max_batches, max_length), dtype=mx.uint16)
    batched_tokens[:, 0:prompt_tokens] = tokens

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

if __name__ == '__main__':
    import argparse
    supported_models = ['AilmV1']

    # Define the command line argument parser and parse any arguments provided.
    argparser = argparse.ArgumentParser(description='Run inference with the model')
    argparser.add_argument('model', type=str, help='Safetensors or npz file with the model weights')
    argparser.add_argument('--model-arch', type=str, help='The architecture of the model', choices=supported_models)
    argparser.add_argument('--token-count', type=int, default=50, help='The number of tokens to generate. Does not include the prompt.')
    argparser.add_argument('--batch-count', type=int, default=4, help='The number of batches of tokens to generate in parallel.')
    argparser.add_argument('--tokenizer', type=str, default='gpt2', help='The tokenizer to use.')
    argparser.add_argument('--temperature', type=float, default=0.5, help='The temperature for sampling.')
    argparser.add_argument('--top-k', type=int, default=4, help='The k cutoff value for sampling.')
    argparser.add_argument('--prompt', type=str, default='To be or not to be,', help='The prompt with which to start generation.')
    args = argparser.parse_args()

    # Initialize the model and tokenizer.
    config = AilmV1Config(no_rms_norm_weight=False)
    model = AilmV1(config)
    model.load_weights(args.model)

    tokenizer = tiktoken.get_encoding(args.tokenizer)

    # Perform inference and display the result.
    # texts = infer(model, tokenizer, args.token_count, args.batch_count, args.prompt, args.top_k, args.temperature)
    # text = '\n'.join(texts)
    # print(text)

    inference_manager = InferenceManager(model, tokenizer, args.token_count, args.batch_count, args.prompt, args.top_k, args.temperature)
    try:
        inference_manager.infer()
    finally:
        inference_manager.finish()
