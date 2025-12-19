'''
We don't have enough disk to unpack tokenized copies of the fineweb dataset, so tokenize as we go.

'''
import os
import time

import tiktoken

import random
import typing
import mlx.core
from mlx.core import array as Tensor

import datasets

from abc import ABC, abstractmethod

class AbstractDataLoader(ABC): # Inherit from ABC
    @abstractmethod
    def encode(self) -> None:
        pass
    @abstractmethod
    def next_batch(self) -> tuple[Tensor, Tensor]:
        pass
    @abstractmethod
    def reset(self) -> None:
        pass
    @abstractmethod
    def column(self) -> int:
        pass

class FineWebDataLoader(AbstractDataLoader):
    def __init__(self, B: int, T: int, file_path: str, tokenizer: tiktoken.Encoding, subset='train', start_column=None, feature='text', shuffle=False):
        self.durations = []
        self.shuffle = shuffle
        self.B = B
        self.T = T
        if isinstance(file_path, list):
            parquets = file_path
        else:
            parquets = [os.path.join(file_path, fname) for fname in os.listdir(file_path) if fname.endswith('parquet')]

        dataset = datasets.load_dataset('parquet', data_files=parquets)
        dataset = typing.cast(datasets.DatasetDict, dataset)
        self.dataset: typing.List = dataset[subset][feature]
        self.tokens = None
        self.tokens_pos = 0

        if self.shuffle:
            self.shuffled_indices = (list(range(len(self.dataset))))
            random.shuffle(self.shuffled_indices)

        self.encoder = tokenizer

        self.eot = self.encoder._special_tokens['<|endoftext|>']

        self.epoch = 0

        self.current_column = 0
        self.start_column = start_column
        if start_column is not None:
            self.current_column = start_column
        self.current_position = 0

        self.dtype = mlx.core.uint16
        self.batch = mlx.core.zeros((B * T + 1), dtype=self.dtype)

        #self.progress_bar = tqdm(total=len(self.dataset), unit='cols', desc='cols ')
        self.progress_bar = None

        self.encoded = None
        self.next_encoded = None

    #def __del__(self):
    #    self.progress_bar.close()
    #def batches_per_epoch(self):
    #    total_tokens = 10e9
    #    return total_tokens / (self.B * self.T)

    def column(self) -> int:
        return self.current_column

    def percent(self):
        return self.current_column / len(self.dataset) * 100

    def get_column(self, index):
        if self.shuffle:
            index = self.shuffled_indices[index]
        return self.dataset[index]
    def encode(self):
        if self.next_encoded is None:
            t0 = time.time()

            tokens_list = [self.eot] + self.encoder.encode(self.get_column(self.current_column), allowed_special={'<|endoftext|>',})

            self.next_encoded = mlx.core.array(tokens_list, dtype=self.dtype)
            t1 = time.time()
            self.durations.append(t1 - t0)

    def get_encoded(self):
        if self.next_encoded is None:
            self.encode()
        assert self.next_encoded is not None
        self.encoded = self.next_encoded
        self.next_encoded = None
        return self.encoded

    def reset(self):
        self.current_position = 0
        if self.start_column is not None:
            self.current_column = self.start_column
        else:
            self.current_column = 0

    def view(self, tensor, B, T):
        return tensor.reshape((B, T))

    def next_column(self):
        #print('column', self.current_column)
        if self.current_column >= len(self.dataset):
            self.current_column = 0
            self.epoch += 1
            if self.progress_bar:
                self.progress_bar.reset()

        if self.progress_bar:
            self.progress_bar.update(1)

        #t0 = time.time()
        #if self.backend == SimpleDataLoaderBackend.Torch:
        #    self.tokens = torch.tensor([self.eot] + self.encoder.encode(self.dataset[self.current_column], allowed_special={'<|endoftext|>',}), dtype=self.dtype)
        #else:
        #    self.tokens = mlx.core.array([self.eot] + self.encoder.encode(self.dataset[self.current_column], allowed_special={'<|endoftext|>',}), dtype=self.dtype)
        #t1 = time.time()
        #self.durations.append(t1 - t0)
        self.tokens = self.get_encoded()
        self.tokens_pos = 0
        self.current_column += 1
        return self.tokens
    
    def peek_batch(self):
        B, T = self.B, self.T
        x = self.view(self.batch[:-1], B, T)
        y = self.view(self.batch[1:], B, T)
        return x, y
    def next_batch(self):
        batch_pos = 0
        while batch_pos < len(self.batch):
            if self.tokens_pos == 0:
                #self.batch[batch_pos] = self.eot
                #batch_pos += 1
                if batch_pos == len(self.batch):
                    break
                self.next_column()

            batch_remaining = len(self.batch) - batch_pos
            assert self.tokens is not None
            tokens_remaining = len(self.tokens) - self.tokens_pos
            chunk_len = min(batch_remaining, tokens_remaining)
            self.batch[batch_pos:batch_pos+chunk_len] = self.tokens[self.tokens_pos:self.tokens_pos+chunk_len]
            batch_pos += chunk_len
            self.tokens_pos += chunk_len
            if self.tokens_pos >= len(self.tokens):
                self.tokens_pos = 0
                self.tokens = None
        return self.peek_batch()


