'''
FineWeb data loader - tokenizes on the fly from parquet files.
Matches BinDataLoader API for drop-in replacement.
'''
import os
import typing
import tiktoken
import mlx.core as mx
from datasets import load_dataset


class StreamingFineWebDataLoader:
    def __init__(self, B: int, T: int, file_path: str, tokenizer: tiktoken.Encoding, subset='train', start_column=None, feature='text', shuffle=False, allowed_special_tokens: typing.Optional[set[str]] = None):
        self.B = B
        self.T = T
        self.start_column = start_column
        self.current_column = start_column or 0
        self.feature = feature

        # Load parquet files
        if isinstance(file_path, list):
            parquets = file_path
        else:
            parquets = [os.path.join(file_path, fname)
                       for fname in sorted(os.listdir(file_path))
                       if fname.endswith('.parquet')]

        print(f"Loading {len(parquets)} parquet files (streaming)...")
        self.dataset = load_dataset('parquet', data_files=parquets, streaming=True)['train']
        # self.iterator = None
        self._restart_iterator()

        # Skip to start_column
        if self.current_column > 0:
            print(f"Skipping to column {self.current_column}...")
            for _ in range(self.current_column):
                next(self.iterator)

        # Tokenizer
        self.encoder = tokenizer

        self.eot = self.encoder._special_tokens['<|endoftext|>']

        # Current document state
        self.tokens = None
        self.tokens_pos = 0

        # Batch buffer
        self.batch = mx.zeros((B * T + 1,), dtype=mx.uint16)

        # For async encoding
        self.next = None
        self.epoch = 0

    def _restart_iterator(self):
        self.iterator = iter(self.dataset)

    def reset(self):
        self.current_column = self.start_column or 0
        self.tokens = None
        self.tokens_pos = 0
        self._restart_iterator()

    def _load_next_document(self):
        """Load and tokenize the next document."""
        try:
            row = next(self.iterator)
        except StopIteration:
            self._restart_iterator()
            self.epoch += 1
            row = next(self.iterator)
            self.current_column = 0

        text = row[self.feature]
        tokens = [self.eot] + self.encoder.encode(text, allowed_special={'<|endoftext|>'})
        self.tokens = mx.array(tokens, dtype=mx.uint16)
        self.tokens_pos = 0
        self.current_column += 1

    def column(self):
        return self.current_column

    def _fill_batch(self):
        """Fill the batch buffer by consuming tokens from documents."""
        batch_pos = 0
        batch_len = len(self.batch)

        while batch_pos < batch_len:
            # Need new document?
            if self.tokens is None or self.tokens_pos >= len(self.tokens):
                self._load_next_document()
                # assert self.tokens is not None

            # Copy tokens to batch
            tokens_remaining = len(self.tokens) - self.tokens_pos
            batch_remaining = batch_len - batch_pos
            chunk_len = min(tokens_remaining, batch_remaining)

            self.batch[batch_pos:batch_pos + chunk_len] = \
                self.tokens[self.tokens_pos:self.tokens_pos + chunk_len]

            batch_pos += chunk_len
            self.tokens_pos += chunk_len

    def encode(self):
        """Prepare next batch (matches BinDataLoader API)."""
        self._fill_batch()

        shape = (self.B, self.T)
        prefixes = self.batch[:-1].reshape(shape)
        targets = self.batch[1:].reshape(shape)

        self.next = (prefixes, targets)

    def next_batch(self):
        """Return prepared batch (matches BinDataLoader API)."""
        result = self.next
        assert result is not None, "Call encode() before next_batch()"
        self.next = None
        return result
