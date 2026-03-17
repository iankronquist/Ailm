import math
import mlx.core as  mx
import numpy as np
from tqdm import tqdm

class BinDataLoader:
    def __init__(self, B: int, T: int, file_path: str, start_column=None, fraction=None):
        self.B = B
        self.T = T
        self.start_column = start_column
        self.current_column = self.start_column or 0
        self.current_position = 0
        self.data = np.memmap(
            file_path,
            dtype=np.uint16,
            mode="r",
            offset=0,
            )

        #if fraction:
        #    end = int(len(self.data) * fraction)
        #    self.data = self.data[:end]

        self.total_batches = math.ceil(len(self.data) / self.B / self.T)
        print('total tokens', len(self.data))
        print('total batches', self.total_batches)

        #self.progress_bar = tqdm(total=self.total_batches, unit='col', desc='cols ')
        #self.progress_bar.update(self.current_column)

        self.next = None

    def percent(self):
        return self.current_column / self.total_batches * 100

    def reset(self):
        self.current_column = self.start_column or 0

    def encode(self):
        batch_size = self.B * self.T
        batch_buffer_length = batch_size + 1
        start = self.current_column * batch_size
        end = start + batch_buffer_length
        if end > len(self.data):
            start = 0
            end = batch_buffer_length
        batch = mx.array(self.data[start:end])
        shape = (self.B, self.T)
        prefixes = batch[:-1].reshape(shape)
        targets  = batch[ 1:].reshape(shape)

        self.next = prefixes, targets
        return

    def next_batch(self):
        result = self.next
        assert result is not None
        self.current_column += 1
        self.next = None
        #self.progress_bar.update(1)
        return result



class MinBinDataLoader:
    def __init__(self, B: int, T: int, file_path: str, start_column=None, fraction=None):
        self.B = B
        self.T = T
        self.start_column = start_column
        self.current_column = self.start_column or 0
        self.current_position = 0
        self.data = mx.array(np.memmap(
            file_path,
            dtype=np.uint16,
            mode="r",
            offset=0,
            ))


