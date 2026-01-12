from mlx.core import array as Tensor
from fineweb_data_loader import AbstractDataLoader, FineWebDataLoader

class MultiDataLoader(AbstractDataLoader):
    '''Train with multiple datasets. This doesn't shuffle between datasts, it just trains each one in series.'''
    def __init__(self, data_loaders: list[AbstractDataLoader]):
        self.data_loaders = data_loaders
        self.current_data_loader_index = 0

    def _current_data_loader(self) -> AbstractDataLoader:
        '''Get the current data loader object. self.current_data_loader_index must be less than the length of the list of data loaders.'''
        return self.data_loaders[self.current_data_loader_index]
    def encode(self) -> None:
        return self._current_data_loader().encode()

    def next_batch(self) -> tuple[Tensor, Tensor]:
        while True:
            # FIXME: We check if we've exhausted a dataset by seeing if the column index has wrapped back around.
            #
            # This won't work if total tokens in the dataset is shorter than B * T, but that's a silly degenerate case extremely unlikely to happen in practice.
            #
            # The actual problem really is that we will incur two calls to encode when we wrap around: the first call to encode from the first column of the current dataset,
            # and the second call to encode on the first column of the next dataset. This should happen rarely enough that it's not a major performance issue, but it is wasteful.
            #
            # Note that if there is a ragged edge at the end of a dataset which doesn't fit inside a batch (ie (len(dataset_text) % (B * T)) != 0), then we will never train on those tokens.
            # This is also wasteful in some sense, but is probably preferable to training on tokens we've already seen, or introducing a padding token which the model has never seen before.

            current_column = self.column()
            next_batch = self._current_data_loader().next_batch()
            next_column = self.column()
            # If we wrapped around, switch to the next data loader and get its batch.
            if next_column < current_column:
                self.current_data_loader_index += 1
                # If we're done with the final data loader, reset all of them and start again.
                if self.current_data_loader_index > len(self.data_loaders):
                    self.reset()
                continue
            else:
                # We did not wrap around. Return the batch we got.
                return next_batch
        
    def reset(self) -> None:
        # Reset each data loader, and then begin anew from the first data loader.
        for data_loader in self.data_loaders:
            data_loader.reset()
        self.current_data_loader_index = 0

    def column(self) -> int:
        return self._current_data_loader().column()
    

class QaReasoningDataLoader(FineWebDataLoader):
    '''A data loader for training on the reasoning traces of the PleiAS SYNTH dataset.
    
    https://huggingface.co/datasets/PleIAs/SYNTH
    '''
    def filter_column(self, index: int) -> bool:
        '''Filter our non-English columns.'''
        column = self.dataset[self.subset][index]
        return column['language'] != 'en'
    def get_column_text(self, index):
        '''Get the text for the column at the index. This joins together the user query, synthetic thinking trace, and synthetic answers from the appropriate columns.
        We use a different tokenization than PleiAS does, with tags like <user>, </user>, <think>, </think>, <agent>, </agent>.
        '''
        column = self.dataset[self.subset][index]
        query = column['query']
        synthetic_reasoning = column['synthetic_reasoning']
        synthetic_answer = column['synthetic_answer']

        return f'''<user>{query}</user><think>{synthetic_reasoning}</think><agent>{synthetic_answer}</agent>'''

class QaReasoningSeedTextDataLoader(QaReasoningDataLoader):
    '''A data loader for training on the seed text of the PleiAS SYNTH dataset.
    
    https://huggingface.co/datasets/PleIAs/SYNTH
    '''
    def get_column_text(self, index):
        '''Get the text for the column at the index. Return just the seed text and end of text token.'''
        column = self.dataset[self.subset][index]
        seed_text = column['query_seed_text']
        return seed_text
        #return f'''{seed_text}''

class QaReasoningComboTextDataLoader(QaReasoningDataLoader):
    '''A data loader for training on the seed text of the PleiAS SYNTH dataset.
    
    https://huggingface.co/datasets/PleIAs/SYNTH
    '''
    def get_column_text(self, index):
        '''Get the text for the column at the index. Return just the seed text and end of text token.'''
        column = self.dataset[self.subset][index]
        seed_text = column['query_seed_text']
        query = column['query']
        synthetic_reasoning = column['synthetic_reasoning']
        synthetic_answer = column['synthetic_answer']

        text = f'''{seed_text}<|endoftext|><user>{query}</user><think>{synthetic_reasoning}</think><agent>{synthetic_answer}</agent>'''
        return text
    
    # def filter_column(self, index: int) -> bool:
    #     return False

