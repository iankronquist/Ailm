'''Command line tool to view a particular from the dataset.
This can be useful for diagnosing loss or gradient norm spikes seen during training.
For instance, at one point a model I was training encountered high loss, and when I investigated the column it was training on during that run, I discovered that it had encountered ascii tabular data for the first time.'''
import tiktoken
import argparse
from fineweb_data_loader import FineWebDataLoader
from q_and_a_data_loader import QaReasoningComboTextDataLoader

parser = argparse.ArgumentParser()
parser.add_argument('column', type=int)
parser.add_argument('--dataset', type=str, default='fineweb_edu_sample')

args = parser.parse_args()




tokenizer_name = 'gpt2'
base_tokenizer = tiktoken.get_encoding(tokenizer_name)

extra_tokens_list = [
  "<user>"
, "</user>"
, "<think>"
, "</think>"
, "<agent>"
, "</agent>"
]

shuffle = False



extra_tokens: dict[str, int] = dict()
for (i, token_str) in enumerate(extra_tokens_list):
    extra_tokens[token_str] = base_tokenizer.n_vocab + i

tokenizer = tiktoken.Encoding(
    name="gpt2_custom",
    pat_str=base_tokenizer._pat_str,
    mergeable_ranks=base_tokenizer._mergeable_ranks,
    special_tokens={**base_tokenizer._special_tokens, **extra_tokens},
)


sequences_per_micro_batch = 4
sequence_length = 1024
start_column = 0
if 'fineweb_edu_sample' in args.dataset:
    directory = 'datasets/fineweb_edu_sample'
    data_loader = FineWebDataLoader(sequences_per_micro_batch, sequence_length, directory, tokenizer, start_column=start_column)
else:
    directory = 'datasets/pleias_synth'
    allowed_special_tokens = None

    if extra_tokens_list:
        allowed_special_tokens = set(extra_tokens_list)

    data_loader = QaReasoningComboTextDataLoader(sequences_per_micro_batch, sequence_length, directory, tokenizer, start_column=start_column, shuffle=shuffle, allowed_special_tokens=allowed_special_tokens)


column = args.column
text = data_loader.get_column_text(column)
print(text)

