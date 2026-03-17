# model: gpt-5.2 • thinking: disabled • date: 2025-12-26
import multiprocessing as mp
import tiktoken
import q_and_a_data_loader
import fineweb_data_loader

# ---- Per-process cached loader ----

_WORKER_CACHE = None

allowed_special_tokens =   extra_tokens = {
        "<user>"  ,
        "</user>" ,
        "<agent>" ,
        "</agent>",
        "<think>" ,
        "</think>",
        '<|endoftext|>',
    }

def _get_worker_loader(loader_ctor, ctor_args, ctor_kwargs):
    global _WORKER_CACHE
    if _WORKER_CACHE is None:
        _WORKER_CACHE = loader_ctor(*ctor_args, **ctor_kwargs)
    return _WORKER_CACHE


def count_dataset_worker(args) -> int:
    loader_ctor, ctor_args, ctor_kwargs, index_range = args

    loader = _get_worker_loader(loader_ctor, ctor_args, ctor_kwargs)

    count = 0
    for i in index_range:
        #text = loader.get_column_text(i)
        column = loader.dataset[loader.subset][i]
        # if column['language'] != 'en':
        #     continue
        #encoded = loader.encoder.encode(text, allowed_special=allowed_special_tokens)
        count += 1  # EOT
    return count


def parallel_count_from_ctor(
    loader_ctor,
    ctor_args: tuple,
    ctor_kwargs: dict,
    length: int,
    num_workers: int | None = None,
    chunk_size: int = 10_000,
) -> int:

    ranges = [
        range(start, min(start + chunk_size, length))
        for start in range(0, length, chunk_size)
    ]

    tasks = [
        (loader_ctor, ctor_args, ctor_kwargs, r)
        for r in ranges
    ]

    total = 0
    with mp.Pool(processes=num_workers or mp.cpu_count()) as pool:
        for partial_total in pool.imap_unordered(count_dataset_worker, tasks, chunksize=1):
            total += partial_total

    return total


# ---- Main ----

if __name__ == '__main__':

    gpt2_encoder = tiktoken.get_encoding('gpt2')
    n_vocab = gpt2_encoder.n_vocab

    extra_tokens = {
        "<user>":   n_vocab + 1,
        "</user>":  n_vocab + 2,
        "<agent>":  n_vocab + 3,
        "</agent>": n_vocab + 4,
        "<think>":  n_vocab + 5,
        "</think>": n_vocab + 6,
    }

    allowed_special_tokens = extra_tokens.keys() | gpt2_encoder._special_tokens.keys()
    tokenizer = tiktoken.Encoding(
        name="gpt2_custom",
        pat_str=gpt2_encoder._pat_str,
        mergeable_ranks=gpt2_encoder._mergeable_ranks,
        special_tokens={**gpt2_encoder._special_tokens, **extra_tokens},
    )

    B = 4
    T = 1024
    base_path = './datasets/pleias_synth/'

    # # --- Seed loader metadata ---
    # seed_ctor = q_and_a_data_loader.QaReasoningSeedTextDataLoader
    # seed_args = (B, T, base_path, tokenizer)
    # seed_kwargs = dict(allowed_special_tokens=allowed_special_tokens)
    # seed_len = len(seed_ctor(*seed_args, **seed_kwargs).dataset['train'])

    # # --- Reasoning loader metadata ---
    # reasoning_ctor = q_and_a_data_loader.QaReasoningDataLoader
    # reasoning_args = (B, T, base_path, tokenizer)
    # reasoning_kwargs = dict(allowed_special_tokens=allowed_special_tokens)
    # reasoning_len = len(reasoning_ctor(*reasoning_args, **reasoning_kwargs).dataset['train'])

    # # print('Counting (parallel, cached loaders)...')

    # seed_count = parallel_count_from_ctor(
    #     seed_ctor, seed_args, seed_kwargs, seed_len
    # )
    # print('seed count', seed_count)

    # reasoning_count = parallel_count_from_ctor(
    #     reasoning_ctor, reasoning_args, reasoning_kwargs, reasoning_len
    # )
    # print('reasoning count', reasoning_count)

    # total = seed_count + reasoning_count
    # print('Total', total)

    combo_ctor = q_and_a_data_loader.QaReasoningComboTextDataLoader
    combo_args = (B, T, base_path, tokenizer)
    combo_kwargs = dict(allowed_special_tokens=allowed_special_tokens)
    combo_len = len(combo_ctor(*combo_args, **combo_kwargs).dataset['train'])
    print('combo len', combo_len)
    exit()
    combo_count = parallel_count_from_ctor(
        combo_ctor, combo_args, combo_kwargs, combo_len
    )
    print('combo col count', combo_count)