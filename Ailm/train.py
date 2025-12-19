#!/usr/bin/env python3

'''
An LLM training script.
'''
import os
import signal
import sys
import math
import time
import json
import yaml
import queue
import random
import typing
import threading
import dataclasses
from pprint import pprint

import mlx.nn
import mlx.nn.losses
import mlx.optimizers
import mlx.core as mx
from mlx.core import array as Tensor
from mlx.utils import tree_reduce, tree_map, tree_flatten

# import Foundation
import wandb
import tiktoken

from infer import InferenceManager, infer
from model import AilmV1Config, AilmV1
from fineweb_data_loader import FineWebDataLoader, AbstractDataLoader

# MARK: Utilities

TERMINAL_COLOR_RED='\033[0;31m'
TERMINAL_COLOR_RESET='\033[0m'

def handle_pdb(sig, frame):
    import pdb
    pdb.Pdb().set_trace(frame)    

# def name_current_thread(name: str):
#     '''Give the current NSThread a nice name which shows up in the XCode Instruments profiler'''
#     Foundation.NSThread.currentThread().setName_(name) # pyright: ignore reportAttributeAccessIssue

def count_params(model: mlx.nn.Module) -> int:
    '''Count the number of parameters in an MLX module. Does not double count modules which share weights.'''
    # Because we tie lm_head to wte, we need to track which weights we've seen before to
    # avoid double counting.
    seen_weights = set()
    def accum_if_not_seen(accumulator: int, array: Tensor):
        nonlocal seen_weights
        if array not in seen_weights:
            seen_weights.add(array)
            return accumulator + array.size
        else:
            return accumulator
    params_count = tree_reduce(accum_if_not_seen, model, 0)
    params_count = typing.cast(int, params_count)
    return params_count

def format_count(count: int):
    '''Given a count (eg of parameters), return a string in a more easily readable format.'''
    if count >= 1e9:
        suffix = 'B'
        rounded = count / 1e9
    elif count >= 1e6:
        suffix = 'M'
        rounded = count / 1e6
    elif count >= 1e3:
        suffix = 'K'
        rounded = count / 1e3
    else:
        return f'{count:,.2f}'
    return f'{rounded:.2f}{suffix} ({count})'

def format_time(seconds: float) -> str:
    """Format seconds into a human-readable duration."""
    seconds = int(seconds)

    seconds_per_minute = 60
    seconds_per_hour = 60 * seconds_per_minute
    seconds_per_day = 24 * seconds_per_hour

    if seconds < seconds_per_minute:
        return f"{seconds}s"

    if seconds < seconds_per_hour:
        minutes, secs = divmod(seconds, seconds_per_minute)
        return f"{minutes}m {secs}s"

    if seconds < seconds_per_day:
        hours, remainder = divmod(seconds, seconds_per_hour)
        minutes = remainder // seconds_per_minute
        return f"{hours}h {minutes}m"

    days, remainder = divmod(seconds, seconds_per_day)
    hours, remainder = divmod(remainder, seconds_per_hour)
    minutes = remainder // seconds_per_minute
    return f"{days}d {hours}h {minutes}m"

# MARK: Config Classes

@dataclasses.dataclass
class LearningRateScheduleConfig:
    '''Configuration for the learning rate schedule.

    - kind: May be one of fixed, trapezoid, or cosine. Other learning rate schedules such as trapezoidal and warmup->hold->cosine may be supported in the future.
    - learning_rate_max: The highest value of the learning rate.
    - learning_rate_min: Required for cosine. The lowest value of the learning rate. Often 10% of the highest value.
    - warmup_percent: Required for cosine. The percent of steps of the linear warmup ramp.
    - rampdown_percent: Required for cosine. The percent of steps of the linear rampdown.
    '''
    kind: str
    learning_rate_max: float
    learning_rate_min: typing.Optional[float]
    warmup_percent: typing.Optional[float]
    rampdown_percent: typing.Optional[float] = None

@dataclasses.dataclass
class OptimizerConfig:
    '''
    Configuration for the optimizer.

    - name: May be adamw or muon
    - weight_decay: This amount will be subtracted from the weights each update. This acts as a regularization and may help prune certain neurons as well.
    - gradient_norm_clipping: The norms of the gradients will be clipped to this value every update. This helps keep the model from making wild updates the wrong direction.
    - muon_momentum: The momentum term for muon. See this blog post from Keller Jordan: https://kellerjordan.github.io/posts/muon/
    - adamw_betas: beta1 and beta2 in the adamw equation. See the AdamW paper: https://arxiv.org/pdf/1711.05101 
    - adamw_epsilon: epsilon in the adamw equation.
    '''
    name: str
    weight_decay: float
    gradient_norm_clipping: float
    # TODO: These should be required for their respective optimizer kinds.
    muon_momentum: typing.Optional[float] = None
    adamw_betas: typing.Optional[list[float]] = None
    adamw_epsilon: typing.Optional[float] = None

@dataclasses.dataclass
class DataLoaderConfig:
    '''Configuration stanza for the data loader.
    
    - kind: Currently we only support fineweb.
    - directory: Directory full of parquet files.
    - start_column: Useful for resuming from an earlier run which got part of the way through the dataset.
    '''
    kind: str
    directory: str
    shuffle: bool

@dataclasses.dataclass
class BatchConfig:
    '''Controls batching and sequence lengths
    
    - batch_size: The full batch size.
      Larger batch sizes help absorb some gradient noise.
      Should be a multiple of micro_batch_size and larger than micro_batch_size, as well as a multiple of sequence_length.
      GPT2 used half a million, larger models use around a million.
    - micro_batch_size: The batch size the GPU will operate on in one go. We will use gradient accumulation.
    '''
    tokens_per_batch: int
    sequences_per_micro_batch: int
    sequence_length: int

@dataclasses.dataclass
class TokenizingConfig:
    '''
    Configuration for the tokenizer thread.

    - tokenizer_name: The name of the tokenizer we currently use tiktoken as the tokenizer library, and recommend gpt2.
      The model's vocab_size must be greater than or equal to the number of tokens the tokenizer will produce, or training will collapse into a pile of NaNs.
    - tokenizer_queue_length: The length of the queue which holds tokenized texts.
      Tokenization happens on a separate thread from the GPU dispatch thread, and prepared prefix and target buffers are placed on a shared queue.
      This queue needs to be long enough that we don't stall the GPU waiting on tokenization.
    - encoder_stall_seconds: The timeout for the tokenizer thread.
    - trainer_stall_seconds: The timeout for the trainer (GPU dispatch) thread.
    '''
    tokenizer_name: str
    tokenizer_queue_length: int
    encoder_stall_seconds: int
    trainer_stall_seconds: int

@dataclasses.dataclass
class IntervalsConfig:
    '''Intervals at which to run certain slow steps. These intervals are expressed in terms of update steps.
    
    - save_interval: How many updates pass until the optimizer, model weights, and other state are saved again.
    - validation_interval: How many updates pass until the validation is run again.
    - log_interval: How many updates pass until wandb and terminal logging are run again.
    '''
    save_interval: int
    validation_interval: int
    log_interval: int
    inference_interval: typing.Optional[int]

@dataclasses.dataclass
class InferenceConfig:
    '''Periodically run inference to get a subjective impression of how good the text actually is.

    - prompt: The prompt to provide to the model.
    - max_tokens_to_generate: The number of tokens to generate. Does not count the length of the prompt.
    - max_batches: The number of batches to generate in parallel from the same prompt.
    - temperature: The temperature at which to sample.
    - k: The k cutoff for top-k sampling.
    '''
    prompt: str
    max_tokens_to_generate: int
    max_batches: int
    k: int
    temperature: float

@dataclasses.dataclass
class ResumeConfig:
    resume_weights: str
    resume_optimizer: str
    resume_step: int
    resume_column: int

# MARK: Constructors

def percent_to_steps(percent: float, max_steps: int) -> int:
    assert percent < 100
    fraction = percent / 100
    return int(max_steps * fraction)

def create_learning_rate_scheduler(optimizer_config: LearningRateScheduleConfig, max_steps: int):
    '''
    Given a configuration specifying a learning rate schedule, return a function or scalar suitable for the MLX optimizer's learning rate.

    The optimizer config must have the following keys:
    - learning_rate_max: A float representing the maximum learning rate
    - max_steps: An integer representing the maximum number of steps in the training run.
    - kind: The kind of schedule required

    The following kinds are supported:
    - fixed: A scalar learning_rate_max
    - cosine: A linear ramp specified by the required key warmup_steps followed by cosine decay down to the required learning_rate_min.
    '''

    # TODO: Consider implementing a setup, hold, followed by cosine decay, a kind of compromise between trapezoid and cosine decay.
    learning_rate_max = mx.array(optimizer_config.learning_rate_max)

    if optimizer_config.kind == 'cosine':
        assert optimizer_config.learning_rate_min is not None
        assert optimizer_config.warmup_percent is not None
        learning_rate_min = mx.array(optimizer_config.learning_rate_min)

        warmup_steps = mx.array(percent_to_steps(optimizer_config.warmup_percent, max_steps))

        def warmup_cosine_decay_lr_schedule(step: Tensor) -> Tensor:
            step_f = step.astype(mx.float32)

            # Compute cosine decay factor
            decay_ratio = (step_f - warmup_steps) / (max_steps - warmup_steps)
            decay_ratio = mx.maximum(0.0, mx.minimum(1.0, decay_ratio))  # clamp [0, 1]
            coeff = 0.5 * (1.0 + mx.cos(math.pi * decay_ratio))
            cosine_lr = learning_rate_min + coeff * (learning_rate_max - learning_rate_min)

            # Warmup linear ramp
            warmup_lr = learning_rate_max * (step_f + 1) / warmup_steps

            # Final schedule using MX control logic
            lr = mx.where(step < warmup_steps, warmup_lr, cosine_lr)
            lr = mx.where(step > max_steps, learning_rate_min, lr)

            return lr
        return warmup_cosine_decay_lr_schedule
    elif optimizer_config.kind == 'fixed':
        return optimizer_config.learning_rate_max
    elif optimizer_config.kind == 'trapezoid':

        assert optimizer_config.learning_rate_min is not None
        assert optimizer_config.warmup_percent is not None
        assert optimizer_config.rampdown_percent is not None

        learning_rate_min = mx.array(optimizer_config.learning_rate_min)
        warmup_steps = mx.array(percent_to_steps(optimizer_config.warmup_percent, max_steps))
        rampdown_steps = mx.array(percent_to_steps(optimizer_config.rampdown_percent, max_steps))

        rampdown_begin = max_steps - rampdown_steps
        learning_rate_diff = learning_rate_max - learning_rate_min

        def trapezoid_lr_schedule(step: Tensor) -> Tensor:
            r'''This function should produce a graph like this:

            learning_rate_max |      ___________
                              |     /           \
                              |    /             \
                              |   /               \
            learning_rate_min |  /                 \____
                       near 0 | /
                              |_____|__________|___|____
                               0    ^          ^   ^  ^
                                    |          |   |  |
                           warmup end          |   |  |
                                        rampdown   |  |
                                           max steps  |
                                       beyond max steps - eg rounding up to the batch size or overtraining

            The Hugging Face Smol Training Playbook recommends this schedule because it's easy to extend training an experiment for longer:
            https://huggingface.co/spaces/HuggingFaceTB/smol-training-playbook#learning-rate
            '''
            warmup_lr   = learning_rate_max * (step + 1) / warmup_steps
            rampdown_lr = learning_rate_diff * (max_steps - step) / rampdown_steps + learning_rate_min

            lr = mx.where(step < warmup_steps, warmup_lr, learning_rate_max)
            lr = mx.where(step < rampdown_begin, lr, rampdown_lr)
            lr = mx.where(step < max_steps, lr, learning_rate_min)
            return lr
        return trapezoid_lr_schedule

    else:
        raise NotImplementedError(f'optimizer schedule kind "{optimizer_config.kind}" is not implemented')

def create_data_loader(data_loader_config: DataLoaderConfig, batch_config: BatchConfig, tokenizer: tiktoken.Encoding, resume_config: typing.Optional[ResumeConfig]) -> AbstractDataLoader:
    '''Create a data loader from a data loader config.'''

    start_column = resume_config.resume_column if resume_config else 0

    if data_loader_config.kind == 'fineweb':
        return FineWebDataLoader(batch_config.sequences_per_micro_batch, batch_config.sequence_length, data_loader_config.directory, tokenizer, start_column=start_column, shuffle=data_loader_config.shuffle)
    raise NotImplementedError(f"Unknown data loader kind {data_loader_config.kind}")

def create_model(training_config: dict, resume_config: typing.Optional[ResumeConfig]) -> tuple[AilmV1, AilmV1Config]:
    '''Create the model from the training config'''
    # For now we only support AilmV1, but I may want to change the architecture later.
    assert training_config['model_name'] == 'AilmV1'
    # Initialize the model config from the relevant section of the training config if present.
    training_config_model_section = training_config.get('model_config') or {}
    model_config = AilmV1Config(*training_config_model_section)

    # Initialize the model.
    model = AilmV1(model_config)
    if resume_config:
        model.load_weights(resume_config.resume_weights)
    return model, model_config

def create_optimizer(optimizer_config: OptimizerConfig, resume_config: typing.Optional[ResumeConfig], learning_rate_schedule: typing.Union[float, typing.Callable[[mx.array], mx.array]]) -> mlx.optimizers.Optimizer:
    '''Create the optimizer for our training run.'''
    if optimizer_config.name == 'muon':
        assert optimizer_config.muon_momentum is not None
        optimizer = mlx.optimizers.Muon(learning_rate=learning_rate_schedule, momentum=optimizer_config.muon_momentum, weight_decay=optimizer_config.weight_decay)
    elif optimizer_config.name == 'adamw':
        assert optimizer_config.adamw_betas is not None
        assert optimizer_config.adamw_epsilon is not None
        optimizer = mlx.optimizers.AdamW(learning_rate=learning_rate_schedule, betas=optimizer_config.adamw_betas, weight_decay=optimizer_config.weight_decay, eps=optimizer_config.adamw_epsilon)
    else:
        raise NotImplementedError(f"Unimplemented optimizer {optimizer_config.name}")
    if resume_config:
        resume_state = mx.load(resume_config.resume_weights)
        optimizer.state = typing.cast(dict, resume_state)
    return optimizer
        
def calculate_optimal_token_budget(parameter_count: int) -> int:
    '''See the Deepmind chinchilla paper: https://arxiv.org/pdf/2203.15556'''
    return parameter_count * 20

def loss_fn(model: mlx.nn.Module, inputs: Tensor, targets: Tensor) -> Tensor:
    """Compute cross-entropy loss."""
    logits: Tensor = model(inputs)
    B, T, V = logits.shape
    logits = logits.reshape(B * T, V)
    targets = targets.reshape(B * T)
    loss = mlx.nn.losses.cross_entropy(logits, targets, reduction='mean')
    return loss

# MARK: Validation

@dataclasses.dataclass
class ValidationResult:
    perplexity: float
    mean_loss: float
    current_column: int

def validate(model: mlx.nn.Module, val_loader: AbstractDataLoader, val_queue: queue.Queue, num_batches: int, should_reset: bool, stall_secs: int) -> ValidationResult:
    """Run validation and return average loss and perplexity"""
    total_loss = 0.0
    if should_reset:
        val_loader.reset()
    
    current_column = -1
    for _ in range(num_batches):
        val_loader.encode()
        inputs, targets, current_column = val_queue.get(block=True, timeout=stall_secs)
        loss = loss_fn(model, inputs, targets)
        mx.eval(loss)
        total_loss += loss.item()
    
    mean_loss = total_loss / num_batches
    perplexity = math.exp(mean_loss)
    return ValidationResult(perplexity=perplexity, mean_loss=mean_loss, current_column=current_column)

# MARK: Encoding

def create_encoder_worker(tokenizing_config: TokenizingConfig, data_loader: AbstractDataLoader, name: str) -> queue.Queue:
    '''Create and start the encoder worker thread. We perform tokenization encoding on a separate thread so we don't stall the GPU while waiting for tokenization to complete. The main thread will pop completed batches of tokens off of a queue, which will awaken the tokenization thread to perform its work.'''

    encoder_queue = queue.Queue(tokenizing_config.tokenizer_queue_length)
    worker = threading.Thread(name=name, target=encoding_worker, args=(tokenizing_config, encoder_queue, data_loader, name))
    worker.start()
    return encoder_queue

def encoding_worker(tokenizing_config: TokenizingConfig, encoder_queue: queue.Queue, data_loader: AbstractDataLoader, name: str):
    '''Encoder worker thread. In an infinite loop, encode, push to the queue, and wait until there is room on the queue to encode again.'''
    print('Starting tokenizer worker', name)
    # if name:
    #     name_current_thread(name)
    # FIXME This seems to be necessary to fix a strange import bug in the event of a save shutdown
    import queue
    while True:
        try:
            data_loader.encode()
            prefixes, targets = data_loader.next_batch()

            encoder_queue.put((prefixes, targets, data_loader.column()), block=True, timeout=None)
        except queue.ShutDown:
            break

# MARK: Saving

def create_save_worker(model: mlx.nn.Module, optimizer: mlx.optimizers.Optimizer, run_dir_path: str, timeout: typing.Optional[int], name: str) -> tuple[queue.Queue, threading.Lock]:
    '''Create and start the save worker thread.
    We perform weight saving on a separate thread so we don't stall the GPU while waiting for disk writes to complete.
    The training thread will send a message to the queue telling the worker to save the weights.
    The worker will acquire the update lock, save the model weights, optimizer, and perhaps other state, and then release the lock.
    The training thread acquire the update lock whenever it updates the weights.
    This allows forward passes through the model to happen in parallel to writing out the model's current state.
    '''

    save_queue = queue.Queue(1)
    lock = threading.Lock()
    worker = threading.Thread(name=name, target=save_worker, args=(model, optimizer, run_dir_path, save_queue, lock, name))
    worker.start()
    return save_queue, lock

def save_worker(model: mlx.nn.Module, optimizer: mlx.optimizers.Optimizer, run_dir_path: str, save_queue: queue.Queue, save_lock: threading.Lock, name: str):
    print('Starting tokenizer worker', name)
    # if name:
    #     name_current_thread(name)
    model_checkpoint_name = os.path.join(run_dir_path, 'model_checkpoint.npz')
    optimizer_checkpoint_name = os.path.join(run_dir_path, 'opt_checkpoint.safetensors')
    # FIXME This seems to be necessary to fix a strange import bug in the event of a save shutdown
    import queue
    while True:
        try:
            _ = save_queue.get(block=True, timeout=None)
            save_lock.acquire()
            model.save_weights(model_checkpoint_name)
            optimizer_state = tree_flatten(optimizer.state, destination={})
            optimizer_state = typing.cast(dict[str, Tensor], optimizer_state)
            mx.save_safetensors(optimizer_checkpoint_name, optimizer_state)
            save_lock.release()
        except queue.ShutDown:
            break

# MARK: Training

def train_model(training_config: dict[str, typing.Any], save_directory: str, no_save: bool):
    '''Train a model.
    Arguments:
    - training_config: A dictionary of configuration keys which are used to set hyperparameters and control model training
    - save_directory: The directory in which to save checkpoints, model weights, optimizer weights, and other detritus
    - no_save: Disables saving the model weights and wandb logging. Useful for kicking off a quick test run. 
    '''
    training_start = time.strftime("%Y%m%d-%H%M")

    # HACK: It seems kind of janky but if I send SIGUSR1 to the training process I should be able to break into it with PDB and edit things on the fly.
    signal.signal(signal.SIGUSR1, handle_pdb)

    # Initialize config objects.
    learning_rate_schedule_config = LearningRateScheduleConfig(**config['learning_rate_schedule'])
    optimizer_config = OptimizerConfig(**training_config['optimizer'])
    batch_config = BatchConfig(**training_config['batching'])
    training_loader_config = DataLoaderConfig(**training_config['training_data_loader'])
    validation_loader_config = DataLoaderConfig(**training_config['validation_data_loader'])
    tokenizing_config = TokenizingConfig(**training_config['tokenizing'])
    intervals_config = IntervalsConfig(**training_config['intervals'])
    inference_config = InferenceConfig(**training_config['inference'])
    if training_config.get('resume'):
        resume_config = ResumeConfig(**training_config['resume'])
    else:
        resume_config = None

    # Basic sanity check that our validation set is different than our training set.
    assert training_loader_config.directory != validation_loader_config.directory

    # Sanity check our batch size
    assert (batch_config.tokens_per_batch % (batch_config.sequence_length * batch_config.sequences_per_micro_batch)) == 0
    assert batch_config.sequences_per_micro_batch <= batch_config.tokens_per_batch

    # Seed RNGs before creating the model.
    rng_seed = training_config['rng_seed']
    random.seed(rng_seed)
    mx.random.seed(rng_seed)

    # TODO: Optionally reload module weights if the training config specifies them.
    model, model_config = create_model(training_config, resume_config)

    parameter_count = count_params(model)

    # Calculate the token budget if necessary.
    if training_config.get('token_budget') is None:
        training_config['token_budget'] = calculate_optimal_token_budget(parameter_count)
    token_budget = training_config['token_budget']
 
    update_count = training_config['update_count'] = int(token_budget / batch_config.tokens_per_batch)

    # Calculate our gradient accumulation steps.
    if training_config.get('gradient_accumulation_steps') is None:
        training_config['gradient_accumulation_steps'] = int(math.ceil(batch_config.tokens_per_batch / batch_config.sequences_per_micro_batch / batch_config.sequence_length))

    gradient_accumulation_steps = training_config['gradient_accumulation_steps']
        
    # Calculate the number of steps we will take.
    # This is the number of times we will call the model, not the number of updates we will perform.
    # Each step we process `batch_config.sequences_per_micro_batch * batch_config.sequence_length` tokens.
    if training_config.get('step_count') is None:
        training_config['step_count'] = update_count * gradient_accumulation_steps
    step_count = training_config['step_count']

    update_count = training_config['update_count'] = int(step_count / gradient_accumulation_steps)

    tokens_to_process = training_config['tokens_to_process'] = step_count * batch_config.sequence_length * batch_config.sequences_per_micro_batch

    validation_batches = training_config['validation_batches']

    # Get the learning rate schedule for our training run. It is either a function which can by compiled by MLX or a scalar.
    learning_rate_schedule = create_learning_rate_scheduler(learning_rate_schedule_config, update_count)

    # Create the optimizer for our training run.
    optimizer = create_optimizer(optimizer_config, resume_config, learning_rate_schedule)

    # Initialize the tokenizer and data loaders. We require both training and validation data loaders.
    tokenizer = tiktoken.get_encoding(tokenizing_config.tokenizer_name)
    training_data_loader = create_data_loader(training_loader_config, batch_config, tokenizer, resume_config)
    # Do not resume from where we were in the validation loader.
    validation_loader = create_data_loader(validation_loader_config, batch_config, tokenizer, None)

    # Sanity check that our lm_head is at least as big as our tokenizer's vocabulary.
    assert tokenizer.n_vocab <= model_config.vocab_size
    
    # Create the worker threads and work queues to communicate with them.
    training_batch_queue = create_encoder_worker(tokenizing_config, training_data_loader, "Training encoder")
    validation_batch_queue = create_encoder_worker(tokenizing_config, validation_loader, "Validation encoder")

    # Create the directory to hold all of our save files.
    run_name = f'run_{training_start}'
    run_dir_path = os.path.join(save_directory, run_name)
    final_model_checkpoint_name = os.path.join(run_dir_path, 'final_model.npz')
    final_optimizer_checkpoint_name = os.path.join(run_dir_path, 'final_opt.safetensors')
    receipt_name = os.path.join(run_dir_path, 'receipt.json')
    if not no_save:
        os.mkdir(run_dir_path)
 
    #save_queue, save_lock = create_save_worker(model, optimizer, run_dir_path, None, , "Save weights")

    inference_manager = InferenceManager(model, tokenizer, inference_config.max_tokens_to_generate, inference_config.max_batches, inference_config.prompt, inference_config.k, inference_config.temperature)

    # Log to wandb if we're saving the model
    if no_save:
        wandb_run = None
    else:
        wandb_run = wandb.init(
            entity="muricula-mus-inc",
            project='Ailm',
            save_code=True,
            config=training_config
        )
    run_command = ' '.join(sys.argv)
    # It is in some sense a miracle that LLM training works at all.
    print('᚛ᚁᚓᚅᚇᚇᚐᚉᚈᚐᚅᚔᚋᚂ᚜')
    print('Ailm LLM Trainer')
    pprint(training_config)

    print(f'Training process pid is {os.getpid()}')
    print('Training started at', training_start)
    print("The model has", format_count(parameter_count), "parameters")
    print('tokens to process', format_count(tokens_to_process))
    print('step count', step_count)
    print('gradient_accumulation_steps', gradient_accumulation_steps)
    print('update count', update_count)
    print('Saving to', final_model_checkpoint_name)

    # Don't listen to the IDE's lies, this is indeed a public API:
    # https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.nn.value_and_grad.html
    # This wraps the model and returns a function which calculates the loss and the corresponding grads for one pass through the model.
    loss_and_grad_fn = mlx.nn.value_and_grad(model, loss_fn) # pyright: ignore [reportPrivateImportUsage]

    # Compile a single forward pass and gradient accumulation step, closing over the model's state.
    step_state = [model.state]
    def do_step(prefixes: Tensor, targets: Tensor, accum_grads: typing.Optional[Tensor]) -> tuple[Tensor, Tensor]:
        '''Perform one step of the model tracking loss and accumulating gradients.'''
        loss, grads = loss_and_grad_fn(model, prefixes, targets)

        # Perform gradient accumulation.
        if accum_grads is None:
            # If there are no existing grads, upcast to float32 to keep the gradients in higher precision until we average them.
            new_grads = tree_map(lambda array: array.astype(mx.float32), grads)
        else:
            # Otherwise add the new grads to the accumulated grads, implicitly up-casting to float32.
            new_grads = tree_map(mx.add, accum_grads, grads)
            # Delete the old grads so we don't have to wait for the GC to reuse this memory.
            del accum_grads

        return (loss, new_grads)
    do_step = mx.compile(do_step, inputs=step_state, outputs=step_state)
    
    # I haven't benchmarked it rigorously, but I suspect it may be faster to use multiplication than division.
    gradient_averaging_scale = mx.array(1.0 / gradient_accumulation_steps)

    update_state = [model.state, optimizer.state]
    def do_update(accum_grads: Tensor) -> Tensor:
        '''Perform an update. Scale and clip the gradients, step the optimizer, and update the weights.'''
        averaged_grads = tree_map(lambda g: (g * gradient_averaging_scale).astype(model_config.dtype), accum_grads)

        clipped_gradients, norm = mlx.optimizers.clip_grad_norm(averaged_grads, optimizer_config.gradient_norm_clipping)

        optimizer.update(model, clipped_gradients)
        return norm
    do_update = mx.compile(do_update, inputs=update_state, outputs=update_state)
    

    if gradient_accumulation_steps <= intervals_config.save_interval * 10:
        print(f"{TERMINAL_COLOR_RED}Warning: gradient accumulation steps {gradient_accumulation_steps} is close to the save interval {intervals_config.save_interval}. We may wind up stalling the GPU waiting for tokenization.{TERMINAL_COLOR_RESET}")

    # MacOS will hang if the compute graph grows beyond the size of memory, so in order to crash instead of hanging we set a memory limit from the training config.
    memory_limit = training_config['memory_limit']

    # Base case for the accumulated gradients.
    accum_grads = None

    # Running statistics for logging
    total_tokens = 0
    running_tokens = 0
    running_loss = 0.0
    running_grad_norm = 0.0
    updates_since_last_log = 0
    last_update_step = 0

    # Mollify the linter in case we never make it through the loop successfully.
    step = -1
    training_column = -1

    start_step = resume_config.resume_step if resume_config else 0

    # MARK: Training Loop
    try:
        # The training loop. Handles updates and forward passes in one big loop.
        print("Training loop start")
        # Start the stopwatch used to calculate tokens per second.
        training_start = stopwatch_start = time.time()
        for step in range(start_step, step_count):

            prefixes, targets, training_column = training_batch_queue.get(block=True, timeout=tokenizing_config.trainer_stall_seconds)

            mean_loss, accum_grads = do_step(prefixes, targets, accum_grads)

            # We need to evaludate our gradients every step to prevent the compute graph from exploding
            mx.eval(mean_loss)
            mx.eval(accum_grads)

            # Accumulate per-step running statistics.
            running_loss += mean_loss.item()
            running_tokens += prefixes.size

            # Check for OOM. MacOS will hang if we schedule a compute graph which is too large to fit into memory.
            if mx.get_active_memory() > memory_limit:
                print(f"\nMemory limit exceeded at step {step}!")
                print(f"Active memory: {mx.get_active_memory() / 1024**3:.2f}GB")
                print(f"Memory limit: {memory_limit / 1024**3:.2f}GB")
                raise MemoryError("OOM")

            if ((step+1) % gradient_accumulation_steps) == 0:
                total_tokens += batch_config.sequences_per_micro_batch * batch_config.sequence_length * gradient_accumulation_steps
                update_start = time.time()
                # The update must be performed while holding the save lock so that the save worker doesn't save a partially updated set of weights.
                #save_lock.acquire()
                norm = do_update(accum_grads)
                #save_lock.release()
                # Reset accumulated gradients for the next gradient_accumulation_steps period.
                accum_grads = None
                running_grad_norm += norm.item()
                update_end = time.time()

                updates_since_last_log += 1

                update_step = (step+1) / gradient_accumulation_steps
                if (update_step % intervals_config.save_interval) == 0 and not no_save:
                    # FIXME this currently leads to a sigfault the first time I save, so save synchronously.
                    #save_queue.put(step)
                    model.save_weights(final_model_checkpoint_name)
                    optimizer_state = tree_flatten(optimizer.state, destination={})
                    optimizer_state = typing.cast(dict[str, Tensor], optimizer_state)
                    mx.save_safetensors(final_optimizer_checkpoint_name, optimizer_state)
                    # Save a receipt file with metadata about how this run was started.

                    with open(receipt_name, 'w') as receipt_file:
                        json.dump({
                            'step': step,
                            'current_column': training_column,
                            'run_command': run_command,
                            'training_config': training_config,
                        }, receipt_file)

                # Periodically run validation
                if (update_step % intervals_config.validation_interval) == 0:

                    validation_start = time.time()
                    validation_result = validate(model, validation_loader, validation_batch_queue, validation_batches, True, tokenizing_config.trainer_stall_seconds)
                    validation_end = time.time()
                    validation_duration = validation_end - validation_start
                    if wandb_run:
                        validation_log_message = {
                            'val/loss': validation_result.mean_loss,
                            # 'step': step,
                            'val/perplexity': math.exp(validation_result.mean_loss),
                            'val/column': validation_result.current_column,
                            'val/duration': validation_duration,
                        }
                        wandb_run.log(validation_log_message)
                        print('Validation:', validation_log_message)

                if intervals_config.inference_interval is not None and (update_step % intervals_config.inference_interval) == 0:
                    # text = infer(model, tokenizer, max_tokens=inference_config.max_tokens_to_generate, max_batches=inference_config.max_batches, prompt=inference_config.prompt, k=inference_config.k, temperature=inference_config.temperature)
                    # print(text)
                    inference_manager.infer()

                # Periodically log to wandb and the terminal
                if (update_step % intervals_config.log_interval) == 0:

                    # Stop the stopwatch which we will use to compute tokens per second.
                    # Calculating tokens per second and ETA every logging interval makes it so an outlier logging interval,
                    # for example if my Mac goes to sleep while I carry it to my local cafe, won't affect long term scores permanently.
                    stopwatch_stop = time.time()
                    stopwatch_duration = stopwatch_stop - stopwatch_start
                    step_duration = stopwatch_duration / updates_since_last_log
                    tokens_processed = updates_since_last_log * batch_config.sequences_per_micro_batch * batch_config.sequence_length * gradient_accumulation_steps
                    assert running_tokens == (updates_since_last_log * batch_config.sequences_per_micro_batch * batch_config.sequence_length * gradient_accumulation_steps)
                    tokens_per_second = tokens_processed / stopwatch_duration
                    update_duration = update_end - update_start

                    # Calculate average gradient norm, loss, and perplexity for this log interval.
                    average_loss_this_update = running_loss / (gradient_accumulation_steps * updates_since_last_log)
                    average_grad_norm = running_grad_norm / updates_since_last_log
                    perplexity = math.exp(average_loss_this_update)

                    current_learning_rate = float(optimizer.learning_rate.item())

                    # Reset running calculations.
                    running_loss = 0.0
                    running_grad_norm = 0.0
                    updates_since_last_log = 0
                    running_tokens = 0

                    log_message = {
                        'train/step': step,
                        'train/loss': average_loss_this_update,
                        'train/gradient_norm': average_grad_norm,
                        'train/learning_rate': current_learning_rate,
                        'train/perplexity': perplexity,
                        'train/tokens_per_second': tokens_per_second,
                        'train/current_column': training_column,
                        'train/tokens': total_tokens,
                        'train/update_duration': update_duration,
                        'train/step_duration': step_duration,
                    }

                    if wandb_run:
                        wandb_run.log(log_message)

                    # Calculate our ETA
                    elapsed_time = stopwatch_stop - training_start
                    mean_time_per_step = elapsed_time / (step+1)
                    steps_remaining = step_count - step
                    eta_seconds = mean_time_per_step * steps_remaining

                    print(*log_message.values(), format_time(eta_seconds))
                    # The stopwatch restarts from the moment we last stopped it, that is the beginning of the logging step.
                    stopwatch_start = stopwatch_stop
                last_update_step = step
    finally:
        # Save our last known weights, and try to safely shutdown our worker threads.
        # This should happen if there is an exception, or if we successfully completed training.
        try:
            if not no_save:
                # Save final state
                model.save_weights(final_model_checkpoint_name)
                optimizer_state = tree_flatten(optimizer.state, destination={})
                optimizer_state = typing.cast(dict[str, Tensor], optimizer_state)
                mx.save_safetensors(final_optimizer_checkpoint_name, optimizer_state)
                if step_count:
                    with open(receipt_name, 'w') as receipt_file:
                        json.dump({
                            'step': step,
                            'current_column': training_column,
                            'run_command': run_command,
                            'training_config': training_config,
                        }, receipt_file)
        finally:
            validation_batch_queue.shutdown()
            training_batch_queue.shutdown()
            inference_manager.finish()
            # save_queue.shutdown()

    # Finish the wandb run successfully.
    if wandb_run:
        wandb_run.finish()


if __name__ == '__main__':
    import argparse

    # Define the command line argument parser and parse any arguments provided.
    # Most notably we require a yaml config file defining all the settings for this training run.
    argparser = argparse.ArgumentParser(description='LLM Trainer')
    argparser.add_argument('training_config', help='A yaml configuration file for this training run')
    argparser.add_argument('--save-directory', type=str, default='runs', help='Directory in which to save files')
    argparser.add_argument('--no-save', help="Don't generate any save files", default=False, action='store_true')
    args = argparser.parse_args()

    # Load the config file full of hyperparameters and other settings.
    with open(args.training_config) as config_file:
        config = yaml.safe_load(config_file)

    # Kick off the training run.
    train_model(config, args.save_directory, args.no_save)
