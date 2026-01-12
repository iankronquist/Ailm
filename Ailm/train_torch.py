#!/usr/bin/env python3

'''
An LLM training script - PyTorch Port
Model: Claude Sonnet 4.5 (2025-01-12)
'''
import os
import sys
import math
import time
import json
import psutil
import yaml
import queue
import random
import signal
import typing
import threading
import dataclasses
from pprint import pprint

import torch
from torch import Tensor
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_

import Foundation
import wandb
import tiktoken

from model_utils import model_name_to_class_and_config
from infer import InferenceManager, infer
from model_torch import AilmV1Config, AilmV1, dtype_name_to_torch_dtype
from fineweb_data_loader import FineWebDataLoader, AbstractDataLoader
from q_and_a_data_loader import QaReasoningComboTextDataLoader

# MARK: Utilities

TERMINAL_COLOR_RED='\033[0;31m'
TERMINAL_COLOR_RESET='\033[0m'

def handle_pdb(sig, frame):
    import pdb
    pdb.Pdb().set_trace(frame) 

def running_under_pdb() -> bool:
    return 'pdb' in sys.modules

def running_caffeinated() -> bool:
    '''Check if we were started using the caffeinate utility.'''
    us = psutil.Process()
    return any([child.name() == 'caffeinate' for child in us.children()])   

def warn(*args, **kwargs):
    '''Print a warning in red to stderr colored red with ANSI terminal escape sequences.'''
    if 'file' in kwargs.keys():
        kwargs.pop('file')
    print(TERMINAL_COLOR_RED, *args, TERMINAL_COLOR_RESET, **kwargs, file=sys.stderr)

def name_current_thread(name: str):
    '''Give the current NSThread a nice name which shows up in the XCode Instruments profiler'''
    # Foundation.NSThread.currentThread().setName_(name)

def count_params(model: nn.Module) -> int:
    '''Count the number of parameters in a PyTorch module. Does not double count tied weights.'''
    seen_params = set()
    total = 0
    for param in model.parameters():
        if id(param) not in seen_params:
            seen_params.add(id(param))
            total += param.numel()
    return total

# def has_nan(x: Tensor) -> bool:
#     return torch.isnan(x).any().item()

def has_nan_tree(model: nn.Module) -> bool:
    '''Check if any parameter in the model has NaN values'''
    for param in model.parameters():
        if torch.isnan(param).any():
            return True
    return False

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
    '''Configuration for the learning rate schedule.'''
    kind: str
    learning_rate_max: float
    learning_rate_min: typing.Optional[float]
    warmup_percent: typing.Optional[float]
    rampdown_percent: typing.Optional[float] = None

@dataclasses.dataclass
class OptimizerConfig:
    '''Configuration for the optimizer.'''
    name: str
    weight_decay: float
    gradient_norm_clipping: float
    muon_momentum: typing.Optional[float] = None
    adamw_betas: typing.Optional[list[float]] = None
    adamw_epsilon: typing.Optional[float] = None

@dataclasses.dataclass
class DataLoaderConfig:
    '''Configuration stanza for the data loader.'''
    kind: str
    directory: str
    shuffle: bool

@dataclasses.dataclass
class BatchConfig:
    '''Controls batching and sequence lengths'''
    tokens_per_batch: int
    sequences_per_micro_batch: int
    sequence_length: int

@dataclasses.dataclass
class TokenizingConfig:
    '''Configuration for the tokenizer thread.'''
    tokenizer_name: str
    tokenizer_queue_length: int
    encoder_stall_seconds: int
    trainer_stall_seconds: int
    extra_tokens: typing.Optional[list[str]] = None

@dataclasses.dataclass
class IntervalsConfig:
    '''Intervals at which to run certain slow steps.'''
    save_interval: int
    validation_interval: int
    log_interval: int
    inference_interval: typing.Optional[int]

@dataclasses.dataclass
class InferenceConfig:
    '''Periodically run inference to get a subjective impression of how good the text actually is.'''
    prompt: str
    max_tokens_to_generate: int
    max_batches: int
    k: int
    temperature: float

@dataclasses.dataclass
class ResumeConfig:
    '''Used for resuming from a previous run.'''
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
    '''Given a configuration specifying a learning rate schedule, return a callable or scalar.'''
    
    learning_rate_max = optimizer_config.learning_rate_max

    if optimizer_config.kind == 'cosine':
        assert optimizer_config.learning_rate_min is not None
        assert optimizer_config.warmup_percent is not None
        learning_rate_min = optimizer_config.learning_rate_min
        warmup_steps = percent_to_steps(optimizer_config.warmup_percent, max_steps)

        def warmup_cosine_decay_lr_schedule(step: int) -> float:
            if step < warmup_steps:
                return learning_rate_max * (step + 1) / warmup_steps
            elif step > max_steps:
                return learning_rate_min
            else:
                decay_ratio = (step - warmup_steps) / (max_steps - warmup_steps)
                decay_ratio = max(0.0, min(1.0, decay_ratio))
                coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
                return learning_rate_min + coeff * (learning_rate_max - learning_rate_min)
        
        return warmup_cosine_decay_lr_schedule
    
    elif optimizer_config.kind == 'fixed':
        return lambda step: optimizer_config.learning_rate_max
    
    elif optimizer_config.kind == 'trapezoid':
        assert optimizer_config.learning_rate_min is not None
        assert optimizer_config.warmup_percent is not None
        assert optimizer_config.rampdown_percent is not None

        learning_rate_min = optimizer_config.learning_rate_min
        warmup_steps = percent_to_steps(optimizer_config.warmup_percent, max_steps)
        rampdown_steps = percent_to_steps(optimizer_config.rampdown_percent, max_steps)
        rampdown_begin = max_steps - rampdown_steps
        learning_rate_diff = learning_rate_max - learning_rate_min

        def trapezoid_lr_schedule(step: int) -> float:
            if step < warmup_steps:
                return learning_rate_max * (step + 1) / warmup_steps
            elif step < rampdown_begin:
                return learning_rate_max
            elif step < max_steps:
                return learning_rate_diff * (max_steps - step) / rampdown_steps + learning_rate_min
            else:
                return learning_rate_min
        
        return trapezoid_lr_schedule
    
    else:
        raise NotImplementedError(f'optimizer schedule kind "{optimizer_config.kind}" is not implemented')

def create_tokenizer(tokenizer_config: TokenizingConfig) -> tiktoken.Encoding:
    '''Create a tokenizer from the provided config.'''
    tokenizer = tiktoken.get_encoding(tokenizer_config.tokenizer_name)
    if tokenizer_config.extra_tokens:
        base_tokenizer = tokenizer
        n_vocab = base_tokenizer.n_vocab

        extra_tokens: dict[str, int] = dict()
        for (i, token_str) in enumerate(tokenizer_config.extra_tokens):
            extra_tokens[token_str] = n_vocab + i

        tokenizer = tiktoken.Encoding(
            name=tokenizer_config.tokenizer_name + "_custom",
            pat_str=base_tokenizer._pat_str,
            mergeable_ranks=base_tokenizer._mergeable_ranks,
            special_tokens={**base_tokenizer._special_tokens, **extra_tokens},
        )
    return tokenizer

def create_data_loader(data_loader_config: DataLoaderConfig, batch_config: BatchConfig, tokenizer: tiktoken.Encoding, tokenizer_config: TokenizingConfig, resume_config: typing.Optional[ResumeConfig]) -> AbstractDataLoader:
    '''Create a data loader from a data loader config.'''
    start_column = resume_config.resume_column if resume_config is not None else 0

    if data_loader_config.kind == 'fineweb':
        return FineWebDataLoader(batch_config.sequences_per_micro_batch, batch_config.sequence_length, data_loader_config.directory, tokenizer, start_column=start_column, shuffle=data_loader_config.shuffle)
    if data_loader_config.kind == 'synth_combo':
        allowed_special_tokens = None
        if tokenizer_config.extra_tokens:
            allowed_special_tokens = set(tokenizer_config.extra_tokens)
        return QaReasoningComboTextDataLoader(batch_config.sequences_per_micro_batch, batch_config.sequence_length, data_loader_config.directory, tokenizer, start_column=start_column, shuffle=data_loader_config.shuffle, allowed_special_tokens=allowed_special_tokens)
    raise NotImplementedError(f"Unknown data loader kind {data_loader_config.kind}")

def create_model(training_config: dict, resume_weights: typing.Optional[str], device: torch.device) -> tuple[AilmV1, AilmV1Config]:
    '''Create the model from the training config'''
    # model_class, config_class = model_name_to_class_and_config(training_config['model_name'])
    assert training_config['model_name'] == 'AilmV1'

    training_config_model_section = training_config.get('model_config') or {}
    model_config = AilmV1Config(**training_config_model_section)

    model = AilmV1(model_config)
    assert model is not None
    if resume_weights:
        model.load_state_dict(torch.load(resume_weights, map_location=device))
    
    model = model.to(device)
    return model, model_config

def create_optimizer(model: nn.Module, optimizer_config: OptimizerConfig, resume_config: typing.Optional[ResumeConfig], device: torch.device) -> torch.optim.Optimizer:
    '''Create the optimizer for our training run.'''
    if optimizer_config.name == 'adamw':
        assert optimizer_config.adamw_betas is not None
        assert optimizer_config.adamw_epsilon is not None
        optimizer = torch.optim.AdamW(
            model.parameters(),
            # lr=0.92,  # Will be overridden by scheduler
            # betas=tuple(optimizer_config.adamw_betas),
            eps=optimizer_config.adamw_epsilon,
            weight_decay=optimizer_config.weight_decay
        )
    else:
        raise NotImplementedError(f"Unimplemented optimizer {optimizer_config.name}")
    
    if resume_config:
        optimizer.load_state_dict(torch.load(resume_config.resume_optimizer, map_location=device))
    
    return optimizer
        
def calculate_optimal_token_budget(parameter_count: int) -> int:
    '''See the Deepmind chinchilla paper: https://arxiv.org/pdf/2203.15556'''
    return parameter_count * 20

def loss_fn(model: nn.Module, inputs: Tensor, targets: Tensor) -> Tensor:
    """Compute cross-entropy loss."""
    logits: Tensor = model(inputs)
    B, T, V = logits.shape
    logits = logits.reshape(B * T, V)
    targets = targets.reshape(B * T)
    loss = torch.nn.functional.cross_entropy(logits, targets, reduction='mean')
    return loss

# MARK: Validation

@dataclasses.dataclass
class ValidationResult:
    perplexity: float
    mean_loss: float
    current_column: int

def validate(model: nn.Module, val_loader: AbstractDataLoader, val_queue: queue.Queue[typing.Tuple[Tensor, Tensor, int]], num_batches: int, should_reset: bool, stall_secs: int, device: torch.device) -> ValidationResult:
    """Run validation and return average loss and perplexity"""
    model.eval()
    total_loss = 0.0
    if should_reset:
        val_loader.reset()
    
    current_column = -1
    with torch.no_grad():
        for _ in range(num_batches):
            val_loader.encode()
            inputs, targets, current_column = val_queue.get(block=True, timeout=stall_secs)
            inputs = inputs.to(device)
            targets = targets.to(device)
            loss = loss_fn(model, inputs, targets)
            total_loss += loss.item()
    
    model.train()
    mean_loss = total_loss / num_batches
    perplexity = math.exp(mean_loss)
    return ValidationResult(perplexity=perplexity, mean_loss=mean_loss, current_column=current_column)

# MARK: Encoding

def create_encoder_worker(tokenizing_config: TokenizingConfig, data_loader: AbstractDataLoader, name: str) -> queue.Queue:
    '''Create and start the encoder worker thread.'''
    encoder_queue = queue.Queue(tokenizing_config.tokenizer_queue_length)
    worker = threading.Thread(name=name, target=encoding_worker, args=(tokenizing_config, encoder_queue, data_loader, name), daemon=True)
    worker.start()
    return encoder_queue

def encoding_worker(tokenizing_config: TokenizingConfig, encoder_queue: queue.Queue, data_loader: AbstractDataLoader, name: str):
    '''Encoder worker thread.'''
    print('Starting tokenizer worker', name)
    if name:
        name_current_thread(name)
    import queue as queue_module
    while True:
        try:
            data_loader.encode()
            prefixes, targets = data_loader.next_batch()
            encoder_queue.put((prefixes, targets, data_loader.column()), block=True, timeout=None)
        except queue_module.ShutDown:
            break
        except Exception as e:
            print(f"Encoder worker error: {e}")
            break

# MARK: Training

def train_model(training_config: dict[str, typing.Any], save_directory: str, no_save: bool):
    '''Train a model.'''
    training_start = time.strftime("%Y%m%d-%H%M")

    if not no_save:
        signal.signal(signal.SIGUSR1, handle_pdb)

    # Initialize config objects
    learning_rate_schedule_config = LearningRateScheduleConfig(**training_config['learning_rate_schedule'])
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

    # Sanity checks
    assert training_loader_config.directory != validation_loader_config.directory
    assert (batch_config.tokens_per_batch % (batch_config.sequence_length * batch_config.sequences_per_micro_batch)) == 0
    assert batch_config.sequences_per_micro_batch <= batch_config.tokens_per_batch

    # Setup device
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"Using CUDA: {torch.cuda.get_device_name(0)}")
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
        print("Using Apple MPS")
    else:
        device = torch.device('cpu')
        print("Using CPU")

    # Seed RNGs
    rng_seed = training_config['rng_seed']
    random.seed(rng_seed)
    torch.manual_seed(rng_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(rng_seed)

    # Create model
    resume_weights = resume_config.resume_weights if resume_config is not None else None
    model, model_config = create_model(training_config, resume_weights, device)

    parameter_count = count_params(model)

    # Calculate token budget
    if training_config.get('token_budget') is None:
        training_config['token_budget'] = calculate_optimal_token_budget(parameter_count)
    token_budget = training_config['token_budget']
 
    update_count = training_config['update_count'] = int(token_budget / batch_config.tokens_per_batch)

    # Calculate gradient accumulation steps
    if training_config.get('gradient_accumulation_steps') is None:
        training_config['gradient_accumulation_steps'] = int(math.ceil(batch_config.tokens_per_batch / batch_config.sequences_per_micro_batch / batch_config.sequence_length))
    gradient_accumulation_steps = training_config['gradient_accumulation_steps']
        
    # Calculate step count
    if training_config.get('step_count') is None:
        training_config['step_count'] = update_count * gradient_accumulation_steps
    step_count = training_config['step_count']

    update_count = training_config['update_count'] = int(step_count / gradient_accumulation_steps)
    tokens_to_process = training_config['tokens_to_process'] = step_count * batch_config.sequence_length * batch_config.sequences_per_micro_batch
    validation_batches = training_config['validation_batches']

    # Create learning rate schedule
    learning_rate_schedule = create_learning_rate_scheduler(learning_rate_schedule_config, update_count)

    # Create optimizer
    optimizer = create_optimizer(model, optimizer_config, resume_config, device)

    # Create LR scheduler
    lr_lambda = lambda step: learning_rate_schedule(step // gradient_accumulation_steps) / learning_rate_schedule_config.learning_rate_max
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # Initialize tokenizer
    tokenizer = create_tokenizer(tokenizing_config)

    # Initialize data loaders
    training_data_loader = create_data_loader(training_loader_config, batch_config, tokenizer, tokenizing_config, resume_config)
    validation_loader = create_data_loader(validation_loader_config, batch_config, tokenizer, tokenizing_config, resume_config=None)

    # Sanity check
    assert tokenizer.n_vocab <= model_config.vocab_size
    
    # Create encoder workers
    training_batch_queue = create_encoder_worker(tokenizing_config, training_data_loader, "Training encoder")
    validation_batch_queue = create_encoder_worker(tokenizing_config, validation_loader, "Validation encoder")

    # Create save directory
    run_name = f'run_{training_start}'
    run_dir_path = os.path.join(save_directory, run_name)
    final_model_checkpoint_name = os.path.join(run_dir_path, 'final_model.pt')
    final_optimizer_checkpoint_name = os.path.join(run_dir_path, 'final_opt.pt')
    receipt_name = os.path.join(run_dir_path, 'receipt.json')
    
    if not no_save:
        os.makedirs(run_dir_path, exist_ok=True)

    inference_manager = None
    # inference_manager = InferenceManager(model, tokenizer, inference_config.max_tokens_to_generate, inference_config.max_batches, inference_config.prompt, inference_config.k, inference_config.temperature)

    # Initialize wandb
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
    
    print('᚛ᚁᚓᚅᚇᚇᚐᚉᚈᚐᚅᚔᚋᚂ᚜')
    print('Ailm LLM Trainer - PyTorch')
    pprint(training_config)

    print(f'Training process pid is {os.getpid()}')
    print('Training started at', training_start)
    print("The model has", format_count(parameter_count), "parameters")
    print('tokens to process', format_count(tokens_to_process))
    print('step count', step_count)
    print('gradient_accumulation_steps', gradient_accumulation_steps)
    print('update count', update_count)
    print('Saving to', final_model_checkpoint_name)

    if not running_caffeinated():
        warn("Warning: This process was not started using the caffeinate utility.")

    # Training state
    model.train()
    scaler = torch.cuda.amp.GradScaler() if device.type == 'cuda' else None
    
    # Running statistics
    total_tokens = 0
    running_loss = 0.0
    running_grad_norm = 0.0
    updates_since_last_log = 0

    step = -1
    training_column = -1

    start_step = resume_config.resume_step if resume_config else 0

    # MARK: Training Loop
    try:
        print("Training loop start")
        training_start_time = stopwatch_start = time.time()
        
        optimizer.zero_grad()
        
        for step in range(start_step, step_count):
            prefixes, targets, training_column = training_batch_queue.get(block=True, timeout=tokenizing_config.trainer_stall_seconds)
            
            prefixes = prefixes.to(device)
            targets = targets.to(device)

            # Forward pass
            if scaler is not None:
                with torch.cuda.amp.autocast():
                    loss = loss_fn(model, prefixes, targets)
                    loss = loss / gradient_accumulation_steps
                scaler.scale(loss).backward()
            else:
                loss = loss_fn(model, prefixes, targets)
                loss = loss / gradient_accumulation_steps
                loss.backward()
            
            if torch.isnan(loss).item():
                warn('Loss is nan! Breaking into debugger.')
                no_save = True
                import pdb
                pdb.set_trace()

            running_loss += loss.item() * gradient_accumulation_steps

            # Perform update
            if ((step + 1) % gradient_accumulation_steps) == 0:
                total_tokens += batch_config.sequences_per_micro_batch * batch_config.sequence_length * gradient_accumulation_steps
                update_start = time.time()

                # Clip gradients
                if scaler is not None:
                    scaler.unscale_(optimizer)
                
                grad_norm = clip_grad_norm_(model.parameters(), optimizer_config.gradient_norm_clipping)
                running_grad_norm += grad_norm.item()

                # Optimizer step
                if scaler is not None:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                
                scheduler.step()
                optimizer.zero_grad()

                update_end = time.time()
                updates_since_last_log += 1

                update_step = (step + 1) // gradient_accumulation_steps

                # Save checkpoint
                if (update_step % intervals_config.save_interval) == 0 and not no_save:
                    save_state(model, final_model_checkpoint_name, optimizer, final_optimizer_checkpoint_name, receipt_name, {
                        'step': step,
                        'wandb_run_name': wandb_run.name if wandb_run else None,
                        'current_column': training_column,
                        'total_tokens': total_tokens,
                        'run_command': run_command,
                        'training_config': training_config,
                    })

                # Run validation
                if (update_step % intervals_config.validation_interval) == 0:
                    validation_start = time.time()
                    validation_result = validate(model, validation_loader, validation_batch_queue, validation_batches, True, tokenizing_config.trainer_stall_seconds, device)
                    validation_end = time.time()
                    validation_duration = validation_end - validation_start
                    
                    if wandb_run:
                        validation_log_message = {
                            'val/loss': validation_result.mean_loss,
                            'val/perplexity': validation_result.perplexity,
                            'val/column': validation_result.current_column,
                            'val/duration': validation_duration,
                        }
                        wandb_run.log(validation_log_message)
                        print('Validation:', validation_log_message)

                # Run inference
                if intervals_config.inference_interval is not None and (update_step % intervals_config.inference_interval) == 0 and inference_manager is not None:
                    inference_manager.infer()

                # Logging
                if (update_step % intervals_config.log_interval) == 0:
                    stopwatch_stop = time.time()
                    stopwatch_duration = stopwatch_stop - stopwatch_start
                    step_duration = stopwatch_duration / updates_since_last_log
                    tokens_processed = updates_since_last_log * batch_config.sequences_per_micro_batch * batch_config.sequence_length * gradient_accumulation_steps
                    tokens_per_second = tokens_processed / stopwatch_duration
                    update_duration = update_end - update_start

                    average_loss_this_update = running_loss / (gradient_accumulation_steps * updates_since_last_log)
                    average_grad_norm = running_grad_norm / updates_since_last_log
                    perplexity = math.exp(average_loss_this_update)

                    current_learning_rate = scheduler.get_last_lr()[0]

                    running_loss = 0.0
                    running_grad_norm = 0.0
                    updates_since_last_log = 0

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

                    elapsed_time = stopwatch_stop - training_start_time
                    mean_time_per_step = elapsed_time / (step + 1)
                    steps_remaining = step_count - step
                    eta_seconds = mean_time_per_step * steps_remaining

                    print(*log_message.values(), format_time(eta_seconds))
                    stopwatch_start = stopwatch_stop

    finally:
        try:
            if not no_save:
                save_state(model, final_model_checkpoint_name, optimizer, final_optimizer_checkpoint_name, receipt_name, {
                    'step': step,
                    'wandb_run_name': wandb_run.name if wandb_run else None,
                    'current_column': training_column,
                    'total_tokens': total_tokens,
                    'run_command': run_command,
                    'training_config': training_config,
                })
        finally:
            if inference_manager:
                inference_manager.finish()

    if wandb_run:
        wandb_run.finish()

def save_state(model: AilmV1, final_model_checkpoint_name: str, optimizer: torch.optim.Optimizer, final_optimizer_checkpoint_name: str, receipt_name: str, receipt_state: typing.Dict[str, typing.Any]):
    torch.save(model.state_dict(), final_model_checkpoint_name)
    torch.save(optimizer.state_dict(), final_optimizer_checkpoint_name)
    if receipt_state:
        with open(receipt_name, 'w') as receipt_file:
            json.dump(receipt_state, receipt_file)

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
