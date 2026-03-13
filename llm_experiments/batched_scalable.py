import os
from contextlib import nullcontext
from glob import glob
import json
import random
from tqdm import tqdm
import argparse
import math 

import pandas as pd
import numpy as np
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
from torch.utils.data import Dataset, DataLoader
from dataclasses import dataclass
from datasets import Dataset, load_dataset, concatenate_datasets


import torch
import torch.nn as nn
from torch.nn import functional as F
import transformers
from transformers.cache_utils import DynamicCache

from grader_utils.parse_utils import parse_answer
from constants import *

### DESCRIPTION ###
# power sampling to sample from p^{alpha}, where p is the base model
# takes in 1/alpha (temperature) as an argument (default 0.25), and mcmc_power_samp implements sampling from p^{alpha} 


class AutoregressiveSampler:
    def __init__(self, model, tokenizer, device):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.block_size = self.model.config.max_position_embeddings

    # returns log probs
    @torch.no_grad()
    def next_token(self, prefix):
        device = self.device
        torch_prefix = torch.tensor([prefix], dtype=torch.long, device=device)
        prefix_cond = torch_prefix if torch_prefix.size(1) <= self.block_size else torch_prefix[:, -self.block_size:]
        output = self.model(prefix_cond)
        logits = output.logits
        logits = logits[0, -1, :]
        probs = F.softmax(logits, dim=-1)
        return torch.log(probs)



# # returns probabilities (normed)
# def normalize(dist):
#     probs = F.softmax(dist, dim=-1)
#     return probs

# # returns sum of logits (product of distributions p*q)
# def dist_product(logit_p, logit_q):
#     return logit_p+logit_q

# # returns logit scaled by temp (temperature scaling p^(1/tau))
# def dist_temp_scale(logit_p, temp):
#     return logit_p * torch.tensor(1 / temp, dtype=logit_p.dtype, device=logit_p.device)

# low-temperature sampling proposal distribution
@torch.no_grad()
def low_temp(p : AutoregressiveSampler, context, temp):
    """
    Samples 1 token using low temp sampling
    
    :param p:
     Base Model
    :type p: AutoregressiveSampler
    :param context: input into model
    :param temp: alpha = 1/temp

    Returns:
    :prop:  Prompt + generated tokens
    : log_probs_norm: target_distribution prob for generated tokens
    """
    device = p.device
    input_ids = torch.tensor([context], dtype=torch.long, device=device)
    logits = p.model(input_ids).logits[0, -1, :]
    probs = F.softmax(logits / temp, dim=-1)
    token = torch.multinomial(probs, 1)
    del logits, probs, input_ids
    return token


@torch.no_grad()
def top_K_from_base(p : AutoregressiveSampler, context, K, temp, L, block_size):
    """
    Generate L blocks of size block_size. Use base probabilities to get the top K blocks .

    :param p: Base model
    :type p: AutoregressiveSampler
    :param context: Token IDs for the current context
    :param K: Number of top blocks to return
    :param temp: Alpha = 1/temp for power scaling
    :param L: evaluation budget from which to choose top K
    :param block_size: Number of tokens in each block

    Returns:
    :top_blocks: Token sequences of the K most probable blocks; shape (K, block_size)
    :log_power_probs: Log powered probs (1/temp)*log p(block) for each top-K block; shape (K,)
    :past_kv: Cached key-values from the model for continuation
    """

    device = p.device
    c = len(context)
    input_ids = torch.tensor([context], dtype=torch.long, device=device)
    tokenizer = p.tokenizer

    # Replicate context L times to generate L independent block samples
    # Shape: (L, len(context))
    tokens_col = input_ids.repeat(L, 1)

    # Generate L blocks of block_size tokens each
    output = p.model.generate(
            input_ids=tokens_col,
            max_new_tokens=block_size,
            do_sample=True,
            temperature=1,  # Sample uniformly; we'll reweight by power distribution
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
            return_dict_in_generate=True,
            output_scores=False,
            output_logits=True,
        )
    
    # Raw model logits; shape: (block_size, L, vocab_size)
    unscaled_logits = torch.stack(output.logits, dim=0)

    # Generated tokens for all L blocks; shape: (L, block_size)
    generated_tokens = output.sequences[:, c:]
    
    log_softmax_logits = F.log_softmax(unscaled_logits, dim=-1)  # (block_size, L, vocab_size)
    
    # Transpose generated_tokens to match: (block_size, L)
    generated_tokens_t = generated_tokens.t()
    
    # Create indices for gathering: (block_size, L, 1)
    idx = generated_tokens_t.unsqueeze(-1)
    
    # Gather log-probs of generated tokens: (block_size, L)
    log_probs_tokens = torch.gather(log_softmax_logits, -1, idx).squeeze(-1)
    
    # Sum log-probs across block positions for each of the L blocks: (L,)
    block_log_probs = log_probs_tokens.sum(dim=0)
    
    # Get top K blocks by their base model log-probability
    top_K_logprobs, top_K_indices = torch.topk(block_log_probs, k=min(K, L), dim=0)
    
    # Extract the top K blocks: shape (K, block_size)
    top_blocks = generated_tokens[top_K_indices]
    
    # Compute log of power-scaled probabilities for the top K blocks
    log_power_probs = (1/temp) * top_K_logprobs

    # Get cached key-values from context for efficient continuation
    output_cache = p.model(input_ids, use_cache=True)
    past_kv = output_cache.past_key_values

    del output, unscaled_logits, log_softmax_logits, generated_tokens, block_log_probs
    del generated_tokens_t, idx, log_probs_tokens, top_K_indices
    del output_cache

    return top_blocks, log_power_probs, past_kv, top_K_logprobs


@torch.no_grad()
def sample_remainder_block(p: AutoregressiveSampler, context, temp, L, remainder_block_size):
    """
    Generate the final remainder block of length (q + 1 = T - B*floor(T/B) + 1).

    Samples L candidate completions from the base model, computes power-scaled
    probabilities p(completion)^(1/temp) for each, and samples one completion
    according to those powered probabilities (batch low-temperature sampling).

    :param p: Base model
    :type p: AutoregressiveSampler
    :param context: Token IDs for the current context
    :param temp: Alpha = 1/temp for power scaling
    :param L: Number of candidate completions to sample from the base model
    :param remainder_block_size: Number of tokens to generate (q + 1)

    Returns:
    :sampled_block: List of token IDs for the sampled remainder completion
    """
    device = p.device
    tokenizer = p.tokenizer
    ctx_len = len(context)
    input_ids = torch.tensor([context], dtype=torch.long, device=device)

    # Generate L completions of length remainder_block_size from the base model
    output = p.model.generate(
        input_ids=input_ids.repeat(L, 1),
        max_new_tokens=remainder_block_size,
        do_sample=True,
        temperature=1,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id,
        return_dict_in_generate=True,
        output_scores=False,
        output_logits=True,
    )

    # completions: (L, remainder_block_size)
    completions = output.sequences[:, ctx_len:]
    # logits: (remainder_block_size, L, vocab_size)
    logits = torch.stack(output.logits, dim=0)

    # Gather log-probs of each generated token under the base model
    log_softmax = F.log_softmax(logits, dim=-1)
    idx = completions.t().unsqueeze(-1)  # (remainder_block_size, L, 1)
    log_probs_tokens = torch.gather(log_softmax, -1, idx).squeeze(-1)  # (remainder_block_size, L)

    # Sum across positions for block log-prob, then power-scale: (1/temp) * log p(block)
    block_log_probs = log_probs_tokens.sum(dim=0)  # (L,)
    power_log_probs = (1 / temp) * block_log_probs  # (L,)

    # Normalize in log space and sample one completion according to powered probabilities
    power_probs = torch.exp(power_log_probs - torch.logsumexp(power_log_probs, dim=0))
    sampled_idx = torch.multinomial(power_probs, 1).item()
    sampled_block = completions[sampled_idx].tolist()

    del output, logits, log_softmax, idx, log_probs_tokens, block_log_probs, power_log_probs, power_probs, completions

    return sampled_block



@torch.no_grad()
def compute_xi_batched(p: AutoregressiveSampler, context, top_blocks, temp, M, T, past_kv, H, batch_size=None, debug=False):
    """
    Estimate ζ(block) for all K candidate blocks in batched generate calls.
    For each of K blocks, run M rollouts of length H to estimate lookahead value.

    :param p: Base model
    :type p: AutoregressiveSampler
    :param context: Token IDs for the current context (before sampled block)
    :param top_blocks: Tensor of K candidate block sequences; shape (K, block_size)
    :param temp: Alpha = 1/temp for power scaling
    :param M: Number of rollouts per candidate block
    :param T: Total generated trajectory length + prompt length (pass in T+c from before)
    :param past_kv: Cached key-values from the context for efficient generation
    :param H: rollout horizon length
    :param batch_size: Max rollouts per mini-batch (default: 0 ->  no batching)
    :param debug: None = silent | "verbose" = timing/memory info

    Returns:
    :log_xis: Log of mean ζ estimate for each candidate block; shape (K,)
    :log_xis_loo: Log of leave-one-out ζ estimates for each candidate; shape (K, M)
    """
    device = p.device
    tokenizer = p.tokenizer
    K = top_blocks.shape[0]
    block_size = top_blocks.shape[1]
    total_rollouts = K * M

    if batch_size == 0:
        batch_size = total_rollouts

    # Expand each block M times: shape (K*M, block_size)
    expanded_blocks = top_blocks.unsqueeze(1).repeat(1, M, 1).reshape(total_rollouts, block_size)

    ctx_len = past_kv.get_seq_length()
    c = ctx_len + block_size  # Context length + sampled block

    # Process rollouts in mini-batches
    import time
    def _gpu():
        if torch.cuda.is_available():
            alloc = torch.cuda.memory_allocated() / 1024**3
            reserved = torch.cuda.memory_reserved() / 1024**3
            return f"alloc={alloc:.2f}GB, reserved={reserved:.2f}GB"
        return "n/a"
    _dbg = (lambda msg: print(f"    [DEBUG] {msg}", flush=True)) if debug == "verbose" else (lambda msg: None)

    log_probs_power_parts = []
    for start in range(0, total_rollouts, batch_size):
        end = min(start + batch_size, total_rollouts)
        chunk_size = end - start

        _dbg(f"chunk {start}-{end} (size={chunk_size}) | GPU: {_gpu()}")

        t0 = time.time()
        expanded_kv = DynamicCache()
        for layer_idx in range(len(past_kv)):
            key = past_kv.layers[layer_idx].keys
            value = past_kv.layers[layer_idx].values
            expanded_kv.update(
                key.repeat_interleave(chunk_size, dim=0),
                value.repeat_interleave(chunk_size, dim=0),
                layer_idx,
            )
        _dbg(f"KV expand: {time.time()-t0:.2f}s | expanded seq_len={expanded_kv.get_seq_length()}, layers={len(expanded_kv)} | GPU: {_gpu()}")

        chunk_blocks = expanded_blocks[start:end]  # (chunk_size, block_size)

        # Cache position must cover all block_size tokens: [ctx_len, ctx_len+1, ..., ctx_len+block_size-1]
        # A scalar (single position) would give wrong positional encodings for every token after the first.
        cache_position = torch.arange(ctx_len, ctx_len + block_size, dtype=torch.long, device=device)

        # Rollout should be limited to T - i dont think so anymore
        # max_new_tokens = min(T-c, H)
        
        t0 = time.time()
        _dbg(f"Starting generate: input_ids={chunk_blocks.shape}, max_new_tokens={H}, cache_position={cache_position}")
        output = p.model.generate(
            input_ids=chunk_blocks,
            past_key_values=expanded_kv,
            cache_position=cache_position,
            max_new_tokens=H,
            do_sample=True,
            temperature=1,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
            return_dict_in_generate=True,
            output_scores=False,
            output_logits=True,
        )
        _dbg(f"generate done: {time.time()-t0:.2f}s | output seq_len={output.sequences.shape} | GPU: {_gpu()}")

        # Tokens generated after the block: shape (chunk_size, num_new_tokens)
        tokens_generated = output.sequences[:, block_size:]
        num_new_tokens = tokens_generated.shape[1]

        # Sanity check: output.logits must have exactly one entry per generated token.
        # If this fails, the block_size slice is misaligned with what generate() returned.
        assert len(output.logits) == num_new_tokens, (
            f"Logit/token mismatch: got {len(output.logits)} logit steps "
            f"but {num_new_tokens} generated tokens. "
            f"output.sequences.shape={output.sequences.shape}, block_size={block_size}"
        )
        unscaled_logits = torch.stack(output.logits, dim=0)  # (num_new_tokens, chunk_size, vocab_size)

        log_softmax_logits = F.log_softmax(unscaled_logits, dim=-1)
        idx = tokens_generated.t().unsqueeze(-1)  # (num_new_tokens, chunk_size, 1)
        gathered = torch.gather(log_softmax_logits, -1, idx).squeeze(-1)  # (num_new_tokens, chunk_size)

        # Mask out tokens after EOS
        eos_id = tokenizer.eos_token_id
        is_eos = (tokens_generated.t() == eos_id) # (num_new_tokens, chunk_size)
        cumulative_eos = is_eos.cumsum(dim=0)
        valid_mask = (cumulative_eos == 0) | ((cumulative_eos == 1) & is_eos)
        masked_gathered = gathered * valid_mask.float()

        # Compute power scaling contribution: (1/temp - 1) * sum(log p(tokens))
        chunk_log_probs = ((1/temp - 1) * masked_gathered).sum(dim=0)  # (chunk_size,)
        log_probs_power_parts.append(chunk_log_probs)

        del output, unscaled_logits, log_softmax_logits, gathered, idx
        del is_eos, cumulative_eos, valid_mask, masked_gathered, expanded_kv

    # (K*M,) log values — never exponentiate to avoid underflow
    log_probs_power = torch.cat(log_probs_power_parts, dim=0)

    # Reshape to (K, M) in log space
    log_matrix = log_probs_power.reshape(K, M)

    # log(mean_m exp(x_km)) = logsumexp(x_km) - log(M)
    log_xis = torch.logsumexp(log_matrix, dim=1) - math.log(M)  # (K,)

    # log of LOO mean: log((sum_m exp(x_km) - exp(x_ks)) / (M-1))
    # = log_total_k + log(1 - exp(x_ks - log_total_k)) - log(M-1)
    log_total = torch.logsumexp(log_matrix, dim=1, keepdim=True)  # (K, 1)
    log_xis_loo = log_total + torch.log1p(-torch.exp(log_matrix - log_total)) - math.log(M - 1)  # (K, M)

    return log_xis, log_xis_loo




# power sampling using lookahead approximations
@torch.no_grad()
def batched_scalable_power_samp(p : AutoregressiveSampler, prompt, temp, M, T, K, L, block_size, H, batch_size_xi=0, debug=None):
    """
    Main loop for sampling. Each iteration samples a block of tokens.

    :param p: Base model
    :type p: AutoregressiveSampler
    :param prompt: Prompt added depending on dataset
    :param temp: Alpha = 1/temp.
    :param M: Number of rollouts per candidate block
    :param T: Trajectory Length (total tokens to generate)
    :param K: Number of top candidate blocks to evaluate
    :param L: Number of candidate blocks sampled from base model
    :param block_size: Number of tokens per block
    :param H: horizon length for rollouts
    :param batch_size_xi: Batches for computing rollouts (memory efficiency)
    :param debug: None = silent | "verbose" = timing/memory info | "probs" = per-block probabilities
    """

    print(f'alpha: {1/temp}')

    # T + c is length prompt + total generated tokens
    c = len(prompt)
    total_input = []
    if prompt is not None:
        total_input = prompt.copy()

    import time
    def gpu_mem():
        if torch.cuda.is_available():
            alloc = torch.cuda.memory_allocated() / 1024**3
            reserved = torch.cuda.memory_reserved() / 1024**3
            return f"alloc={alloc:.2f}GB, reserved={reserved:.2f}GB"
        return "n/a"
    dbg = (lambda msg: print(f"  [DEBUG] {msg}", flush=True)) if debug == "verbose" else (lambda msg: None)

    # Compute number of full blocks and remainder
    num_full_blocks = T // block_size
    remainder = T % block_size

    argmax_match_count = 0
    blocks_processed = 0

    # Process full blocks. If blocks§
    for t in tqdm(range(num_full_blocks), desc="Scalable power blocks", unit="block"):

        # Sample L candidate blocks and select top K by base model likelihood
        t0 = time.time()
        dbg(f"Before top_K_from_base | GPU: {gpu_mem()}")
        top_blocks, log_power_probs, past_kv, top_K_logprobs = top_K_from_base(p, total_input, K, temp, L, block_size)
        # top_blocks: (K, block_size), log_power_probs: (K,)
        dbg(f"top_K_from_base: {time.time()-t0:.2f}s | ctx_len={len(total_input)}, K={K}, L={L} | GPU: {gpu_mem()}")

        # For each of the K candidate blocks, run M rollouts to estimate lookahead value ζ
        # This requires expanding the block dimension to K*M
        t0 = time.time()
        log_xis, log_xis_loo = compute_xi_batched(p, total_input, top_blocks, temp, M, T +c, past_kv, H, batch_size=batch_size_xi, debug=debug)
        # log_xis: (K,), log_xis_loo: (K, M)
        dbg(f"compute_xi_batched: {time.time()-t0:.2f}s | total_rollouts={K*M}, batch_size={batch_size_xi} | GPU: {gpu_mem()}")
        del past_kv

        # Compute selection probabilities for top K blocks using power distribution
        # All in log space to avoid underflow
        log_unnorm = log_power_probs + log_xis
        log_probs_a_pow = log_unnorm - torch.logsumexp(log_unnorm, dim=0)
        probs_a_pow = torch.exp(log_probs_a_pow)

        
        log_unnorm_loo = log_power_probs.unsqueeze(1) + log_xis_loo  # (K, M)
        log_probs_loo = log_unnorm_loo - torch.logsumexp(log_unnorm_loo, dim=0, keepdim=True)
        probs_a_pow_loo = torch.exp(log_probs_loo)

        # Sum across rollouts
        probs_a_pow_loo_summed = probs_a_pow_loo.sum(dim=1)

        # Variance-reduced selection probabilities
        probs_jk = M * probs_a_pow - ((M-1)/M) * probs_a_pow_loo_summed
        probs_jk = probs_jk.clamp(min=0)
        probs_jk = probs_jk / probs_jk.sum()

        if debug == "probs":
            def fmt(t): return [float(f"{v:.3g}") for v in t.tolist()]
            print(f"  ---- Block {t+1} ----")
            print(f"  block_log_power_probs : {fmt(log_power_probs)}")
            print(f"  log_xis               : {fmt(log_xis)}")
            print(f"  probs_no_jk           : {fmt(probs_a_pow)}")
            print(f"  probs_jk              : {fmt(probs_jk)}")
            print(f"  base_probs            : {fmt(torch.exp(top_K_logprobs))}")
            print(f"  argmax match %        : {100.0 * argmax_match_count / blocks_processed:.1f}% ({argmax_match_count}/{blocks_processed})")
            
        # Track how often argmax(probs_a_pow) == argmax(probs_jk)
        argmax_match_count += int(probs_a_pow.argmax().item() == probs_jk.argmax().item())
        blocks_processed += 1

        # Sample one of the K blocks according to the power distribution
        sampled_block_idx = torch.multinomial(probs_jk, 1).item()
        sampled_block = top_blocks[sampled_block_idx]

        # Append the sampled block to the trajectory
        total_input.extend(sampled_block.tolist())

        # Check for early stopping (EOS token)
        if p.tokenizer.eos_token_id in sampled_block:
            eos_idx = sampled_block.tolist().index(p.tokenizer.eos_token_id)
            total_input = total_input[:-len(sampled_block)] + sampled_block[:eos_idx+1].tolist()
            break

        del top_blocks, log_power_probs, log_xis, probs_a_pow
        del log_xis_loo, probs_a_pow_loo, probs_a_pow_loo_summed, probs_jk

    
    if blocks_processed > 0:
        match_pct = 100.0 * argmax_match_count / blocks_processed
        print(f"argmax(probs_a_pow) == argmax(probs_jk): {argmax_match_count}/{blocks_processed} blocks ({match_pct:.1f}%)")

    # Generate the remainder block: q+1 = T - B*floor(T/B) + 1 tokens.
    # Skip if EOS was already hit in the full-blocks loop.
    if remainder > 0 and total_input[-1] != p.tokenizer.eos_token_id:
        remainder_block_size = remainder
        t0 = time.time()
        dbg(f"Before remainder block | remainder_size={remainder_block_size} | GPU: {gpu_mem()}")
        sampled_rem_block = sample_remainder_block(p, total_input, temp, L, remainder_block_size)
        dbg(f"remainder block done: {time.time()-t0:.2f}s | GPU: {gpu_mem()}")

        # Append remainder block, stopping at EOS if present within it
        if p.tokenizer.eos_token_id in sampled_rem_block:
            eos_idx = sampled_rem_block.index(p.tokenizer.eos_token_id)
            total_input.extend(sampled_rem_block[:eos_idx + 1])
        else:
            total_input.extend(sampled_rem_block)

    return total_input


def format_prompt(question, model, tokenizer, cot=True):
    if model == "qwen":
        format_str = PROMPT + question
        if cot:
            format_str+=COT
        else:
            format_str+=BASE

    elif model == "qwen_math":
        format_str = PROMPT + question
        if cot:
            format_str+=COT
        else:
            format_str+=BASE

    elif model == "qwen_math_grpo":
        content_str = PROMPT + question
        if cot:
            content_str+=COT
        else:
            content_str+=BASE
        answer_context = [{"role": "user", "content": content_str}]
        format_str = tokenizer.apply_chat_template(answer_context, tokenize=False, add_generation_prompt=True)

    elif model == "phi_grpo":
        content_str = PROMPT + question
        if cot:
            content_str+=COT
        else:
            content_str+=BASE
        answer_context = [{"role": "user", "content": content_str}]
        format_str = tokenizer.apply_chat_template(answer_context, tokenize=False, add_generation_prompt=True)

    elif model == "phi":
        content_str = PROMPT + question
        if cot:
            content_str+=COT
        else:
            content_str+=BASE
        answer_context = [{"role": "user", "content": content_str}]
        format_str = tokenizer.apply_chat_template(answer_context, tokenize=False, add_generation_prompt=True)

    elif model == "tulu":
        content_str = PROMPT + question
        if cot:
            content_str+=COT
        else:
            content_str+=BASE
        answer_context = [{"role": "user", "content": content_str}]
        format_str = tokenizer.apply_chat_template(answer_context, tokenize=False, add_generation_prompt=True)

    return format_str
