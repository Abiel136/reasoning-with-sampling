import os

from contextlib import nullcontext
from glob import glob
import json
import random
from tqdm import tqdm
import argparse

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



# returns probabilities (normed)
def normalize(dist):
    probs = F.softmax(dist, dim=-1)
    return probs

# returns sum of logits (product of distributions p*q)
def dist_product(logit_p, logit_q):
    return logit_p+logit_q

# returns logit scaled by temp (temperature scaling p^(1/tau))
def dist_temp_scale(logit_p, temp):
    return logit_p * torch.tensor(1 / temp, dtype=logit_p.dtype, device=logit_p.device)

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
    c = len(context)
    device = p.device
    tokenizer = p.tokenizer
    input_ids = torch.tensor([context], dtype=torch.long, device=device)
    output = p.model.generate(
        input_ids=input_ids,
        max_new_tokens=1,
        do_sample=True,
        temperature=temp,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id, 
        return_dict_in_generate=True,
        output_scores=False,
        output_logits=False,
    )

    # New token
    token = output.sequences[0][c:]

    return token


@torch.no_grad()
def top_K_from_base(p : AutoregressiveSampler, context, K, temp):
    """
    Get the top K tokens by base model probability and their powered log-probabilities.

    :param p: Base model
    :type p: AutoregressiveSampler
    :param context: Token IDs for the current context
    :param K: Number of top tokens to return
    :param temp: Alpha = 1/temp for power scaling

    Returns:
    :top_indices: Token IDs of the K most probable tokens under the base model; shape (K,)
    :power_probs: Powered probs p(token)^(1/temp) for each top-K token; length K array
    """
    device = p.device
    tokenizer = p.tokenizer
    input_ids = torch.tensor([context], dtype=torch.long, device=device)
    output = p.model.generate(
        input_ids=input_ids,
        max_new_tokens=1,
        do_sample=True,
        temperature=1,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id,
        return_dict_in_generate=True,

        # Dont need the scaled logits
        output_scores=False,
        output_logits=True,
    )  

    # unscaled_logits shape: (num_new_tokens, 1, vocab_size)
    unscaled_logits = torch.stack(output.logits, dim=0)

    top_indices = torch.topk(unscaled_logits.squeeze(1), k=K, dim=-1).indices.squeeze(0)
    top_indices_3d = top_indices.view(1,1,K)
    power_log_probs = (1/temp * torch.gather(F.log_softmax(unscaled_logits, dim=-1), -1, top_indices_3d)).view(-1)
    power_probs = torch.exp(power_log_probs)
    # top_indices shape: (K) — the token IDs
    return top_indices, power_probs





@torch.no_grad()
def compute_xi(p: AutoregressiveSampler, context, test_token, temp, M, T):
    """
    Estimate ζ(xt) via M rollouts from the base model conditioned on context + test_token.
    Each rollout computes prod_{s>t} p(xs)^(1/temp - 1) along the sampled trajectory.

    :param p: Base model
    :type p: AutoregressiveSampler
    :param context: Token IDs for the current context (before test_token)
    :param test_token: Candidate token xt to evaluate
    :param temp: Alpha = 1/temp for power scaling
    :param M: Number of rollouts for Monte Carlo estimate
    :param T: Total trajectory length (prompt + generation)

    Returns:
    :xi: Monte Carlo estimate of ζ(xt), averaged over M rollouts; scalar tensor
    :xi_loos: Leave-one-out estimates of ζ(xt), each excluding one rollout; shape (M,)
    """
    context = context +[test_token]

    device = p.device
    tokenizer = p.tokenizer
    input_ids = torch.tensor([context], dtype=torch.long, device=device)
    total = []
    max_new_tokens = T - len(context)
    for rollout in range(M):
        c = len(context)
        output = p.model.generate(
            input_ids=input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=1,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
            return_dict_in_generate=True,

            # Dont need the scaled logits
            output_scores=False,
            output_logits=True,
        )

        tokens = output.sequences[0][c:]
        unscaled_logits = torch.stack(output.logits, dim=0)

        idx = tokens.view(unscaled_logits.shape[0], 1, 1)
        log_probs_power = ((1/temp-1) * torch.gather(F.log_softmax(unscaled_logits, dim=-1), -1, idx)).view(-1).sum()
        probs_power = torch.exp(log_probs_power)
        total += [probs_power]

    total_tensor = torch.stack(total)
    xi = total_tensor.mean() 
    # xi_loo = sum(total[:-1])/(M-1)
    xi_loos = (total_tensor.sum()-total_tensor)/(M-1)
    return xi, xi_loos




# power sampling using lookahead approximations
@torch.no_grad()
def scalable_power_samp(p : AutoregressiveSampler, prompt, temp, M, T, K):
    """
    Main loop for sampling
    
    :param p: Base mode;
    :type p: AutoregressiveSampler
    :param context: Prompt added depending on dataset
    :param temp: Alpha = 1/temp. 
    :param M: Number of rollouts - can change later
    :param T: Trajectory Length
    :param K: size of vocab for expectation of xi
    """

    print(f'alpha: {1/temp}')

    c = len(prompt)
    total_input = []
    if prompt is not None:
        total_input = prompt.copy()

    # All the way upto the last token which is sampled using low-temp
    for t in range(T-1):
        # G is the set of promising candidates according to the base model
        G, power_probs= top_K_from_base(p, total_input, K, temp)
        xis = []

        # List of lists containing every loo for each x_t
        xis_loos = []
        
        for token in G:
            # token is xt and the xt's are everything in G. Need to find all the p^ for each xt and then sample from this
            xi, xi_loos = compute_xi(p, total_input, token, temp, M, T+c)
            xis += [xi]
            xis_loos += [xi_loos]

        # probs_jk = get_jk(p, G, xis, xis_loo, temp)
        
        xis_tensor = torch.stack(xis)
        unnorm_probs = power_probs*xis_tensor
        probs_a_pow = unnorm_probs/ unnorm_probs.sum()

        # xis_loo_matrix shape: (K, M) — each row is LOO estimates for one token across rollouts                                                                                                                             
        xis_loo_matrix = torch.stack(xis_loos)  

        # Broadcast power_probs (K,) -> (K, 1) to multiply with (K, M)
        unnorm_loo = power_probs.unsqueeze(1) * xis_loo_matrix

        # Normalize over tokens (dim=0) for each rollout s; shape: (K, M)
        probs_a_pow_loo = unnorm_loo / unnorm_loo.sum(dim=0, keepdim=True)

        # Now sum all the rollouts: shape(K,)
        probs_a_pow_loo_summed = probs_a_pow_loo.sum(dim =1)

        # Shape (K, )
        probs_jk = M*probs_a_pow - ((M-1)/M) * probs_a_pow_loo_summed

        probs_jk = probs_jk.clamp(min=0)                                                                                                                                                                         
        probs_jk = probs_jk / probs_jk.sum()

        sampled_idx = torch.multinomial(probs_jk, 1).item()
        sampled_token = G[sampled_idx].item()
        total_input.append(sampled_token)

    last_token = low_temp(p, total_input, temp).item()
    total_input.append(last_token)


    if p.tokenizer.eos_token_id in total_input:
            eos_idx = total_input.index(p.tokenizer.eos_token_id)
            total_input = total_input[:eos_idx + 1]

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
