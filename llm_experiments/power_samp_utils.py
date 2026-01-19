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

# The AutoregressiveSampler is a safety wrapper. 
# It ensures you don't feed the model too much text (crashing it), 
# and it extracts the specific probability distribution for the next word, 
# ready for a sampling strategy (like Greedy, Temperature, or Nucleus sampling) 
# to actually choose the word.
class AutoregressiveSampler:
    def __init__(self, model, tokenizer, device):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.block_size = self.model.config.max_position_embeddings
        # Context Limit: Every LLM has a hard limit on how much text 
        # it can read at once (e.g., 2048 or 4096 tokens).
        # The code extracts this limit (max_position_embeddings) 
        # and saves it as block_size so the sampler knows when to cut off old text.

    # returns log probs
    @torch.no_grad()# This turns off the "learning" mode (gradient calculation). This drastically reduces memory usage and speeds up the code since we are only generating, not training.
    def next_token(self, prefix):
      # Input: 'prefix' is a List of Integers (token IDs) representing the text written so far.
      # Output: A 1D Tensor of Log-Probabilities for the NEXT token.
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
def naive_temp(p : AutoregressiveSampler, context, temp, seq_len):
    c = len(context)
    # 1. SETUP & INITIALIZATION
    # Calculate the length of the prompt (context) so we know where to start reading the answer.

    device = p.device
    tokenizer = p.tokenizer
    # Get the hardware device (GPU/CPU) and tokenizer from the sampler object.

    input_ids = torch.tensor([context], dtype=torch.long, device=device)
    # Convert the Python list context into a PyTorch Tensor on the correct device.
    # We add an extra dimension [context] -> [[...]] because the model expects a Batch dimension.
    
    output = p.model.generate(
        input_ids=input_ids,
        max_new_tokens=seq_len - c,
        do_sample=True,
        temperature=temp,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id,
        return_dict_in_generate=True,
        output_scores=True,
        output_logits=True,
    )
    unscaled_logits = torch.stack(output.logits, dim=0)
    scaled_logits = torch.stack(output.scores, dim=0)
    tokens = output.sequences[0][c:]
    prop = output.sequences[0].tolist()

    assert len(tokens) == unscaled_logits.shape[0] == scaled_logits.shape[0]


    idx = tokens.view(unscaled_logits.shape[0], 1, 1)

    log_probs_unnorm = (1/temp * torch.gather(F.log_softmax(unscaled_logits, dim=-1), -1, idx)).view(-1).tolist()
    log_probs_norm = torch.gather(F.log_softmax(scaled_logits, dim=-1), -1, idx).view(-1).tolist()

    assert len(tokens) == len(log_probs_unnorm) == len(log_probs_norm)

    return prop, log_probs_norm, log_probs_unnorm


# alpha = infty power sampling; temp is for proposal distribution
def max_swap(p : AutoregressiveSampler, context, temp, mcmc_steps, max_new_tokens, block_num=16):
    c = len(context)
    print(f'Temp: {temp}')
    gen = []
    if context is not None:
        gen = context.copy()
    log_probs_norm = []
    log_probs_unnorm = []


    print(max_new_tokens)
    assert max_new_tokens % block_num == 0
    jump_size = int(max_new_tokens // block_num)
    print(jump_size)
    attempts = 0
    acceptances = 0


    for _ in tqdm(range(block_num)):
        gen, lp_norm, lp_unnorm = naive_temp(p, gen, temp=temp, seq_len=jump_size+len(gen))
        log_probs_norm.extend(lp_norm)
        log_probs_unnorm.extend(lp_unnorm)

        for _ in tqdm(range(mcmc_steps)):
            attempts+=1
            t = len(gen)
            idx = random.randint(c, t-1)
            # llm query takes the burden of time
            prop, log_prob_prop, target_log_prob_prop = naive_temp(p, gen[:idx], temp=temp, seq_len=t)
            s = len(prop)
            assert(len(log_prob_prop) == s - idx)
            assert(len(target_log_prob_prop) == s - idx)
            log_prob_cur = log_probs_norm.copy()[idx-c:s-c]
            target_log_prob_cur = log_probs_unnorm.copy()[idx-c:s-c]
            log_r = sum(target_log_prob_prop) - sum(target_log_prob_cur)

            if log_r > 0:
                acceptances+=1
                gen = prop.copy()
                log_probs_norm[idx-c:] = log_prob_prop.copy()
                log_probs_unnorm[idx-c:] = target_log_prob_prop.copy()

                del prop
                del log_prob_prop
                del target_log_prob_cur

        if p.tokenizer.eos_token_id in gen:
            eos_idx = gen.index(p.tokenizer.eos_token_id)
            gen = gen[:eos_idx + 1]
            log_probs_norm = log_probs_norm[:eos_idx + 1]
            log_probs_unnorm = log_probs_unnorm[:eos_idx + 1]
            acceptance_ratio = acceptances/attempts
            return gen, log_probs_norm, log_probs_unnorm, acceptance_ratio

    acceptance_ratio = acceptances/attempts
    return gen, log_probs_norm, log_probs_unnorm, acceptance_ratio

# power sampling with autoregressive mcmc
def mcmc_power_samp(p : AutoregressiveSampler, context, temp, mcmc_steps, max_new_tokens, block_num=16):
    c = len(context)
    # 1. SETUP & INITIALIZATION
    # Calculate the length of the initial prompt (context) so we know 
    # where the "user text" ends and "AI text" begins.
    print(f'alpha: {1/temp}')
    gen = []
    # Initialize the list that will hold our generated token IDs.
    if context is not None:
        gen = context.copy()
    log_probs_norm = []
    log_probs_unnorm = []
    # If the user provided a prompt (context),
    # copy it into our generation buffer to start.
    # Initialize empty lists to store the probability scores of the tokens we generate.
    # 'norm' = Normalized Log Probabilities (Proposal Distribution Q).
    # 'unnorm' = Unnormalized Log Probabilities (Target Distribution P).

    # 2. BLOCK PLANNING
    # Print total tokens to generate (debugging info).
    print(max_new_tokens)
    assert max_new_tokens % block_num == 0
    jump_size = int(max_new_tokens // block_num)
    print(jump_size)
    attempts = 0
    acceptances = 0

    # 3. OUTER LOOP (The "Writer")
    # This loop runs 16 times (block_num). Each time, 
    # it extends the story by 'jump_size' tokens.
    for _ in tqdm(range(block_num)):
        gen, lp_norm, lp_unnorm = naive_temp(p, gen, temp=temp, seq_len=jump_size+len(gen))
        log_probs_norm.extend(lp_norm)
        log_probs_unnorm.extend(lp_unnorm)

        for _ in tqdm(range(mcmc_steps)):
            attempts+=1
            # Increment the attempt counter.
            t = len(gen)
            # Get the current total length of our text.


            idx = random.randint(c, t-1)
            # Pick a RANDOM CUT POINT ('idx').
            # We ensure it is after the prompt ('c') but before the end of the text ('t-1').
            # We are going to try to rewrite everything that comes AFTER this point.
            # llm query takes the burden of time

            # 5. GENERATE PROPOSAL (The Rewrite)
            prop, log_prob_prop, target_log_prob_prop = naive_temp(p, gen[:idx], temp=temp, seq_len=t)
            
            # Take the text up to the cut point (gen[:idx]) and ask the model to generate a NEW ending
            # that reaches the same length 't'.
            # 'prop' = The new proposed text sequence.
            # 'log_prob_prop' = The probabilities of this new ending (Proposal Q).
            # 'target_log_prob_prop' = The target probabilities of this new ending (Target P).
            
            s = len(prop)
            # Get the length of the new proposal (should be same as 't').
            assert(len(log_prob_prop) == s - idx)
            assert(len(target_log_prob_prop) == s - idx)
            # Safety Checks: Ensure the probability lists match 
            # the length of the new segment we generated.

            # 6. GATHER DATA FOR COMPARISON
            log_prob_cur = log_probs_norm.copy()[idx-c:s-c]
            target_log_prob_cur = log_probs_unnorm.copy()[idx-c:s-c]
            # Extract the probability scores of the CURRENT (Old) text segment that we might replace.
            # We need these to calculate if the new version is better than the old version.
           
            # 7. CALCULATE METROPOLIS-HASTINGS RATIO
            log_r = sum(target_log_prob_prop) + sum(log_prob_cur) - sum(target_log_prob_cur) - sum(log_prob_prop)
            # This formula decides if we accept the change.
            # log_r = log(P_new) + log(Q_old|new) - log(P_old) - log(Q_new|old)
            # Roughly: (How good is the new text?) - (How good was the old text?)
            # More detailed explaination:
            #####################################
            # We need to decide if the New Proposal ('prop') is better than the Current Text ('cur').
            # The formula is: log_r = (New Quality - Old Quality) + (Old Luck - New Luck)
            
            # PART A: The Quality Check (Target P)
            # sum(target_log_prob_prop): How much the model *loves* the new answer (The "Energy").
            # sum(target_log_prob_cur):  How much the model *loved* the old answer.
            # If (New - Old) is positive, the new answer is mathematically "smarter".
            
            # PART B: The Luck Correction (Proposal Q)
            # sum(log_prob_prop): How "easy" it was to generate this text randomly (Bias).
            # We subtract this to penalize answers that are just "lucky" or common, 
            # ensuring we favor answers that are truly intelligent, not just probable.
            
            # Formula: log(P_new) + log(Q_old|new) - log(P_old) - log(Q_new|old)
            # More detailed explaination:
            #####################################
            # 8. THE DECISION
            # Generate a random number between 0 and 1.
            # If it is less than exp(log_r), we ACCEPT the proposal.
            # This allows us to always accept better answers, but sometimes accept worse ones (to explore).
            # 
            # We convert the log score back to a probability: np.exp(log_r).
            # Then we compare it against a random coin flip (0 to 1).
            
            # Case 1: New answer is BETTER (log_r > 0) -> exp(log_r) > 1.
            #         We ALWAYS accept (since random number is always < 1).
            
            # Case 2: New answer is WORSE (log_r < 0) -> exp(log_r) is a decimal (e.g., 0.3).
            #         We accept it ONLY if the random coin flip is low (30% chance).
            #         This "exploration" allows the model to escape bad reasoning traps.
            if np.random.rand() < np.exp(log_r):
                acceptances+=1
                # We accepted! Increment the counter.

                gen = prop.copy()
                # Overwrite the main sequence 'gen' with the new proposal 'prop'.


                log_probs_norm[idx-c:] = log_prob_prop.copy()
                log_probs_unnorm[idx-c:] = target_log_prob_prop.copy()
                # Update our probability logs to match the new text.
                # We slice [idx-c:] because we only changed the text after the cut point.

                del prop
                del log_prob_prop
                del target_log_prob_cur
                # Clean up memory (delete variables we don't need anymore).

        # 9. CHECK FOR COMPLETION (EOS)
        # After every block, check if the model wrote the "End of Sentence" token.
        if p.tokenizer.eos_token_id in gen:
            eos_idx = gen.index(p.tokenizer.eos_token_id)
            # Find exactly where the EOS token is.

            gen = gen[:eos_idx + 1]
            log_probs_norm = log_probs_norm[:eos_idx + 1]
            log_probs_unnorm = log_probs_unnorm[:eos_idx + 1]
            # Cut off the text and logs right at the EOS token (discard anything after it).
            
            acceptance_ratio = acceptances/attempts
            # Calculate the final acceptance ratio.

            return gen, log_probs_norm, log_probs_unnorm, acceptance_ratio
            # Return the final result immediately (exit the function).

    # 10. FINAL RETURN
    # If we finished all blocks without seeing an EOS token, return what we have.        
    acceptance_ratio = acceptances/attempts
    return gen, log_probs_norm, log_probs_unnorm, acceptance_ratio

# This function is a Prompt Engineer's adapter. 
# Its job is to take your raw math question and "dress it up" 
# in the specific format that each different AI model expects to see.
# Different models (like Qwen, Phi, or Tulu) require different formatting rules
#—some want raw text, while others require a "Chat Template" (User/Assistant structure).
# these PROMPT. COT, BASE are stored in constants.py
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
