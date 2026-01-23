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

def print_full_tokens(tokenizer, ids, title=""):
    """
    ids: 1D torch tensor on CPU (dtype long), e.g. [seq_len]
    """
    ids_list = ids.tolist()
    toks = tokenizer.convert_ids_to_tokens(ids_list)

    if title:
        print(f"\n===== {title} =====")
    print(f"Total tokens: {len(ids_list)}")

    # 1) 打印完整文本（包含 prompt + completion）
    full_text = tokenizer.decode(ids_list, skip_special_tokens=False)
    print("\n[Full decoded text]")
    print(full_text)

    # 2) 打印每个 token（含 id）
    print("\n[Token list: index | id | token]")
    for i, (tid, tok) in enumerate(zip(ids_list, toks)):
        print(f"{i:04d} | {tid:>6d} | {tok}")


def plot_surprisal_timeline_matplotlib(
    surprisal_scores,
    entropy,
    velocities,
    tokens_text=None,
    context_len=0,
    current_idx=None,
    step_num=0,
    mcmc_step=0,
    accepted=None,
    save_path=None,
    window_size=5,
    peak_distance=10,
    peak_prominence=0.3
):
    """
    Plot cognitive load / surprisal timeline using matplotlib.
    
    Args:
        surprisal_scores: Combined surprisal scores
        entropy: Raw entropy values
        velocities: Raw velocity values
        tokens_text: List of token strings (optional)
        context_len: Length of the prompt/context
        current_idx: Current cut point in MCMC (for highlighting)
        step_num: Current block number
        mcmc_step: Current MCMC step within the block
        accepted: Whether the proposal was accepted (True/False/None)
        save_path: Path to save the figure (optional)
        window_size: Window size for smoothing
        peak_distance: Minimum distance between peaks
        peak_prominence: Minimum prominence for peak detection
    """
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    
    steps = np.arange(len(surprisal_scores))
    
    # Smooth the surprisal scores
    smoothed_surprisal = uniform_filter1d(surprisal_scores, size=window_size)
    
    # Detect peaks
    peaks, properties = find_peaks(
        smoothed_surprisal, 
        distance=peak_distance, 
        prominence=peak_prominence
    )
    
    # Color scheme for regions
    prompt_color = '#90EE90'  # Light green for prompt
    generation_color = '#ADD8E6'  # Light blue for generation
    cut_color = '#FFB6C1'  # Light pink for cut region
    
    # ===== Plot 1: Combined Surprisal =====
    ax1 = axes[0]
    
    # Background regions
    if context_len > 0:
        ax1.axvspan(0, context_len, alpha=0.3, color=prompt_color, label='Prompt')
    ax1.axvspan(context_len, len(steps), alpha=0.2, color=generation_color, label='Generation')
    
    # Highlight cut point region if provided
    if current_idx is not None:
        ax1.axvline(x=current_idx, color='red', linestyle='--', linewidth=2, label=f'Cut Point (idx={current_idx})')
        ax1.axvspan(current_idx, len(steps), alpha=0.15, color=cut_color, label='Rewrite Region')
    
    # Plot smoothed surprisal
    ax1.plot(steps, smoothed_surprisal, color='#2ca02c', linewidth=2, label='Cognitive Load (Smoothed)')
    ax1.fill_between(steps, 0, smoothed_surprisal, alpha=0.3, color='#2ca02c')
    
    # Mark peaks
    if len(peaks) > 0:
        ax1.scatter(peaks, smoothed_surprisal[peaks], color='red', s=100, zorder=5, 
                   marker='o', edgecolors='darkred', linewidths=2, label='Peaks')
        
        # Add peak labels (token text if available)
        for peak in peaks[:10]:  # Limit to top 10 peaks for readability
            label = tokens_text[peak][:15] if tokens_text and peak < len(tokens_text) else f"t{peak}"
            ax1.annotate(label, (peak, smoothed_surprisal[peak]), 
                        textcoords="offset points", xytext=(0, 10), 
                        ha='center', fontsize=8, rotation=45)
    
    ax1.set_ylabel('Surprisal Score', fontsize=11)
    ax1.set_title(f'Block {step_num} | MCMC Step {mcmc_step} | Accepted: {accepted}', fontsize=12, fontweight='bold')
    ax1.legend(loc='upper right', fontsize=9)
    ax1.grid(True, alpha=0.3)
    
    # ===== Plot 2: Entropy =====
    ax2 = axes[1]
    
    # Background regions
    if context_len > 0:
        ax2.axvspan(0, context_len, alpha=0.3, color=prompt_color)
    ax2.axvspan(context_len, len(steps), alpha=0.2, color=generation_color)
    
    if current_idx is not None:
        ax2.axvline(x=current_idx, color='red', linestyle='--', linewidth=2)
        ax2.axvspan(current_idx, len(steps), alpha=0.15, color=cut_color)
    
    ax2.plot(steps, entropy, color='#1f77b4', linewidth=1.5, label='Entropy')
    ax2.fill_between(steps, 0, entropy, alpha=0.3, color='#1f77b4')
    ax2.set_ylabel('Entropy', fontsize=11)
    ax2.legend(loc='upper right', fontsize=9)
    ax2.grid(True, alpha=0.3)
    
    # ===== Plot 3: Velocity =====
    ax3 = axes[2]
    
    # Background regions
    if context_len > 0:
        ax3.axvspan(0, context_len, alpha=0.3, color=prompt_color)
    ax3.axvspan(context_len, len(steps), alpha=0.2, color=generation_color)
    
    if current_idx is not None:
        ax3.axvline(x=current_idx, color='red', linestyle='--', linewidth=2)
        ax3.axvspan(current_idx, len(steps), alpha=0.15, color=cut_color)
    
    ax3.plot(steps, velocities, color='#ff7f0e', linewidth=1.5, label='Velocity (1 - cosine sim)')
    ax3.fill_between(steps, 0, velocities, alpha=0.3, color='#ff7f0e')
    ax3.set_xlabel('Token Position', fontsize=11)
    ax3.set_ylabel('Velocity', fontsize=11)
    ax3.legend(loc='upper right', fontsize=9)
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()
        plt.close()


def plot_surprisal_timeline_plotly(
    surprisal_scores,
    entropy,
    velocities,
    tokens_text=None,
    context_len=0,
    current_idx=None,
    step_num=0,
    mcmc_step=0,
    accepted=None,
    save_path=None,
    window_size=5,
    peak_distance=10,
    peak_prominence=0.3
):
    """
    Plot cognitive load / surprisal timeline using Plotly (interactive).
    """
    if not USE_PLOTLY:
        return plot_surprisal_timeline_matplotlib(
            surprisal_scores, entropy, velocities, tokens_text,
            context_len, current_idx, step_num, mcmc_step, accepted,
            save_path, window_size, peak_distance, peak_prominence
        )
    
    steps = np.arange(len(surprisal_scores))
    
    # Smooth the surprisal scores
    smoothed_surprisal = uniform_filter1d(surprisal_scores, size=window_size)
    
    # Detect peaks
    peaks, _ = find_peaks(smoothed_surprisal, distance=peak_distance, prominence=peak_prominence)
    
    # Create subplots
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        subplot_titles=('Cognitive Load (Surprisal)', 'Entropy', 'Velocity'),
        vertical_spacing=0.08
    )
    
    # Colors
    prompt_color = 'rgba(144, 238, 144, 0.3)'
    generation_color = 'rgba(173, 216, 230, 0.3)'
    cut_color = 'rgba(255, 182, 193, 0.3)'
    
    for row in range(1, 4):
        # Prompt region
        if context_len > 0:
            fig.add_vrect(x0=0, x1=context_len, fillcolor=prompt_color,
                         layer="below", line_width=0, row=row, col=1)
        
        # Generation region
        fig.add_vrect(x0=context_len, x1=len(steps), fillcolor=generation_color,
                     layer="below", line_width=0, row=row, col=1)
        
        # Cut point
        if current_idx is not None:
            fig.add_vline(x=current_idx, line_dash="dash", line_color="red",
                         line_width=2, row=row, col=1)
            fig.add_vrect(x0=current_idx, x1=len(steps), fillcolor=cut_color,
                         layer="below", line_width=0, row=row, col=1)
    
    # Create hover text
    hover_texts = []
    for i in range(len(steps)):
        token = tokens_text[i][:20] if tokens_text and i < len(tokens_text) else f"token_{i}"
        hover_texts.append(f"Step: {i}<br>Token: {token}<br>Surprisal: {smoothed_surprisal[i]:.3f}")
    
    # Plot 1: Surprisal
    fig.add_trace(
        go.Scatter(x=steps, y=smoothed_surprisal, mode='lines',
                  line=dict(color='#2ca02c', width=2),
                  name='Cognitive Load', hovertext=hover_texts, hoverinfo='text'),
        row=1, col=1
    )
    
    # Peak markers
    if len(peaks) > 0:
        peak_texts = [tokens_text[p][:15] if tokens_text and p < len(tokens_text) else f"t{p}" 
                     for p in peaks]
        fig.add_trace(
            go.Scatter(x=peaks, y=smoothed_surprisal[peaks], mode='markers+text',
                      marker=dict(size=12, color='red', symbol='circle',
                                 line=dict(width=2, color='darkred')),
                      text=peak_texts, textposition='top center',
                      name='Peaks'),
            row=1, col=1
        )
    
    # Plot 2: Entropy
    fig.add_trace(
        go.Scatter(x=steps, y=entropy, mode='lines',
                  line=dict(color='#1f77b4', width=1.5),
                  name='Entropy', fill='tozeroy', fillcolor='rgba(31, 119, 180, 0.3)'),
        row=2, col=1
    )
    
    # Plot 3: Velocity
    fig.add_trace(
        go.Scatter(x=steps, y=velocities, mode='lines',
                  line=dict(color='#ff7f0e', width=1.5),
                  name='Velocity', fill='tozeroy', fillcolor='rgba(255, 127, 14, 0.3)'),
        row=3, col=1
    )
    
    # Update layout
    accept_str = "✓ Accepted" if accepted else ("✗ Rejected" if accepted is False else "N/A")
    fig.update_layout(
        title=dict(
            text=f"<b>Block {step_num} | MCMC Step {mcmc_step} | {accept_str}</b>",
            x=0.5, font=dict(size=16)
        ),
        height=800,
        showlegend=True,
        template='plotly_white',
        hovermode='x unified'
    )
    
    fig.update_xaxes(title_text="Token Position", row=3, col=1)
    fig.update_yaxes(title_text="Surprisal", row=1, col=1)
    fig.update_yaxes(title_text="Entropy", row=2, col=1)
    fig.update_yaxes(title_text="Velocity", row=3, col=1)
    
    if save_path:
        fig.write_html(save_path)
    else:
        fig.show()

def compute_surprisal(entropy, velocities, window=5):
    """
    Combines Entropy and Velocity into a single Smoothed Surprisal Score.
    
    Args:
        entropy (list[float]): List of entropy values per token.
        velocities (list[float]): List of velocity values per token.
        window (int): Window size for smoothing (default 5).

    Returns:
        list[float]: The final smoothed surprisal scores.
    """
    # 1. Validation
    if len(entropy) != len(velocities):
        raise ValueError(f"Input lengths mismatch! Entropy: {len(entropy)}, Velocities: {len(velocities)}")
    
    # 2. Create DataFrame
    df = pd.DataFrame({
        "RawEntropy": entropy,
        "RawVelocity": velocities,
        "Cluster": 0  # Dummy cluster for grouping
    })

    # 3. Define Logic Helper
    def calc_rank_surprisal(s):
        # Rank: Largest value = Rank 1 (Most Surprising)
        ranks = s.rank(ascending=False, method="min")
        # Surprisal = -log(Rank / N)
        return -np.log(ranks / (len(s) + 1))

    # 4. Apply Calculation
    grouped = df.groupby("Cluster")
    
    # Calculate separate surprisals
    df["Surprisal_Entropy"] = grouped["RawEntropy"].transform(calc_rank_surprisal)
    df["Surprisal_Velocity"] = grouped["RawVelocity"].transform(calc_rank_surprisal)
    
    # Combine (Sum)
    df["SurprisalScore"] = df["Surprisal_Entropy"] + df["Surprisal_Velocity"]

    # 5. Smoothing (Moving Average)
    # min_periods=1 ensures we get values even for short sequences or edges
    df["Smoothed_Score"] = df["SurprisalScore"].rolling(window=window, center=True, min_periods=1).mean()

    # 6. Return as simple list
    return df["Smoothed_Score"].tolist()

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
        output_hidden_states=True,  # <--- NEW: Request hidden states
    )
    # 3. PROCESS LOGITS (Existing Logic)
    # output.logits is a tuple of tensors. We stack them to get shape (Seq_Len, Batch, Vocab)
    unscaled_logits = torch.stack(output.logits, dim=0)
    scaled_logits = torch.stack(output.scores, dim=0)
    
    # Get the generated tokens (removing the prompt)
    tokens = output.sequences[0][c:]
    prop = output.sequences[0].tolist()

    # Integrity Check
    assert len(tokens) == unscaled_logits.shape[0] == scaled_logits.shape[0]

    # Calculate Log Probs (Existing Logic)
    # Reshape tokens to (Seq, 1, 1) for gathering from the logits tensor
    idx = tokens.view(unscaled_logits.shape[0], 1, 1)

    log_probs_unnorm = (1/temp * torch.gather(F.log_softmax(unscaled_logits, dim=-1), -1, idx)).view(-1).tolist()
    log_probs_norm = torch.gather(F.log_softmax(scaled_logits, dim=-1), -1, idx).view(-1).tolist()

    # --- NEW LOGIC START ---

    # ... inside naive_temp function ...

    # 4. PROCESS HIDDEN STATES (h_last)
    # output.hidden_states is a tuple (one item per generated token).
    # Each item is a tuple of layers. We want the LAST layer (-1).
    
    h_last_steps = []
    for step_tup in output.hidden_states:
        # Get the tensor for the last layer
        last_layer_tensor = step_tup[-1] # Shape could be [1, 72, 3584] or [1, 1, 3584]
        
        # We always want the LAST token's state from that step
        # [:, -1, :] converts [1, 72, 3584] -> [1, 3584]
        #            converts [1, 1, 3584]  -> [1, 3584]
        last_token_state = last_layer_tensor[:, -1, :]  
        
        h_last_steps.append(last_token_state)
    
    # Now all tensors are [Batch, Hidden], so we can stack them
    h_last_stacked = torch.stack(h_last_steps, dim=0) # Shape: [Seq_Len, Batch, Hidden]
    
    # Squeeze to remove batch dim (assuming batch=1) -> (Seq_Len, Hidden_Size)
    h_last = h_last_stacked.squeeze(1).cpu()

  

    # 5. CALCULATE ENTROPY
    # Entropy measures the uncertainty of the model: H(x) = - sum(p(x) * log(p(x)))
    # We use 'unscaled_logits' to measure the BASE model's uncertainty (ignoring temperature).
    
    probs = F.softmax(unscaled_logits, dim=-1)
    log_probs = F.log_softmax(unscaled_logits, dim=-1)
    
    # Calculate entropy for each token step
    entropy_tensor = -torch.sum(probs * log_probs, dim=-1) # Shape: (Seq_Len, Batch)
    
    # Flatten and convert to list
    entropy = entropy_tensor.view(-1).tolist()

    # --- NEW LOGIC END ---

    # Final Integrity Check
    assert len(tokens) == len(log_probs_unnorm) == len(log_probs_norm) == len(entropy) == h_last.size(0)

    # Return prop (list), log_probs (lists), h_last (Tensor), entropy (list)
    return prop, log_probs_norm, log_probs_unnorm, h_last, entropy


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
        gen, lp_norm, lp_unnorm,h_last, entropy = naive_temp(p, gen, temp=temp, seq_len=jump_size+len(gen))
        log_probs_norm.extend(lp_norm)
        log_probs_unnorm.extend(lp_unnorm)

        for _ in tqdm(range(mcmc_steps)):
            attempts+=1
            t = len(gen)
            idx = random.randint(c, t-1)
            # llm query takes the burden of time
            prop, log_prob_prop, target_log_prob_prop,prop_h_last, prop_entropy = naive_temp(p, gen[:idx], temp=temp, seq_len=t)
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
def mcmc_power_samp(
  p : AutoregressiveSampler, 
  context, 
  temp, 
  mcmc_steps, 
  max_new_tokens, 
  block_num=16
  ):
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

    # [NEW] Lists to store the history of the entire generated sequence
    # We only track metrics for the *generated* part (after context 'c')
    full_entropy = [] 
    full_h_last = torch.tensor([], device='cpu') # Initialize empty tensor

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
        gen, lp_norm, lp_unnorm,h_last, entropy = naive_temp(p, gen, temp=temp, seq_len=jump_size+len(gen))
        log_probs_norm.extend(lp_norm)
        log_probs_unnorm.extend(lp_unnorm)
        # 方案：利用切片错位计算 (Vectorized)
        # h_last[:-1] 是从第 0 个到倒数第 2 个
        # h_last[1:]  是从第 1 个到最后 1 个
        # dim=-1 表示在向量维度上计算相似度

        sims = F.cosine_similarity(h_last[:-1], h_last[1:], dim=-1)
        velocities = (1 - sims).tolist() + [0.0]
        print(f"raw_h_last shape: {h_last.shape}")   # Tensor 用 .shape
        print(f"Entropy length: {len(entropy)}")      # List 用 len()
        print(f"Velocities length: {len(velocities)}") # List 用 len()
        surprisal_scores = compute_surprisal(entropy, velocities)

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
            prop, log_prob_prop, target_log_prob_prop,prop_h_last, prop_entropy = naive_temp(p, gen[:idx], temp=temp, seq_len=t)
            
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

def mcmc_power_samp_with_plot(
    p,  # AutoregressiveSampler
    context,
    temp,
    mcmc_steps,
    max_new_tokens,
    block_num=16,
    plot_every=1,  # Plot every N MCMC steps
    save_plots=True,
    plot_dir="mcmc_plots",
    use_plotly=True,
    window_size=5,
    peak_distance=10,
    peak_prominence=0.3
):
    """
    MCMC Power Sampling with integrated surprisal timeline visualization.
    
    Args:
        p: AutoregressiveSampler instance
        context: Initial context token IDs
        temp: Temperature for sampling
        mcmc_steps: Number of MCMC steps per block
        max_new_tokens: Total new tokens to generate
        block_num: Number of blocks
        plot_every: Plot every N MCMC steps (1 = every step)
        save_plots: Whether to save plots to disk
        plot_dir: Directory to save plots
        use_plotly: Use Plotly for interactive plots (else matplotlib)
        window_size: Window size for surprisal smoothing
        peak_distance: Minimum distance between detected peaks
        peak_prominence: Minimum prominence for peak detection
    
    Returns:
        gen: Generated token sequence
        log_probs_norm: Normalized log probabilities
        log_probs_unnorm: Unnormalized log probabilities
        acceptance_ratio: Ratio of accepted proposals
    """
    c = len(context)
    print(f'alpha: {1/temp}')
    
    # Create plot directory
    if save_plots:
        os.makedirs(plot_dir, exist_ok=True)
    
    gen = []
    if context is not None:
        gen = context.copy()
    
    log_probs_norm = []
    log_probs_unnorm = []
    
    # Storage for trajectory data
    full_entropy = []
    full_h_last = torch.tensor([], device='cpu')
    
    print(max_new_tokens)
    assert max_new_tokens % block_num == 0
    jump_size = int(max_new_tokens // block_num)
    print(jump_size)
    
    attempts = 0
    acceptances = 0
    
    # 3. OUTER LOOP (The "Writer")
    for block_idx in tqdm(range(block_num), desc="Blocks"):
        # Generate new tokens for this block
        gen, lp_norm, lp_unnorm, h_last, entropy = naive_temp(
            p, gen, temp=temp, seq_len=jump_size + len(gen)
        )
        log_probs_norm.extend(lp_norm)
        log_probs_unnorm.extend(lp_unnorm)
        
        # Compute velocities from hidden states
        sims = F.cosine_similarity(h_last[:-1], h_last[1:], dim=-1)
        velocities = (1 - sims).tolist() + [0.0]
        
        print(f"raw_h_last shape: {h_last.shape}")
        print(f"Entropy length: {len(entropy)}")
        print(f"Velocities length: {len(velocities)}")
        
        # Compute initial surprisal scores
        surprisal_scores = compute_surprisal(entropy, velocities)
        
        # Get token texts for labeling
        tokens_text = [p.tokenizer.decode([tid]) for tid in gen]
        
        # 4. INNER LOOP (MCMC Steps)
        for mcmc_idx in tqdm(range(mcmc_steps), desc=f"MCMC (Block {block_idx})", leave=False):
            attempts += 1
            t = len(gen)
            
            # Random cut point
            idx = random.randint(c, t - 1)
            
            # 5. GENERATE PROPOSAL
            prop, log_prob_prop, target_log_prob_prop, prop_h_last, prop_entropy = naive_temp(
                p, gen[:idx], temp=temp, seq_len=t
            )
            
            s = len(prop)
            assert len(log_prob_prop) == s - idx
            assert len(target_log_prob_prop) == s - idx
            
            # 6. GATHER DATA FOR COMPARISON
            log_prob_cur = log_probs_norm.copy()[idx - c:s - c]
            target_log_prob_cur = log_probs_unnorm.copy()[idx - c:s - c]
            
            # 7. CALCULATE METROPOLIS-HASTINGS RATIO
            log_r = (sum(target_log_prob_prop) + sum(log_prob_cur) - 
                    sum(target_log_prob_cur) - sum(log_prob_prop))
            
            # 8. ACCEPTANCE DECISION
            accepted = np.random.rand() < np.exp(log_r)
            
            # ========== PLOTTING SECTION ==========
            if mcmc_idx % plot_every == 0:
                # Compute current surprisal for visualization
                current_entropy = list(entropy)  # Current entropy values
                current_velocities = list(velocities)  # Current velocity values
                current_surprisal = compute_surprisal(current_entropy, current_velocities)
                current_tokens = [p.tokenizer.decode([tid]) for tid in gen]
                
                # Determine save path
                if save_plots:
                    if use_plotly and USE_PLOTLY:
                        save_path = os.path.join(
                            plot_dir, 
                            f"block{block_idx:02d}_mcmc{mcmc_idx:04d}.html"
                        )
                    else:
                        save_path = os.path.join(
                            plot_dir,
                            f"block{block_idx:02d}_mcmc{mcmc_idx:04d}.png"
                        )
                else:
                    save_path = None
                
                # Choose plotting function
                if use_plotly and USE_PLOTLY:
                    plot_surprisal_timeline_plotly(
                        surprisal_scores=current_surprisal,
                        entropy=current_entropy,
                        velocities=current_velocities,
                        tokens_text=current_tokens,
                        context_len=c,
                        current_idx=idx,
                        step_num=block_idx,
                        mcmc_step=mcmc_idx,
                        accepted=accepted,
                        save_path=save_path,
                        window_size=window_size,
                        peak_distance=peak_distance,
                        peak_prominence=peak_prominence
                    )
                else:
                    plot_surprisal_timeline_matplotlib(
                        surprisal_scores=current_surprisal,
                        entropy=current_entropy,
                        velocities=current_velocities,
                        tokens_text=current_tokens,
                        context_len=c,
                        current_idx=idx,
                        step_num=block_idx,
                        mcmc_step=mcmc_idx,
                        accepted=accepted,
                        save_path=save_path,
                        window_size=window_size,
                        peak_distance=peak_distance,
                        peak_prominence=peak_prominence
                    )
            # ========== END PLOTTING SECTION ==========
            
            if accepted:
                acceptances += 1
                gen = prop.copy()
                log_probs_norm[idx - c:] = log_prob_prop.copy()
                log_probs_unnorm[idx - c:] = target_log_prob_prop.copy()
                
                # Update entropy and velocities after acceptance
                entropy = prop_entropy
                prop_sims = F.cosine_similarity(prop_h_last[:-1], prop_h_last[1:], dim=-1)
                velocities = (1 - prop_sims).tolist() + [0.0]
                h_last = prop_h_last
                
                # Recompute surprisal scores
                surprisal_scores = compute_surprisal(entropy, velocities)
                tokens_text = [p.tokenizer.decode([tid]) for tid in gen]
                
                del prop
                del log_prob_prop
                del target_log_prob_cur
        
        # 9. CHECK FOR EOS
        if p.tokenizer.eos_token_id in gen:
            eos_idx = gen.index(p.tokenizer.eos_token_id)
            gen = gen[:eos_idx + 1]
            log_probs_norm = log_probs_norm[:eos_idx + 1]
            log_probs_unnorm = log_probs_unnorm[:eos_idx + 1]
            acceptance_ratio = acceptances / attempts
            return gen, log_probs_norm, log_probs_unnorm, acceptance_ratio
    
    # 10. FINAL RETURN
    acceptance_ratio = acceptances / attempts
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
