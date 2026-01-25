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
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.ndimage import uniform_filter1d



from grader_utils.parse_utils import parse_answer
from constants import *

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    USE_PLOTLY = True
except ImportError:
    USE_PLOTLY = False
    print("Plotly not available, using matplotlib for plotting")


# =============================================================================
# ADAPTIVE CUT-POINT SELECTION: Bin-Bandit with Surprisal Prior
# =============================================================================

class BinBandit:
    """
    EXP3-style multi-armed bandit for bin selection.
    
    Partitions the cut-point window into B bins and learns which bins
    lead to more accepted proposals, without any additional model calls.
    """
    
    def __init__(
        self, 
        num_bins=10, 
        gamma=0.1,      # Exploration rate for bin selection
        eta=0.1,        # Learning rate for weight updates
        decay=0.02,     # Weight decay for non-stationarity
        epsilon=1e-8,   # Small constant for numerical stability
        verbose=True    # Whether to print detailed logs
    ):
        """
        Args:
            num_bins: Number of bins (B)
            gamma: Exploration floor for EXP3 (mix with uniform)
            eta: Learning rate for exponential weight updates
            decay: Weight decay factor (lambda) for forgetting
            epsilon: Small constant for numerical stability
            verbose: Whether to print detailed logs
        """
        self.num_bins = num_bins
        self.gamma = gamma
        self.eta = eta
        self.decay = decay
        self.epsilon = epsilon
        self.verbose = verbose
        
        # Initialize weights uniformly
        self.weights = np.ones(num_bins, dtype=np.float64)
        
        # Statistics tracking
        self.bin_attempts = np.zeros(num_bins, dtype=np.int64)
        self.bin_accepts = np.zeros(num_bins, dtype=np.int64)
        
        if self.verbose:
            print(f"\n{'='*60}")
            print(f"[BinBandit] Initialized with {num_bins} bins")
            print(f"  - gamma (exploration): {gamma}")
            print(f"  - eta (learning rate): {eta}")
            print(f"  - decay: {decay}")
            print(f"{'='*60}\n")
    
    def get_bin_distribution(self):
        """
        Compute EXP3-style bin selection distribution.
        
        Returns:
            p_bin: Array of probabilities for each bin
        """
        # Normalize weights
        w_sum = np.sum(self.weights)
        p_exploit = self.weights / (w_sum + self.epsilon)
        
        # Mix with uniform for exploration
        p_uniform = np.ones(self.num_bins) / self.num_bins
        p_bin = (1 - self.gamma) * p_exploit + self.gamma * p_uniform
        
        return p_bin
    
    def sample_bin(self):
        """
        Sample a bin according to the EXP3 distribution.
        
        Returns:
            bin_idx: Sampled bin index
            p_bin: The probability distribution used
        """
        p_bin = self.get_bin_distribution()
        bin_idx = np.random.choice(self.num_bins, p=p_bin)
        return bin_idx, p_bin
    
    def update(self, bin_idx, reward, p_bin, verbose_step=False):
        """
        Update bin weights based on reward (acceptance).
        
        Args:
            bin_idx: The bin that was selected
            reward: Reward signal (1 for accept, 0 for reject, or clipped log_r)
            p_bin: The probability distribution used for selection
            verbose_step: Whether to print this specific update
        """
        self.bin_attempts[bin_idx] += 1
        if reward > 0:
            self.bin_accepts[bin_idx] += 1
        
        old_weight = self.weights[bin_idx]
        
        # EXP3-style importance-weighted update
        importance_weight = reward / (p_bin[bin_idx] + self.epsilon)
        self.weights[bin_idx] *= np.exp(self.eta * importance_weight)
        
        # Prevent numerical overflow
        if np.max(self.weights) > 1e6:
            self.weights /= np.max(self.weights)
        
        if self.verbose and verbose_step:
            print(f"    [Bandit Update] Bin {bin_idx}: weight {old_weight:.4f} -> {self.weights[bin_idx]:.4f} (reward={reward:.2f})")
    
    def apply_decay(self):
        """
        Apply weight decay to handle non-stationarity.
        Moves weights toward uniform distribution.
        """
        old_weights = self.weights.copy()
        uniform = np.ones(self.num_bins)
        self.weights = (1 - self.decay) * self.weights + self.decay * uniform
        
        if self.verbose:
            print(f"  [Bandit Decay] Applied decay (λ={self.decay})")
            print(f"    Max weight change: {np.max(np.abs(self.weights - old_weights)):.6f}")
    
    def get_stats(self):
        """Return statistics about bin performance."""
        # Compute acceptance rates, avoiding divide-by-zero warning
        with np.errstate(divide='ignore', invalid='ignore'):
            acceptance_rates = np.where(
                self.bin_attempts > 0,
                self.bin_accepts / self.bin_attempts,
                0.0
            )
        return {
            'weights': self.weights.copy(),
            'attempts': self.bin_attempts.copy(),
            'accepts': self.bin_accepts.copy(),
            'acceptance_rates': acceptance_rates
        }


class AdaptiveCutPointSampler:
    """
    Adaptive cut-point selection using:
    1. Surprisal prior (within-bin localization)
    2. Bin-bandit (coarse adaptation via EXP3)
    3. Refractory masking (anti-repeat)
    
    NOTE: Window size L is FIXED and determined by:
        L = max_new_tokens // block_num (i.e., jump_size)
    
    For example: max_new_tokens=3072, block_num=16 -> L=192
    With num_bins=10: first 9 bins have 19 tokens, last bin has 21 tokens
    """
    
    def __init__(
        self,
        window_size,            # L: fixed window size = jump_size = max_new_tokens // block_num
        num_bins=10,            # B: number of bins
        delta=0.1,              # Mix factor for surprisal prior with uniform
        epsilon_explore=0.1,    # Forced exploration probability
        refractory_radius=10,   # W: radius for anti-repeat masking
        refractory_factor=0.01, # Downweight factor for refractory region
        bandit_gamma=0.1,       # Exploration rate for bandit
        bandit_eta=0.1,         # Learning rate for bandit
        bandit_decay=0.02,      # Weight decay for bandit
        smoothing_epsilon=1e-6, # Small constant for surprisal smoothing
        verbose=True            # Whether to print detailed logs
    ):
        """
        Args:
            window_size: Fixed window size L = max_new_tokens // block_num
            num_bins: Number of bins (B)
            delta: Mix factor for surprisal prior with uniform
            epsilon_explore: Probability of forced uniform exploration
            refractory_radius: Radius W for anti-repeat masking
            refractory_factor: Multiplicative factor for refractory region
            bandit_gamma: Exploration rate for EXP3
            bandit_eta: Learning rate for EXP3 updates
            bandit_decay: Weight decay for non-stationarity
            smoothing_epsilon: Small constant added to surprisal scores
            verbose: Whether to print detailed logs
        """
        self.window_size = window_size  # L = jump_size
        self.num_bins = num_bins        # B
        self.verbose = verbose
        
        # Compute bin sizes: L = B * K + remainder
        # First (B-1) bins have size K, last bin has size K + remainder
        self.base_bin_size = window_size // num_bins  # K
        self.remainder = window_size % num_bins
        
        # Bin boundaries: bins[i] = (start, end) for bin i
        # First (num_bins - 1) bins have base_bin_size tokens
        # Last bin has (base_bin_size + remainder) tokens
        self.bin_boundaries = []
        pos = 0
        for b in range(num_bins):
            if b < num_bins - 1:
                bin_size = self.base_bin_size
            else:
                bin_size = self.base_bin_size + self.remainder  # Last bin gets remainder
            self.bin_boundaries.append((pos, pos + bin_size))
            pos += bin_size
        
        self.delta = delta
        self.epsilon_explore = epsilon_explore
        self.refractory_radius = refractory_radius
        self.refractory_factor = refractory_factor
        self.smoothing_epsilon = smoothing_epsilon
        
        # Initialize bandit
        self.bandit = BinBandit(
            num_bins=num_bins,
            gamma=bandit_gamma,
            eta=bandit_eta,
            decay=bandit_decay,
            verbose=verbose
        )
        
        # CRITICAL FIX: Store last accepted cut position in GLOBAL coordinates
        # This prevents the refractory mask from drifting when the window slides
        self.last_accepted_global_idx = None
        
        # Statistics
        self.total_samples = 0
        self.exploration_samples = 0
        
        if self.verbose:
            print(f"\n{'='*60}")
            print(f"[AdaptiveCutPointSampler] Configuration:")
            print(f"  - Window size (L): {window_size} (fixed = max_new_tokens // block_num)")
            print(f"  - Num bins (B): {num_bins}")
            print(f"  - Base bin size (K): {self.base_bin_size}")
            print(f"  - Remainder: {self.remainder} (added to last bin)")
            print(f"  - Bin sizes: first {num_bins-1} bins = {self.base_bin_size}, last bin = {self.base_bin_size + self.remainder}")
            print(f"  - Bin boundaries: {self.bin_boundaries}")
            print(f"  - Delta (uniform mix): {delta}")
            print(f"  - Epsilon (exploration prob): {epsilon_explore}")
            print(f"  - Refractory radius (W): {refractory_radius}")
            print(f"  - Refractory factor: {refractory_factor}")
            print(f"  - NOTE: Refractory center stored in GLOBAL coordinates")
            print(f"{'='*60}\n")
    
    def get_bin_for_position(self, local_idx):
        """Get which bin a local position belongs to."""
        for b, (start, end) in enumerate(self.bin_boundaries):
            if start <= local_idx < end:
                return b
        return self.num_bins - 1  # Default to last bin
    
    def compute_surprisal_prior(self, surprisal_scores):
        """
        Compute surprisal-based prior distribution over window positions.
        
        Args:
            surprisal_scores: Surprisal scores for the window (length should = window_size)
            
        Returns:
            p_prior: Prior distribution over local window positions [0, L)
        """
        L = len(surprisal_scores)
        
        if L == 0:
            return np.ones(self.window_size) / self.window_size
        
        # Pad or truncate to window_size
        if L < self.window_size:
            surprisal_scores = np.concatenate([
                surprisal_scores,
                np.ones(self.window_size - L) * np.mean(surprisal_scores) if L > 0 else np.ones(self.window_size - L)
            ])
        elif L > self.window_size:
            surprisal_scores = surprisal_scores[:self.window_size]
        
        # Ensure non-negative and add epsilon
        s_tilde = np.maximum(surprisal_scores, 0) + self.smoothing_epsilon
        
        # Normalize to get surprisal prior
        p_sur = s_tilde / (np.sum(s_tilde) + self.smoothing_epsilon)
        
        # Mix with uniform
        p_uniform = np.ones(self.window_size) / self.window_size
        p_prior = (1 - self.delta) * p_sur + self.delta * p_uniform
        
        return p_prior
    
    def apply_refractory_mask(self, p_prior, window_start, actual_window_size):
        """
        Apply refractory masking to prevent repeated cuts at same location.
        
        CRITICAL FIX: Uses GLOBAL coordinates for refractory center.
        The mask is applied only if the last accepted position falls within
        the current window.
        
        Args:
            p_prior: Prior distribution over local window positions
            window_start: Global index where the window starts
            actual_window_size: Size of the current window
            
        Returns:
            p_masked: Masked and renormalized prior distribution
        """
        if self.last_accepted_global_idx is None:
            return p_prior
        
        # Convert global refractory center to local coordinate
        refractory_local = self.last_accepted_global_idx - window_start
        
        # Check if refractory center is within or near the current window
        # It affects the window if it's within [−W, actual_window_size + W)
        if refractory_local < -self.refractory_radius or refractory_local >= actual_window_size + self.refractory_radius:
            # Refractory center is too far from current window, no masking needed
            if self.verbose:
                print(f"    Refractory center (global={self.last_accepted_global_idx}, local={refractory_local}) outside window range, skipping mask")
            return p_prior
        
        p_masked = p_prior.copy()
        
        # OPTIONAL FIX: Vectorized refractory masking (more efficient)
        lo = max(0, refractory_local - self.refractory_radius)
        hi = min(len(p_masked), refractory_local + self.refractory_radius + 1)
        
        if lo < hi:
            p_masked[lo:hi] *= self.refractory_factor
        
        # Renormalize
        total = np.sum(p_masked)
        if total > self.smoothing_epsilon:
            p_masked /= total
        else:
            # Fallback to uniform if all mass was suppressed
            p_masked = np.ones(len(p_masked)) / len(p_masked)
        
        return p_masked
    
    def sample_cut_point(self, context_len, seq_len, surprisal_scores, mcmc_step=None, block_idx=None):
        """
        Sample a cut point using the adaptive method.
        
        The window covers the TRAILING L tokens of the generated sequence:
            window_start = max(context_len, seq_len - L)
            window covers positions [window_start, seq_len)
        
        Args:
            context_len: Length of prompt/context (c)
            seq_len: Current total sequence length (t)
            surprisal_scores: Surprisal scores for the window (should have length = actual_window_size)
            mcmc_step: Current MCMC step (for logging)
            block_idx: Current block index (for logging)
            
        Returns:
            idx: Global cut point index
            sampling_info: Dict with sampling details for logging (includes sampled_bin_idx for update)
        """
        self.total_samples += 1
        
        # Window covers the trailing L tokens
        # s = max(c, t - L)
        window_start = max(context_len, seq_len - self.window_size)
        actual_window_size = seq_len - window_start
        
        if self.verbose:
            print(f"\n  [CutPoint Sample #{self.total_samples}] Block={block_idx}, MCMC={mcmc_step}")
            print(f"    Sequence: context_len(c)={context_len}, seq_len(t)={seq_len}")
            print(f"    Window: start(s)={window_start}, actual_size={actual_window_size}, expected(L)={self.window_size}")
        
        # Handle edge case: very short sequence
        if actual_window_size <= 1:
            idx = context_len
            if self.verbose:
                print(f"    -> Edge case: window too small, using idx={idx}")
            return idx, {'method': 'edge_case', 'idx': idx, 'window_start': window_start, 'sampled_bin_idx': None}
        
        # Ensure surprisal_scores matches actual_window_size
        surprisal_scores = np.array(surprisal_scores)
        if len(surprisal_scores) != actual_window_size:
            if self.verbose:
                print(f"    Adjusting surprisal_scores: {len(surprisal_scores)} -> {actual_window_size}")
            if len(surprisal_scores) > actual_window_size:
                surprisal_scores = surprisal_scores[-actual_window_size:]
            else:
                pad_size = actual_window_size - len(surprisal_scores)
                mean_val = np.mean(surprisal_scores) if len(surprisal_scores) > 0 else 1.0
                surprisal_scores = np.concatenate([np.ones(pad_size) * mean_val, surprisal_scores])
        
        # IMPORTANT FIX: Derive bin boundaries directly from actual_window_size
        # to avoid empty bins from integer truncation scaling
        if actual_window_size < self.window_size:
            # Recompute boundaries for smaller window
            base_size = actual_window_size // self.num_bins
            remainder = actual_window_size % self.num_bins
            current_boundaries = []
            pos = 0
            for b in range(self.num_bins):
                if b < self.num_bins - 1:
                    bin_size = base_size
                else:
                    bin_size = base_size + remainder
                # Ensure non-empty bins (at least 1 token if possible)
                if bin_size == 0 and pos < actual_window_size:
                    bin_size = 1
                end_pos = min(pos + bin_size, actual_window_size)
                current_boundaries.append((pos, end_pos))
                pos = end_pos
            if self.verbose:
                print(f"    Scaled bin boundaries for smaller window: {current_boundaries}")
        else:
            current_boundaries = self.bin_boundaries
        
        # Compute surprisal prior
        # Ensure non-negative and add epsilon
        s_tilde = np.maximum(surprisal_scores, 0) + self.smoothing_epsilon
        p_sur = s_tilde / (np.sum(s_tilde) + self.smoothing_epsilon)
        
        # Mix with uniform
        p_uniform = np.ones(actual_window_size) / actual_window_size
        p_prior = (1 - self.delta) * p_sur + self.delta * p_uniform
        
        # Apply refractory masking (CRITICAL FIX: pass window_start for global->local conversion)
        p_masked = self.apply_refractory_mask(p_prior, window_start, actual_window_size)
        
        # Log surprisal statistics
        if self.verbose:
            top_k = min(5, actual_window_size)
            top_indices = np.argsort(surprisal_scores)[-top_k:][::-1]
            print(f"    Surprisal stats: mean={np.mean(surprisal_scores):.3f}, max={np.max(surprisal_scores):.3f}")
            print(f"    Top-{top_k} surprisal positions (local): {top_indices.tolist()}")
        
        # Log refractory info with GLOBAL coordinates
        if self.verbose and self.last_accepted_global_idx is not None:
            refractory_local = self.last_accepted_global_idx - window_start
            print(f"    Refractory center: global={self.last_accepted_global_idx}, local={refractory_local}, radius={self.refractory_radius}")
        
        sampling_info = {
            'window_start': window_start,
            'window_size': self.window_size,
            'actual_window_size': actual_window_size,
            'context_len': context_len,
            'seq_len': seq_len,
            'bin_boundaries': current_boundaries,
            'sampled_bin_idx': None  # Will be set if bandit is used
        }
        
        # Epsilon-exploration: uniform random
        if np.random.rand() < self.epsilon_explore:
            self.exploration_samples += 1
            local_idx = np.random.randint(0, actual_window_size)
            idx = window_start + local_idx
            sampling_info.update({
                'method': 'epsilon_exploration',
                'local_idx': local_idx,
                'idx': idx,
                'sampled_bin_idx': None  # No bin sampled during exploration
            })
            if self.verbose:
                print(f"    -> EXPLORATION: uniform random, local_idx={local_idx}, global_idx={idx}")
            return idx, sampling_info
        
        # Sample bin using bandit
        sampled_bin_idx, p_bin = self.bandit.sample_bin()
        
        # Get bin boundaries for current window
        bin_start, bin_end = current_boundaries[sampled_bin_idx]
        
        # Safety checks for empty bins
        if bin_end <= bin_start:
            if self.verbose:
                print(f"    WARNING: Empty bin {sampled_bin_idx}, range=[{bin_start}, {bin_end}), using fallback")
            # Fallback: sample uniformly from entire window
            local_idx = np.random.randint(0, actual_window_size)
            
            # CRITICAL FIX (Issue 2): Recompute the ACTUAL bin from the fallback position
            # This ensures we update the bin that actually produced the cut point
            actual_bin_idx = None
            for b, (bs, be) in enumerate(current_boundaries):
                if bs <= local_idx < be:
                    actual_bin_idx = b
                    break
            
            if self.verbose:
                print(f"    Fallback sampled local_idx={local_idx}, actual_bin={actual_bin_idx} (not empty-bin {sampled_bin_idx})")
            
            # Use actual_bin_idx for update, NOT sampled_bin_idx
            sampled_bin_idx = actual_bin_idx  # Override with actual bin
        else:
            if self.verbose:
                print(f"    Bandit selected bin {sampled_bin_idx} (p={p_bin[sampled_bin_idx]:.4f}), range=[{bin_start}, {bin_end})")
                print(f"    Bin weights: {np.round(self.bandit.weights, 3).tolist()}")
            
            # Sample within bin using masked prior
            bin_probs = p_masked[bin_start:bin_end]
            bin_sum = np.sum(bin_probs)
            
            if bin_sum > self.smoothing_epsilon:
                bin_probs = bin_probs / bin_sum
            else:
                bin_probs = np.ones(bin_end - bin_start) / (bin_end - bin_start)
            
            local_in_bin = np.random.choice(len(bin_probs), p=bin_probs)
            local_idx = bin_start + local_in_bin
        
        # Convert to global index
        idx = window_start + local_idx
        
        # Store the ACTUAL bin used (may differ from originally sampled if fallback occurred)
        sampling_info.update({
            'method': 'bandit_surprisal',
            'sampled_bin_idx': sampled_bin_idx,  # This is now the ACTUAL bin (fixed for fallback)
            'bin_start': bin_start,
            'bin_end': bin_end,
            'local_idx': local_idx,
            'idx': idx,
            'p_bin': p_bin.tolist(),
            'surprisal_at_cut': float(surprisal_scores[local_idx]) if local_idx < len(surprisal_scores) else None
        })
        
        if self.verbose:
            surp_val = sampling_info.get('surprisal_at_cut', 'N/A')
            print(f"    -> BANDIT+SURPRISAL: sampled_bin={sampled_bin_idx}, local_idx={local_idx}, global_idx={idx}")
            print(f"       Surprisal at cut: {surp_val}")
        
        return idx, sampling_info
    
    def update_on_accept(self, global_idx, log_r, sampled_bin_idx=None, p_bin=None):
        """
        Update state after an accepted proposal.
        
        CRITICAL FIX: 
        - Uses GLOBAL coordinates for refractory center
        - Uses sampled_bin_idx (not recomputed) for bandit update
        
        Args:
            global_idx: GLOBAL index of the accepted cut point
            log_r: Log acceptance ratio
            sampled_bin_idx: The bin index that was SAMPLED (not recomputed)
            p_bin: Bin probability distribution used (if available)
        """
        old_refractory = self.last_accepted_global_idx
        self.last_accepted_global_idx = global_idx  # Store in GLOBAL coordinates
        
        # Compute reward (binary or clipped log_r)
        reward = 1.0  # Binary reward for acceptance
        
        # CRITICAL FIX: Use sampled_bin_idx directly, don't recompute from position
        if sampled_bin_idx is not None and p_bin is not None:
            self.bandit.update(sampled_bin_idx, reward, p_bin, verbose_step=self.verbose)
            if self.verbose:
                print(f"    [Bandit Update] Using sampled_bin_idx={sampled_bin_idx} (not recomputed)")
        
        if self.verbose:
            print(f"    [ACCEPTED] log_r={log_r:.4f}, exp(log_r)={np.exp(min(log_r, 700)):.4f}")
            print(f"      Refractory center (GLOBAL): {old_refractory} -> {global_idx}")
    
    def update_on_reject(self, global_idx, log_r, sampled_bin_idx=None, p_bin=None):
        """
        Update state after a rejected proposal.
        
        IMPORTANT NOTE: With reward=0, EXP3 weights don't change (exp(0)=1).
        This update is for statistics tracking only.
        
        Args:
            global_idx: GLOBAL index of the rejected cut point
            log_r: Log acceptance ratio
            sampled_bin_idx: The bin index that was SAMPLED
            p_bin: Bin probability distribution used (if available)
        """
        # IMPORTANT: With R=0, w_b *= exp(0) = 1, so weights don't change
        # This is for statistics tracking only
        if sampled_bin_idx is not None and p_bin is not None:
            self.bandit.update(sampled_bin_idx, 0.0, p_bin, verbose_step=self.verbose)
            if self.verbose:
                print(f"    [Bandit Update] Reject with R=0 (weights unchanged), sampled_bin_idx={sampled_bin_idx}")
        
        if self.verbose:
            print(f"    [REJECTED] log_r={log_r:.4f}, exp(log_r)={np.exp(min(log_r, 700)):.4f}")
    
    def end_block(self, block_idx=None):
        """
        Called at the end of each block to apply weight decay.
        """
        if self.verbose:
            print(f"\n  [Block {block_idx} End] Applying weight decay...")
        self.bandit.apply_decay()
        
        if self.verbose:
            stats = self.get_stats()
            print(f"    Current acceptance rates per bin: {np.round(stats['bandit_stats']['acceptance_rates'], 3).tolist()}")
            print(f"    Total attempts: {self.total_samples}, Exploration: {self.exploration_samples} ({stats['exploration_rate']:.1%})")
            print(f"    Last accepted global idx: {self.last_accepted_global_idx}")
    
    def reset(self):
        """
        Reset the sampler state for a new sequence.
        """
        if self.verbose:
            print("\n[AdaptiveCutPointSampler] Resetting state for new sequence...")
        self.bandit = BinBandit(
            num_bins=self.num_bins,
            gamma=self.bandit.gamma,
            eta=self.bandit.eta,
            decay=self.bandit.decay,
            verbose=self.verbose
        )
        self.last_accepted_global_idx = None  # Reset to global coordinates
        self.total_samples = 0
        self.exploration_samples = 0
    
    def get_stats(self):
        """Return statistics about the adaptive sampler."""
        return {
            'bandit_stats': self.bandit.get_stats(),
            'total_samples': self.total_samples,
            'exploration_samples': self.exploration_samples,
            'exploration_rate': self.exploration_samples / max(1, self.total_samples),
            'last_accepted_global_idx': self.last_accepted_global_idx  # Global coordinates
        }


# =============================================================================
# EXISTING UTILITY FUNCTIONS (unchanged)
# =============================================================================

def print_full_tokens(tokenizer, ids, title=""):
    """
    ids: 1D torch tensor on CPU (dtype long), e.g. [seq_len]
    """
    ids_list = ids.tolist()
    toks = tokenizer.convert_ids_to_tokens(ids_list)

    if title:
        print(f"\n===== {title} =====")
    print(f"Total tokens: {len(ids_list)}")

    # 1) Print full text (including prompt + completion)
    full_text = tokenizer.decode(ids_list, skip_special_tokens=False)
    print("\n[Full decoded text]")
    print(full_text)

    # 2) Print each token (with id)
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
    peak_prominence=0.3,
    bandit_stats=None,
    sampling_info=None
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
        bandit_stats: Statistics from the bin-bandit (optional)
        sampling_info: Information about how the cut point was sampled (optional)
    """
    n_plots = 4 if bandit_stats is not None else 3
    fig, axes = plt.subplots(n_plots, 1, figsize=(14, 3.5 * n_plots), sharex=False)
    
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
    
    # Build title with sampling info
    title = f'Block {step_num} | MCMC Step {mcmc_step} | Accepted: {accepted}'
    if sampling_info:
        method = sampling_info.get('method', 'unknown')
        title += f' | Method: {method}'
    ax1.set_title(title, fontsize=12, fontweight='bold')
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
    
    # ===== Plot 4: Bandit Weights (if available) =====
    if bandit_stats is not None and n_plots > 3:
        ax4 = axes[3]
        
        weights = bandit_stats['weights']
        acceptance_rates = bandit_stats['acceptance_rates']
        num_bins = len(weights)
        
        x = np.arange(num_bins)
        width = 0.35
        
        bars1 = ax4.bar(x - width/2, weights / np.max(weights), width, 
                       label='Normalized Weights', color='#9467bd', alpha=0.8)
        bars2 = ax4.bar(x + width/2, acceptance_rates, width,
                       label='Acceptance Rate', color='#d62728', alpha=0.8)
        
        ax4.set_xlabel('Bin Index', fontsize=11)
        ax4.set_ylabel('Value', fontsize=11)
        ax4.set_title('Bin-Bandit Statistics', fontsize=11)
        ax4.set_xticks(x)
        ax4.legend(loc='upper right', fontsize=9)
        ax4.grid(True, alpha=0.3, axis='y')
    
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
    peak_prominence=0.3,
    bandit_stats=None,
    sampling_info=None
):
    """
    Plot cognitive load / surprisal timeline using Plotly (interactive).
    """
    if not USE_PLOTLY:
        return plot_surprisal_timeline_matplotlib(
            surprisal_scores, entropy, velocities, tokens_text,
            context_len, current_idx, step_num, mcmc_step, accepted,
            save_path, window_size, peak_distance, peak_prominence,
            bandit_stats, sampling_info
        )
    
    steps = np.arange(len(surprisal_scores))
    
    # Smooth the surprisal scores
    smoothed_surprisal = uniform_filter1d(surprisal_scores, size=window_size)
    
    # Detect peaks
    peaks, _ = find_peaks(smoothed_surprisal, distance=peak_distance, prominence=peak_prominence)
    
    # Determine number of rows
    n_rows = 4 if bandit_stats is not None else 3
    subplot_titles = ['Cognitive Load (Surprisal)', 'Entropy', 'Velocity']
    if bandit_stats is not None:
        subplot_titles.append('Bin-Bandit Statistics')
    
    # Create subplots
    fig = make_subplots(
        rows=n_rows, cols=1,
        shared_xaxes=False,
        subplot_titles=subplot_titles,
        vertical_spacing=0.08
    )
    
    # Colors
    prompt_color = 'rgba(144, 238, 144, 0.3)'
    generation_color = 'rgba(173, 216, 230, 0.3)'
    cut_color = 'rgba(255, 182, 193, 0.3)'
    
    for row in range(1, 4):  # First 3 plots share the same x-axis structure
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
    
    # Plot 4: Bandit statistics (if available)
    if bandit_stats is not None:
        weights = bandit_stats['weights']
        acceptance_rates = bandit_stats['acceptance_rates']
        num_bins = len(weights)
        
        fig.add_trace(
            go.Bar(x=list(range(num_bins)), 
                   y=weights / np.max(weights),
                   name='Normalized Weights',
                   marker_color='#9467bd',
                   opacity=0.8),
            row=4, col=1
        )
        
        fig.add_trace(
            go.Bar(x=list(range(num_bins)),
                   y=acceptance_rates,
                   name='Acceptance Rate',
                   marker_color='#d62728',
                   opacity=0.8),
            row=4, col=1
        )
        
        fig.update_xaxes(title_text="Bin Index", row=4, col=1)
        fig.update_yaxes(title_text="Value", row=4, col=1)
    
    # Update layout
    accept_str = "✓ Accepted" if accepted else ("✗ Rejected" if accepted is False else "N/A")
    method_str = ""
    if sampling_info:
        method_str = f" | Method: {sampling_info.get('method', 'unknown')}"
    
    fig.update_layout(
        title=dict(
            text=f"<b>Block {step_num} | MCMC Step {mcmc_step} | {accept_str}{method_str}</b>",
            x=0.5, font=dict(size=16)
        ),
        height=250 * n_rows,
        showlegend=True,
        template='plotly_white',
        hovermode='x unified',
        barmode='group'
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
    
    # Handle empty input
    if len(entropy) == 0:
        return []
    
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


# =============================================================================
# AUTOREGRESSIVE SAMPLER (unchanged)
# =============================================================================

class AutoregressiveSampler:
    def __init__(self, model, tokenizer, device):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.block_size = self.model.config.max_position_embeddings

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


# =============================================================================
# HELPER FUNCTIONS (unchanged)
# =============================================================================

def normalize(dist):
    probs = F.softmax(dist, dim=-1)
    return probs

def dist_product(logit_p, logit_q):
    return logit_p+logit_q

def dist_temp_scale(logit_p, temp):
    return logit_p * torch.tensor(1 / temp, dtype=logit_p.dtype, device=logit_p.device)


def naive_temp(p : AutoregressiveSampler, context, temp, seq_len):
    c = len(context)
    device = p.device
    tokenizer = p.tokenizer
    input_ids = torch.tensor([context], dtype=torch.long, device=device)
    
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
        output_hidden_states=True,
    )
    
    unscaled_logits = torch.stack(output.logits, dim=0)
    scaled_logits = torch.stack(output.scores, dim=0)
    
    tokens = output.sequences[0][c:]
    prop = output.sequences[0].tolist()

    assert len(tokens) == unscaled_logits.shape[0] == scaled_logits.shape[0]

    idx = tokens.view(unscaled_logits.shape[0], 1, 1)

    log_probs_unnorm = (1/temp * torch.gather(F.log_softmax(unscaled_logits, dim=-1), -1, idx)).view(-1).tolist()
    log_probs_norm = torch.gather(F.log_softmax(scaled_logits, dim=-1), -1, idx).view(-1).tolist()

    h_last_steps = []
    for step_tup in output.hidden_states:
        last_layer_tensor = step_tup[-1]
        last_token_state = last_layer_tensor[:, -1, :]  
        h_last_steps.append(last_token_state)
    
    h_last_stacked = torch.stack(h_last_steps, dim=0)
    h_last = h_last_stacked.squeeze(1).cpu()

    probs = F.softmax(unscaled_logits, dim=-1)
    log_probs = F.log_softmax(unscaled_logits, dim=-1)
    entropy_tensor = -torch.sum(probs * log_probs, dim=-1)
    entropy = entropy_tensor.view(-1).tolist()

    assert len(tokens) == len(log_probs_unnorm) == len(log_probs_norm) == len(entropy) == h_last.size(0)

    return prop, log_probs_norm, log_probs_unnorm, h_last, entropy


# =============================================================================
# ORIGINAL MCMC FUNCTIONS (kept for backward compatibility)
# =============================================================================

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


def mcmc_power_samp(
  p : AutoregressiveSampler, 
  context, 
  temp, 
  mcmc_steps, 
  max_new_tokens, 
  block_num=16
  ):
    """Original random-cut MCMC (kept for backward compatibility and comparison)."""
    c = len(context)
    print(f'alpha: {1/temp}')
    gen = []
    if context is not None:
        gen = context.copy()
    log_probs_norm = []
    log_probs_unnorm = []

    full_entropy = [] 
    full_h_last = torch.tensor([], device='cpu')

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
        
        sims = F.cosine_similarity(h_last[:-1], h_last[1:], dim=-1)
        velocities = (1 - sims).tolist() + [0.0]
        print(f"raw_h_last shape: {h_last.shape}")
        print(f"Entropy length: {len(entropy)}")
        print(f"Velocities length: {len(velocities)}")
        surprisal_scores = compute_surprisal(entropy, velocities)

        for _ in tqdm(range(mcmc_steps)):
            attempts+=1
            t = len(gen)
            idx = random.randint(c, t-1)
            
            prop, log_prob_prop, target_log_prob_prop,prop_h_last, prop_entropy = naive_temp(p, gen[:idx], temp=temp, seq_len=t)
            
            s = len(prop)
            assert(len(log_prob_prop) == s - idx)
            assert(len(target_log_prob_prop) == s - idx)

            log_prob_cur = log_probs_norm.copy()[idx-c:s-c]
            target_log_prob_cur = log_probs_unnorm.copy()[idx-c:s-c]
           
            log_r = sum(target_log_prob_prop) + sum(log_prob_cur) - sum(target_log_prob_cur) - sum(log_prob_prop)
            
            if np.random.rand() < np.exp(log_r):
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


# =============================================================================
# NEW: ADAPTIVE CUT-POINT MCMC WITH PLOTTING
# =============================================================================

def mcmc_power_samp_with_plot(
    p,  # AutoregressiveSampler
    context,
    temp,
    mcmc_steps,
    max_new_tokens,
    block_num=16,
    plot_every=1,
    save_plots=True,
    plot_dir="mcmc_plots",
    use_plotly=True,
    window_size=5,
    peak_distance=10,
    peak_prominence=0.3,
    # NEW: Adaptive cut-point parameters
    use_adaptive_cut=True,
    adaptive_num_bins=10,
    adaptive_delta=0.1,
    adaptive_epsilon=0.1,
    adaptive_refractory_radius=10,
    adaptive_bandit_gamma=0.1,
    adaptive_bandit_eta=0.1,
    adaptive_bandit_decay=0.02,
    verbose=True  # Whether to print detailed logging
):
    """
    MCMC Power Sampling with integrated surprisal timeline visualization
    and ADAPTIVE cut-point selection using bin-bandit with surprisal prior.
    
    NOTE: Window size (L) is DYNAMICALLY determined as L = seq_len - context_len,
    i.e., the entire generated region. The number of bins (B) is fixed,
    and bin_size (K = L/B) adapts automatically to the current window.
    
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
        window_size: Window size for surprisal smoothing (for visualization only)
        peak_distance: Minimum distance between detected peaks
        peak_prominence: Minimum prominence for peak detection
        
        # Adaptive cut-point parameters:
        use_adaptive_cut: Whether to use adaptive cut-point selection (default True)
        adaptive_num_bins: Number of bins B for the bandit (fixed, bin size adapts)
        adaptive_delta: Mix factor for surprisal prior with uniform
        adaptive_epsilon: Probability of forced uniform exploration
        adaptive_refractory_radius: Radius W for anti-repeat masking
        adaptive_bandit_gamma: Exploration rate for EXP3
        adaptive_bandit_eta: Learning rate for EXP3 updates
        adaptive_bandit_decay: Weight decay for non-stationarity
        verbose: Whether to print detailed logging information
    
    Returns:
        gen: Generated token sequence
        log_probs_norm: Normalized log probabilities
        log_probs_unnorm: Unnormalized log probabilities
        acceptance_ratio: Ratio of accepted proposals
    """
    c = len(context)
    
    # Compute window size = jump_size = max_new_tokens // block_num
    jump_size = max_new_tokens // block_num
    
    print(f"\n{'='*70}")
    print(f"[MCMC Power Sampling] Starting...")
    print(f"  Temperature: {temp} (alpha={1/temp})")
    print(f"  MCMC steps per block: {mcmc_steps}")
    print(f"  Blocks: {block_num}")
    print(f"  Max new tokens: {max_new_tokens}")
    print(f"  Jump size per block (= Window size L): {jump_size}")
    print(f"  Adaptive cut-point: {use_adaptive_cut}")
    if use_adaptive_cut:
        base_bin_size = jump_size // adaptive_num_bins
        remainder = jump_size % adaptive_num_bins
        print(f"  Window size (L): {jump_size} = {adaptive_num_bins} bins")
        print(f"  Base bin size (K): {base_bin_size}")
        print(f"  Remainder: {remainder} (added to last bin)")
        print(f"  Bin sizes: first {adaptive_num_bins-1} bins = {base_bin_size} tokens, last bin = {base_bin_size + remainder} tokens")
    print(f"  Verbose logging: {verbose}")
    print(f"{'='*70}\n")
    
    # Create plot directory
    if save_plots:
        os.makedirs(plot_dir, exist_ok=True)
    
    gen = []
    if context is not None:
        gen = context.copy()
    
    log_probs_norm = []
    log_probs_unnorm = []
    
    # ========== FIX Issue 1: Maintain FULL surprisal array for entire generation ==========
    # These arrays track surprisal-related metrics for ALL generated tokens (positions c to t)
    # When MH accepts, we update only the rewritten slice [idx-c : t-c]
    full_entropy = []       # Length = t - c (number of generated tokens)
    full_velocities = []    # Length = t - c
    full_surprisal = []     # Length = t - c (computed from entropy + velocities)
    full_h_last = None      # Hidden states for velocity computation
    # ==================================================================================
    
    print(f"Context length: {c}")
    print(f"Max new tokens: {max_new_tokens}")
    assert max_new_tokens % block_num == 0
    print(f"Jump size per block: {jump_size}")
    
    attempts = 0
    acceptances = 0
    
    # Initialize adaptive cut-point sampler with window_size = jump_size
    if use_adaptive_cut:
        cut_point_sampler = AdaptiveCutPointSampler(
            window_size=jump_size,  # L = max_new_tokens // block_num
            num_bins=adaptive_num_bins,
            delta=adaptive_delta,
            epsilon_explore=adaptive_epsilon,
            refractory_radius=adaptive_refractory_radius,
            bandit_gamma=adaptive_bandit_gamma,
            bandit_eta=adaptive_bandit_eta,
            bandit_decay=adaptive_bandit_decay,
            verbose=verbose
        )
    else:
        cut_point_sampler = None
    
    # 3. OUTER LOOP (The "Writer")
    for block_idx in tqdm(range(block_num), desc="Blocks"):
        if verbose:
            print(f"\n{'='*60}")
            print(f"[Block {block_idx}/{block_num}] Starting generation...")
            print(f"  Current sequence length: {len(gen)}")
            print(f"  Current full_surprisal length: {len(full_surprisal)}")
            print(f"{'='*60}")
        
        # Generate new tokens for this block
        gen, lp_norm, lp_unnorm, h_last, entropy = naive_temp(
            p, gen, temp=temp, seq_len=jump_size + len(gen)
        )
        log_probs_norm.extend(lp_norm)
        log_probs_unnorm.extend(lp_unnorm)
        
        # Compute velocities from hidden states
        if h_last.size(0) > 1:
            sims = F.cosine_similarity(h_last[:-1], h_last[1:], dim=-1)
            velocities = (1 - sims).tolist() + [0.0]
        else:
            velocities = [0.0] * len(entropy)
        
        print(f"raw_h_last shape: {h_last.shape}")
        print(f"Entropy length: {len(entropy)}")
        print(f"Velocities length: {len(velocities)}")
        
        # ========== FIX Issue 1: Extend full arrays with new block data ==========
        full_entropy.extend(entropy)
        full_velocities.extend(velocities)
        
        # Compute surprisal for the new block and extend
        block_surprisal = compute_surprisal(entropy, velocities)
        full_surprisal.extend(block_surprisal)
        
        # Store h_last for potential velocity recomputation (we store the last hidden state)
        # Note: For memory efficiency, we could store only the last hidden state for boundary calculation
        full_h_last = h_last  # We'll use this for boundary velocity computation
        
        if verbose:
            print(f"  Extended full arrays: entropy={len(full_entropy)}, velocities={len(full_velocities)}, surprisal={len(full_surprisal)}")
        # ========================================================================
        
        # Get token texts for labeling
        tokens_text = [p.tokenizer.decode([tid]) for tid in gen]
        
        # 4. INNER LOOP (MCMC Steps)
        for mcmc_idx in tqdm(range(mcmc_steps), desc=f"MCMC (Block {block_idx})", leave=False):
            attempts += 1
            t = len(gen)
            gen_len = t - c  # Number of generated tokens
            
            # ========== CUT-POINT SELECTION ==========
            if use_adaptive_cut and cut_point_sampler is not None:
                # Window covers trailing L tokens: [max(c, t-L), t)
                # L = jump_size (stored in cut_point_sampler.window_size)
                L = cut_point_sampler.window_size
                window_start = max(c, t - L)
                window_size = t - window_start
                
                # ========== FIX Issue 1: Use FULL surprisal array ==========
                # full_surprisal has length = gen_len = t - c
                # Window in generation coordinates: [window_start - c, gen_len)
                # i.e., indices [window_start - c, t - c) in full_surprisal
                if window_size > 0 and len(full_surprisal) > 0:
                    # Extract the exact window slice from full_surprisal
                    window_start_gen = window_start - c  # Convert to generation coordinates
                    window_end_gen = gen_len  # = t - c
                    
                    # Safety check
                    window_start_gen = max(0, window_start_gen)
                    window_end_gen = min(len(full_surprisal), window_end_gen)
                    
                    window_surprisal = np.array(full_surprisal[window_start_gen:window_end_gen])
                    
                    if verbose:
                        print(f"    [Surprisal Window] gen_coords=[{window_start_gen}, {window_end_gen}), "
                              f"length={len(window_surprisal)}, expected={window_size}")
                    
                    # If window_surprisal is shorter than expected (shouldn't happen), pad
                    if len(window_surprisal) < window_size:
                        pad_size = window_size - len(window_surprisal)
                        mean_val = np.mean(window_surprisal) if len(window_surprisal) > 0 else 1.0
                        window_surprisal = np.concatenate([
                            np.ones(pad_size) * mean_val,
                            window_surprisal
                        ])
                        if verbose:
                            print(f"    [WARNING] Padded surprisal from {window_size - pad_size} to {window_size}")
                    
                    idx, sampling_info = cut_point_sampler.sample_cut_point(
                        context_len=c,
                        seq_len=t,
                        surprisal_scores=window_surprisal,
                        mcmc_step=mcmc_idx,
                        block_idx=block_idx
                    )
                # ==============================================================
                else:
                    idx = c
                    sampling_info = {'method': 'edge_case', 'window_start': c, 'sampled_bin_idx': None}
            else:
                # Random cut point (original method)
                idx = random.randint(c, t - 1)
                sampling_info = {'method': 'random', 'idx': idx, 'window_start': c, 'sampled_bin_idx': None}
                if verbose:
                    print(f"\n  [Random Cut] Block={block_idx}, MCMC={mcmc_idx}, idx={idx}")
            
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
            # IMPORTANT FIX: Use log-domain test for numerical stability
            # log(u) < log_r is equivalent to u < exp(log_r) but overflow-safe
            log_u = np.log(np.random.rand())
            accepted = log_u < log_r
            
            if verbose:
                print(f"    MH test: log_u={log_u:.4f}, log_r={log_r:.4f}, accepted={accepted}")
            
            # ========== BANDIT UPDATE (no extra model calls) ==========
            if use_adaptive_cut and cut_point_sampler is not None:
                # CRITICAL FIX: Use sampled_bin_idx directly (not recomputed)
                sampled_bin_idx = sampling_info.get('sampled_bin_idx', None)
                
                # Get the bin probability used for this sample
                p_bin = sampling_info.get('p_bin', None)
                if p_bin is not None:
                    p_bin = np.array(p_bin)
                
                # Pass GLOBAL idx for refractory center update
                if accepted:
                    cut_point_sampler.update_on_accept(idx, log_r, sampled_bin_idx, p_bin)
                else:
                    cut_point_sampler.update_on_reject(idx, log_r, sampled_bin_idx, p_bin)
            elif verbose:
                # Log for random cut method
                if accepted:
                    print(f"    [ACCEPTED] log_r={log_r:.4f}")
                else:
                    print(f"    [REJECTED] log_r={log_r:.4f}")
            
            # ========== PLOTTING SECTION ==========
            if mcmc_idx % plot_every == 0:
                # Use full arrays for visualization (not just current block)
                current_entropy = list(full_entropy)
                current_velocities = list(full_velocities)
                current_surprisal = list(full_surprisal)
                current_tokens = [p.tokenizer.decode([tid]) for tid in gen]
                
                # Get bandit stats if available
                bandit_stats = None
                if use_adaptive_cut and cut_point_sampler is not None:
                    bandit_stats = cut_point_sampler.bandit.get_stats()
                
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
                        peak_prominence=peak_prominence,
                        bandit_stats=bandit_stats,
                        sampling_info=sampling_info
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
                        peak_prominence=peak_prominence,
                        bandit_stats=bandit_stats,
                        sampling_info=sampling_info
                    )
            # ========== END PLOTTING SECTION ==========
            
            if accepted:
                acceptances += 1
                gen = prop.copy()
                log_probs_norm[idx - c:] = log_prob_prop.copy()
                log_probs_unnorm[idx - c:] = target_log_prob_prop.copy()
                
                # ========== FIX Issue 1: Update FULL arrays for rewritten suffix ==========
                # The rewritten suffix covers generation coordinates [idx - c, t - c)
                # We update only this slice in full_entropy, full_velocities, full_surprisal
                
                rewrite_start = idx - c  # Start of rewritten region in generation coordinates
                rewrite_end = len(gen) - c  # End of rewritten region (= new gen_len)
                
                # prop_entropy covers the rewritten suffix (length = t - idx)
                prop_entropy_list = list(prop_entropy)
                
                # Compute velocities for the rewritten suffix from prop_h_last
                # NOTE: prop_h_last has shape [t - idx, hidden_dim] (only the new suffix)
                # The velocity at the boundary (between old token idx-1 and new token idx)
                # cannot be computed without storing all hidden states. We accept this
                # approximation - the first velocity in the suffix is between new tokens 0 and 1.
                if prop_h_last.size(0) > 1:
                    prop_sims = F.cosine_similarity(prop_h_last[:-1], prop_h_last[1:], dim=-1)
                    prop_velocities = (1 - prop_sims).tolist() + [0.0]
                else:
                    prop_velocities = [0.0] * len(prop_entropy_list)
                
                # Compute surprisal for the rewritten suffix using prop_entropy and prop_velocities
                prop_surprisal = compute_surprisal(prop_entropy_list, prop_velocities)
                
                if verbose:
                    print(f"    [MH Accept] Computing new surprisal from prop_h_last and prop_entropy:")
                    print(f"      prop_h_last shape: {prop_h_last.shape}")
                    print(f"      prop_entropy length: {len(prop_entropy_list)}")
                    print(f"      prop_velocities length: {len(prop_velocities)}")
                    print(f"      prop_surprisal length: {len(prop_surprisal)}")
                
                # Verify lengths match
                assert len(prop_entropy_list) == len(prop_velocities) == len(prop_surprisal), \
                    f"Length mismatch: entropy={len(prop_entropy_list)}, vel={len(prop_velocities)}, surp={len(prop_surprisal)}"
                
                # Update the full arrays - replace the rewritten slice
                # full_xxx has length = current gen_len before update
                # We replace [rewrite_start:] with the new suffix data
                full_entropy[rewrite_start:] = prop_entropy_list
                full_velocities[rewrite_start:] = prop_velocities
                full_surprisal[rewrite_start:] = prop_surprisal
                
                if verbose:
                    print(f"    [Full Arrays Updated] Replaced slice [{rewrite_start}:] with suffix of length {len(prop_entropy_list)}")
                    print(f"      full_entropy length: {len(full_entropy)}")
                    print(f"      full_velocities length: {len(full_velocities)}")
                    print(f"      full_surprisal length: {len(full_surprisal)}")
                    # Verify consistency
                    expected_len = len(gen) - c
                    if len(full_surprisal) != expected_len:
                        print(f"      WARNING: Expected length {expected_len}, got {len(full_surprisal)}")
                
                # Also update the local variables for plotting
                entropy = prop_entropy_list
                velocities = prop_velocities
                h_last = prop_h_last
                # ===========================================================================
                
                tokens_text = [p.tokenizer.decode([tid]) for tid in gen]
                
                del prop
                del log_prob_prop
                del target_log_prob_cur
        
        # End of block: apply weight decay for non-stationarity
        if use_adaptive_cut and cut_point_sampler is not None:
            cut_point_sampler.end_block(block_idx=block_idx)
        
        # 9. CHECK FOR EOS
        if p.tokenizer.eos_token_id in gen:
            eos_idx = gen.index(p.tokenizer.eos_token_id)
            gen = gen[:eos_idx + 1]
            log_probs_norm = log_probs_norm[:eos_idx + 1]
            log_probs_unnorm = log_probs_unnorm[:eos_idx + 1]
            acceptance_ratio = acceptances / attempts
            
            # Print final statistics
            if use_adaptive_cut and cut_point_sampler is not None:
                stats = cut_point_sampler.get_stats()
                print(f"\n=== Adaptive Cut-Point Statistics ===")
                print(f"Total samples: {stats['total_samples']}")
                print(f"Exploration samples: {stats['exploration_samples']} ({stats['exploration_rate']:.2%})")
                print(f"Bandit weights: {stats['bandit_stats']['weights']}")
                print(f"Bin acceptance rates: {stats['bandit_stats']['acceptance_rates']}")
            
            return gen, log_probs_norm, log_probs_unnorm, acceptance_ratio
    
    # 10. FINAL RETURN
    acceptance_ratio = acceptances / attempts
    
    # Print final statistics
    if use_adaptive_cut and cut_point_sampler is not None:
        stats = cut_point_sampler.get_stats()
        print(f"\n=== Adaptive Cut-Point Statistics ===")
        print(f"Total samples: {stats['total_samples']}")
        print(f"Exploration samples: {stats['exploration_samples']} ({stats['exploration_rate']:.2%})")
        print(f"Bandit weights: {stats['bandit_stats']['weights']}")
        print(f"Bin acceptance rates: {stats['bandit_stats']['acceptance_rates']}")
    
    return gen, log_probs_norm, log_probs_unnorm, acceptance_ratio


# =============================================================================
# PROMPT FORMATTING (unchanged)
# =============================================================================

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
