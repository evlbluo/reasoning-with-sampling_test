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
from power_samp_utils import *





if __name__ == "__main__":
    # It creates the tool (parser) that will interpret your terminal commands 
  
    parser = argparse.ArgumentParser()
    # This initializes the "listener." 
    # It creates a container object that will hold all the rules for what commands the script accepts.

    parser.add_argument("--save_str", action = "store", type = str, default = "results/",  dest = "save_str")
    parser.add_argument("--model", action = "store", default = "qwen", type = str, choices = ["qwen", "qwen_math", "phi", "tulu", "qwen_math_grpo", "phi_grpo"])
    parser.add_argument("--temperature", action = "store", default = 0.25, type = float, dest = "temperature")
    parser.add_argument("--dataset", action = "store", default = "MATH", type = str)
    parser.add_argument("--cot", action = "store", type = bool, default = True)
    parser.add_argument("--mcmc_steps", action = "store", type = int, default = 3)
    parser.add_argument("--device", action = "store", type = str, dest = "device", default = "cuda" if torch.cuda.is_available() else 'cpu')
    parser.add_argument("--batch_idx", action = "store", type = int, default = 0)
    parser.add_argument("--seed", action = "store", type = int, default = 0)
    
    # ===== NEW: Adaptive Cut-Point Selection Arguments =====
    parser.add_argument("--use_adaptive_cut", action = "store_true", default = True,
                        help = "Use adaptive cut-point selection (bin-bandit with surprisal prior)")
    parser.add_argument("--no_adaptive_cut", action = "store_false", dest = "use_adaptive_cut",
                        help = "Disable adaptive cut-point selection (use random cuts)")
    parser.add_argument("--adaptive_num_bins", action = "store", type = int, default = 10,
                        help = "Number of bins B for the bin-bandit (fixed, bin size adapts to window)")
    parser.add_argument("--adaptive_delta", action = "store", type = float, default = 0.1,
                        help = "Mix factor for surprisal prior with uniform distribution")
    parser.add_argument("--adaptive_epsilon", action = "store", type = float, default = 0.1,
                        help = "Probability of forced uniform exploration")
    parser.add_argument("--adaptive_refractory_radius", action = "store", type = int, default = 10,
                        help = "Radius W for anti-repeat refractory masking")
    parser.add_argument("--adaptive_bandit_gamma", action = "store", type = float, default = 0.1,
                        help = "Exploration rate gamma for EXP3 bandit")
    parser.add_argument("--adaptive_bandit_eta", action = "store", type = float, default = 0.1,
                        help = "Learning rate eta for EXP3 bandit weight updates")
    parser.add_argument("--adaptive_bandit_decay", action = "store", type = float, default = 0.02,
                        help = "Weight decay lambda for handling non-stationarity")
    parser.add_argument("--verbose", action = "store_true", default = True,
                        help = "Enable verbose logging during MCMC")
    parser.add_argument("--no_verbose", action = "store_false", dest = "verbose",
                        help = "Disable verbose logging")
    # ===== END NEW ARGUMENTS =====
    
    args = parser.parse_args()
    # This line actually executes the parsing. 
    # It looks at what you typed in the terminal, 
    # matches it against the rules above, 
    # Later in the code, the programmer acts on these settings


    # The above code is to used for configuration as in colab:
    # !python llm_experiments/power_samp_math.py \
    # --seed 0 \
    # --batch_idx 0 \
    # --model qwen_math \
    # --save_str "results_7b"

    random.seed(0)


    model = args.model
    device = args.device
    dataset_name = args.dataset
    cot = args.cot
    temp = args.temperature
    mcmc_steps = args.mcmc_steps

    save_str = os.path.join(args.save_str, model)
    os.makedirs(save_str, exist_ok=True)


    print(model)
    print(device)
    print(mcmc_steps)
    print(f"Adaptive cut-point selection: {args.use_adaptive_cut}")

    # Just for the convenience of model selection.
    if model == "qwen":
        model_str = "Qwen/Qwen2.5-7B"
    elif model == "qwen_math":
        model_str = "Qwen/Qwen2.5-Math-7B"
    elif model == "qwen_math_grpo":
        model_str = "stellalisy/rethink_rlvr_reproduce-ground_truth-qwen2.5_math_7b-lr5e-7-kl0.00-step150"
    elif model == "phi":
        model_str = 'microsoft/Phi-3.5-mini-instruct'
    elif model == "tulu":
        model_str = "allenai/Llama-3.1-Tulu-3-8B-DPO"
      

    # Loading question data
    if dataset_name == "MATH":
        json_file = 'data/MATH500.json'
        dataset = json.load(open(json_file, "r"))

    print("dataset done")

    # Loading the Tokenizer
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_str, trust_remote_code = True)
    # Loading the Model
    hf_model = transformers.AutoModelForCausalLM.from_pretrained(model_str, torch_dtype="auto", device_map="auto", trust_remote_code = True).to(device)
    
    autoreg_sampler = AutoregressiveSampler(hf_model, tokenizer, device)

    print("loaded models")
    results = []

    # batch_idx (short for Batch Index) is a variable that acts 
    # like a "Page Number" for your data.It tells the computer: 
    # "Of all the work we need to do, which specific slice (or batch) 
    # should I work on right now?"
    # below is the usage examples:
    # --batch_idx 0: This is the "page number."
    # Script Logic: The script inside power_samp_math.py 
    # will run the calculation we analyzed earlier:
    # start = 100 * 0 → 0
    # end = 100 * (0 + 1) → 100
    # The script will load your dataset and process only the first 100 
    # math problems (Index 0 through 99).
    start = 1*args.batch_idx
    end = 1*(args.batch_idx+1)

    for problem, data in tqdm(enumerate(dataset[start:end]), desc = "Benchmark on MATH"):
      # desc="Benchmark on MATH": This adds the label "Benchmark on MATH" to the left side of the bar 
      # so you know what task is currently running.

      # tqdm: It wraps the loop to display a live progress bar in your terminal.
      # enumerate(...): It loops through the data but also keeps track of the index number.
      # Variables: problem: This variable gets the index number (0, 1, 2...) of the current item.
      # data: This variable gets the actual content: 
      #  {
       # "prompt":"Convert the point $(0,3)$ in rectangular coordinates to polar coordinates.  Enter your answer in the form $(r,\\theta),$ where $r > 0$ and $0 \\le \\theta < 2 \\pi.$",
       # "answer":"\\left( 3, \\frac{\\pi}{2} \\right)",
       # "source":"math",
       # "id":"test/precalculus/807.json"
       #},
      #
        question = data["prompt"]
        print(question)
        answer = data["answer"]

        input_text = format_prompt(question, model, tokenizer, cot)
        # This function is a Prompt Engineer's adapter. 
        # Its job is to take your raw math question and "dress it up" 
        # in the specific format that each different AI model expects to see.
        input_ids = tokenizer.encode(input_text, return_tensors="pt").to(device)
        # "pt" stands for PyTorch.
        # The Hugging Face tokenizer is designed to work with multiple different 
        # AI frameworks. This argument tells the tokenizer: "Don't just give me a 
        # generic list of numbers; give me a PyTorch Tensor because I am about to 
        # feed this into a PyTorch model."

        prefx = [idx.item() for idx in input_ids[0]]
        # This line of code converts your data from a 
        # PyTorch Tensor (GPU) into a standard Python List (CPU).
        # Here is what happens to the data at each step of that 
        #   Step 1: input_ids[0] -> Selects the sequence (removes batch dimension).
        #   Step 2: idx.item()   -> Extracts the actual integer value from GPU memory to CPU.
        # Result: A flat list like [101, 205, 30] stored in RAM. 
        # instead of a [[101, 205, 30]]stored in GPU

        naive_temp_output = hf_model.generate(
          input_ids, 
          max_new_tokens=3072, 
          # Data for Analysis:
          # Instead of just returning text, force the model to return a dictionary containing Metadata.
          # 'output_scores=True' ensures we get the confidence scores (log-probs) for every token it generated,
          # which allows us to mathematically compare how "sure" this model was versus the MCMC method.
          return_dict_in_generate=True,
          # Standard generation just returns the text.
          output_scores=True, 
          # Controlling Creativity:
          # 'do_sample=True' enables the "dice roll" (randomness) instead of always picking the #1 word.
          # the model picks randomly based on probabilities
          # 'temperature=temp' adjusts the probabilities. 
          #   - Low temp (e.g., 0.2) = Conservative/Factual.
          #   - High temp (e.g., 1.2) = Creative/Random (but prone to math errors).
          do_sample = True, 
          temperature = temp
          # Note: The variable temp comes from args.temperature at the top of your script.
          )
        
        print(tokenizer.decode(naive_temp_output[0][:, len(input_ids[0]):].squeeze().to("cpu"), skip_special_tokens=True))
        print("naive done")
        naive_seq = naive_temp_output.sequences[0].detach().to("cpu")   # full: prompt + completion
        print_full_tokens(tokenizer, naive_seq, title="NAIVE (full prompt+completion)")
        # EXPLANATION OF THE LINE ABOVE:
        # 1. THE SLICE (naive_temp_output[0][:, len(input_ids[0]):])
        #    The Problem: The model returns [Original Prompt + New Answer]. We don't want to print the question again.
        #    The Solution: We calculate the length of the input question 'len(input_ids[0])' and slice the tensor
        #                  to start reading from AFTER the question ends.
        #
        # 2. THE MOVE (.squeeze().to("cpu"))
        #    .squeeze(): The model outputs a 2D matrix (Batch x Length). This flattens it into a simple 1D list.
        #    .to("cpu"): The data is currently on the GPU. We cannot print it with standard Python tools,
        #                so we move it back to system RAM.
        #
        # 3. THE DECODING (skip_special_tokens=True)
        #    Translation: Looks up each ID in the dictionary to find the corresponding word.
        #    skip_special_tokens: Hides hidden computer codes like <|endoftext|> so the output is clean.
        
        std_output = hf_model.generate(
          input_ids, max_new_tokens=3072, 
          return_dict_in_generate=True, 
          output_scores=True, 
          do_sample = True
          )
        
        print(tokenizer.decode(std_output[0][:, len(input_ids[0]):].squeeze().to("cpu"), skip_special_tokens=True))
        print("std done")
        std_seq = std_output.sequences[0].detach().to("cpu")            # full: prompt + completion
        print_full_tokens(tokenizer, std_seq, title="STD (full prompt+completion)")
        # naive output: Uses temp (whatever you set in arguments).
        # std output: Uses 1.0 (The "Natural" state of the model).

        # ===== MCMC WITH ADAPTIVE CUT-POINT SELECTION =====
        mcmc_power_samp_output, _, _, acceptance_ratio = mcmc_power_samp_with_plot(
          autoreg_sampler,
          prefx,
          temp,
          mcmc_steps,
          max_new_tokens=3072,
          block_num=16,
          plot_every=1,          # Plot every N MCMC steps
          save_plots=True,
          plot_dir="mcmc_plots",
          use_plotly=True,
          window_size=5,
          peak_distance=10,
          peak_prominence=0.3,
          # Adaptive cut-point parameters (window size is now DYNAMIC)
          use_adaptive_cut=args.use_adaptive_cut,
          adaptive_num_bins=args.adaptive_num_bins,
          adaptive_delta=args.adaptive_delta,
          adaptive_epsilon=args.adaptive_epsilon,
          adaptive_refractory_radius=args.adaptive_refractory_radius,
          adaptive_bandit_gamma=args.adaptive_bandit_gamma,
          adaptive_bandit_eta=args.adaptive_bandit_eta,
          adaptive_bandit_decay=args.adaptive_bandit_decay,
          verbose=args.verbose,
        )

        print(len(std_output))
        print(len(naive_temp_output))
        print(len(mcmc_power_samp_output))
        print(tokenizer.decode(torch.tensor([mcmc_power_samp_output], dtype=torch.long, device=device).squeeze().to("cpu"), skip_special_tokens=True))
        print("mcmc done")

        naive_generated_ids = naive_temp_output[0][:, len(input_ids[0]):].squeeze().to("cpu")
        std_generated_ids = std_output[0][:, len(input_ids[0]):].squeeze().to("cpu")
        mcmc_power_samp_ids = torch.tensor([mcmc_power_samp_output], dtype=torch.long, device=device).squeeze().to("cpu")

        naive_completion = tokenizer.decode(naive_generated_ids, skip_special_tokens=True)
        std_completion = tokenizer.decode(std_generated_ids, skip_special_tokens=True)
        mcmc_completion = tokenizer.decode(mcmc_power_samp_ids, skip_special_tokens=True)

        naive_answer = parse_answer(naive_completion)
        std_answer = parse_answer(std_completion)
        mcmc_answer = parse_answer(mcmc_completion)
        
        print(naive_answer)
        print(std_answer)
        print(mcmc_answer)
        print(question)
        print(answer)
        print(f'Acceptance: {acceptance_ratio}')


        results.append({
            "question": question,
            "correct_answer": answer,
            "naive_completion": naive_completion,
            "naive_answer": naive_answer,
            "std_completion": std_completion,
            "std_answer": std_answer,
            "mcmc_completion": mcmc_completion,
            "mcmc_answer": mcmc_answer,
            "acceptance_ratio": acceptance_ratio,
            "use_adaptive_cut": args.use_adaptive_cut,
        })

    
    # Build filename with adaptive cut info
    adaptive_str = "adaptive" if args.use_adaptive_cut else "random"
    df = pd.DataFrame(results)
    df.to_csv(os.path.join(save_str, model+"_math_base_power_samp_results_" + adaptive_str + "_" + str(mcmc_steps) + "_" + str(temp) + "_" + str(args.batch_idx)  + "_" + str(args.seed) + ".csv"), index=False)