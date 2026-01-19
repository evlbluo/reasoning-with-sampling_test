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
    parser.add_argument("--mcmc_steps", action = "store", type = int, default = 10)
    parser.add_argument("--device", action = "store", type = str, dest = "device", default = "cuda" if torch.cuda.is_available() else 'cpu')
    parser.add_argument("--batch_idx", action = "store", type = int, default = 0)
    parser.add_argument("--seed", action = "store", type = int, default = 0)
    
    # When you type this in your terminal:
    # Bash
    # --- python script.py --seed 42
    # The parser sees the flag --seed.
    # Because action="store", it looks at the next thing you typed (42).
    # It takes that 42 and stores it inside the variable args.seed.
    # Why do we need to specify it?
    # Actually, "store" is the default behavior, so the programmer didn't 
    # strictly need to write it there. They likely included it for clarity.
    
    
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
    start = 100*args.batch_idx
    end = 100*(args.batch_idx+1)

    for problem, data in tqdm(enumerate(dataset[start:end]), desc = "Benchmark on MATH"):
        question = data["prompt"]
        print(question)
        answer = data["answer"]

        input_text = format_prompt(question, model, tokenizer, cot)
        input_ids = tokenizer.encode(input_text, return_tensors="pt").to(device)
        prefx = [idx.item() for idx in input_ids[0]]

        naive_temp_output = hf_model.generate(input_ids, max_new_tokens=3072, 
                                return_dict_in_generate=True, output_scores=True, do_sample = True, temperature = temp)
        
        print(tokenizer.decode(naive_temp_output[0][:, len(input_ids[0]):].squeeze().to("cpu"), skip_special_tokens=True))
        print("naive done")
        
        
        std_output = hf_model.generate(input_ids, max_new_tokens=3072, 
                                return_dict_in_generate=True, output_scores=True, do_sample = True)
        
        print(tokenizer.decode(std_output[0][:, len(input_ids[0]):].squeeze().to("cpu"), skip_special_tokens=True))
        print("std done")

        mcmc_power_samp_output, _, _, acceptance_ratio = mcmc_power_samp(autoreg_sampler, prefx, temp, mcmc_steps, max_new_tokens=3072)

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
        })

    
    df = pd.DataFrame(results)
    df.to_csv(os.path.join(save_str, model+"_math_base_power_samp_results_" + str(mcmc_steps) + "_" + str(temp) + "_" + str(args.batch_idx)  + "_" + str(args.seed) + ".csv"), index=False)
    












        













