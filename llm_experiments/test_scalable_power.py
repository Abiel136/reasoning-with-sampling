import os
import json
import random
import argparse

import pandas as pd
import numpy as np
from tqdm import tqdm

import torch
import transformers

from scalable_power import AutoregressiveSampler, scalable_power_samp, format_prompt
from constants import *


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_str", action="store", type=str, default="results/", dest="save_str")
    parser.add_argument("--model", action="store", default="qwen_math", type=str,
                        choices=["qwen", "qwen_math", "phi", "tulu", "qwen_math_grpo", "phi_grpo"])
    parser.add_argument("--temperature", action="store", default=0.25, type=float, dest="temperature")
    parser.add_argument("--cot", action="store", type=bool, default=True)
    parser.add_argument("--device", action="store", type=str, dest="device",
                        default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", action="store", type=int, default=0)
    parser.add_argument("--max_tokens", action="store", type=int, default=128)
    # Scalable power sampling params
    parser.add_argument("--M", action="store", type=int, default=3, help="Number of rollouts for xi estimation")
    parser.add_argument("--T", action="store", type=int, default=20, help="Trajectory length (number of new tokens)")
    parser.add_argument("--K", action="store", type=int, default=5, help="Top-K candidates from base model")
    parser.add_argument("--batch_size", action="store", type=int, default=5, help="Mini-batch size for rollouts (reduces peak GPU memory)")
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    model = args.model
    device = args.device
    temp = args.temperature
    max_tokens = args.max_tokens
    M = args.M
    T = args.T
    K = args.K

    save_str = os.path.join(args.save_str, model)
    os.makedirs(save_str, exist_ok=True)

    print(f"Model: {model}")
    print(f"Device: {device}")
    print(f"Temperature (1/alpha): {temp}")
    print(f"M (rollouts): {M}, T (trajectory): {T}, K (top-K): {K}")

    if model == "qwen":
        # model_str = "Qwen/Qwen2.5-7B"
        model_str = "Qwen/Qwen2.5-0.5B-Instruct"
    elif model == "qwen_math":
        model_str = "Qwen/Qwen2.5-0.5B-Instruct"
    elif model == "qwen_math_grpo":
        model_str = "stellalisy/rethink_rlvr_reproduce-ground_truth-qwen2.5_math_7b-lr5e-7-kl0.00-step150"
    elif model == "phi":
        model_str = "microsoft/Phi-3.5-mini-instruct"
    elif model == "tulu":
        model_str = "allenai/Llama-3.1-Tulu-3-8B-DPO"

    # Load dataset
    json_file = "data/test_questions.json"
    dataset = json.load(open(json_file, "r"))
    print(f"Loaded {len(dataset)} test questions")

    # Load model
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_str, trust_remote_code=True)
    hf_model = transformers.AutoModelForCausalLM.from_pretrained(
        model_str, torch_dtype="auto", trust_remote_code=True
    ).to(device)
    print(f"Model stored on: {hf_model.device}")

    autoreg_sampler = AutoregressiveSampler(hf_model, tokenizer, device)
    print("Loaded model")

    results = []

    for problem, data in tqdm(enumerate(dataset), total=len(dataset), desc="Test scalable power sampling"):
        question = data["prompt"]
        answer = data["answer"]
        q_type = data["type"]
        print(f"\n--- Question {problem + 1} ({q_type}) ---")
        print(f"Q: {question}")
        # print(f"Expected: {answer}")

        input_text = format_prompt(question, model, tokenizer, args.cot)
        input_ids = tokenizer.encode(input_text, return_tensors="pt").to(device)
        prefx = [idx.item() for idx in input_ids[0]]

        import time

        # 1. Standard sampling (baseline)
        t0 = time.time()
        std_output = hf_model.generate(
            input_ids, max_new_tokens=max_tokens,
            return_dict_in_generate=True, output_scores=True,
            do_sample=True
        )
        std_time = time.time() - t0
        std_generated_ids = std_output[0][:, len(input_ids[0]):].squeeze().to("cpu")
        std_completion = tokenizer.decode(std_generated_ids, skip_special_tokens=True)
        std_ntokens = std_generated_ids.shape[0] if std_generated_ids.dim() > 0 else 1
        print(f"Standard ({std_ntokens} tokens, {std_time:.2f}s, {std_ntokens/std_time:.1f} tok/s): {std_completion}", flush=True)

        # 2. Naive low-temp sampling (baseline)
        t0 = time.time()
        naive_output = hf_model.generate(
            input_ids, max_new_tokens=max_tokens,
            return_dict_in_generate=True, output_scores=True,
            do_sample=True, temperature=temp
        )
        naive_time = time.time() - t0
        naive_generated_ids = naive_output[0][:, len(input_ids[0]):].squeeze().to("cpu")
        naive_completion = tokenizer.decode(naive_generated_ids, skip_special_tokens=True)
        naive_ntokens = naive_generated_ids.shape[0] if naive_generated_ids.dim() > 0 else 1
        print(f"Naive temp ({naive_ntokens} tokens, {naive_time:.2f}s, {naive_ntokens/naive_time:.1f} tok/s): {naive_completion}", flush=True)

        del std_output, naive_output
        torch.cuda.empty_cache()

        # 3. Scalable power sampling
        scalable_output = scalable_power_samp(autoreg_sampler, prefx, temp, M, T, K, batch_size=args.batch_size)
        scalable_ids = torch.tensor(scalable_output, dtype=torch.long, device=device).to("cpu")
        # Strip the prompt prefix to get only generated tokens
        scalable_generated_ids = scalable_ids[len(prefx):]
        scalable_completion = tokenizer.decode(scalable_generated_ids, skip_special_tokens=True)
        print(f"Scalable power: {scalable_completion}")

        results.append({
            "question": question,
            "type": q_type,
            "correct_answer": answer,
            "std_completion": std_completion,
            "naive_completion": naive_completion,
            "scalable_completion": scalable_completion,
        })

    df = pd.DataFrame(results)
    out_path = os.path.join(
        save_str,
        f"{model}_test_scalable_power_results_{temp}_{args.seed}.csv"
    )
    df.to_csv(out_path, index=False)
    print(f"\nResults saved to {out_path}")
    print(df.to_string())
