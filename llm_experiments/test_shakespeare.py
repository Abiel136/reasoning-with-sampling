"""
Test batched_scalable_power_samp with the trained Shakespeare GPT-2 model.

Usage:
    python test_shakespeare.py --model_path /path/to/hf-shakespeare-bpe
"""

import argparse
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

from batched_scalable import AutoregressiveSampler, batched_scalable_power_samp


def main():
    parser = argparse.ArgumentParser()
    # parser.add_argument("--model_path", type=str, required=True,
    #                     help="Path to the HuggingFace Shakespeare model directory")
    parser.add_argument("--prompt", type=str,
                        default="To be or not to be, that is the",
                        help="Text prompt to continue")
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else  "cpu")) 
    parser.add_argument("--temp", type=float, default=0.25,
                        help="1/alpha — lower means sharper power distribution")
    parser.add_argument("--T", type=int, default=40,
                        help="Total number of new tokens to generate")
    parser.add_argument("--M", type=int, default=3,
                        help="Rollouts per candidate block for xi estimation")
    parser.add_argument("--K", type=int, default=4,
                        help="Top-K candidate blocks to evaluate")
    parser.add_argument("--L", type=int, default=8,
                        help="Candidate blocks sampled from base model")
    parser.add_argument("--block_size", type=int, default=5,
                        help="Tokens per block")
    parser.add_argument("--H", type=int, default=10,
                        help="Rollout horizon length")
    parser.add_argument("--debug", type=str, default=None,
                        choices=["verbose", "probs"],
                        help="verbose: timing/memory info | probs: per-block probabilities")
    args = parser.parse_args()


    model_path = "/Users/abieltalwar/Documents/Code/LLM/reasoning-with-sampling/llm_experiments/shakespeare"
    print(f"Loading model from {model_path}...")
    tokenizer = GPT2Tokenizer.from_pretrained(model_path)
    model = GPT2LMHeadModel.from_pretrained(model_path).to(args.device)
    model.eval()
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model loaded on {args.device} ({num_params/1e6:.1f}M parameters)")

    sampler = AutoregressiveSampler(model, tokenizer, args.device)

    # Encode the prompt into token IDs
    prompt_ids = tokenizer.encode(args.prompt)
    print(f"\nPrompt: {args.prompt}")
    print(f"Prompt tokens: {len(prompt_ids)}")

    # Run batched scalable power sampling
    print(f"\nRunning batched_scalable_power_samp (T={args.T}, M={args.M}, K={args.K}, L={args.L}, block_size={args.block_size})...")
    output_ids = batched_scalable_power_samp(
        p=sampler,
        prompt=prompt_ids,
        temp=args.temp,
        M=args.M,
        T=args.T,
        K=args.K,
        L=args.L,
        block_size=args.block_size,
        H=args.H,
        batch_size_xi=0,
        debug=args.debug,
    )

    # Decode only the newly generated tokens (strip the prompt)
    generated_ids = output_ids[len(prompt_ids):]
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)

    print(f"\n--- Output ---")
    print(f"{args.prompt}{generated_text}")


if __name__ == "__main__":
    main()
