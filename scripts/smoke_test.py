"""Smoke test: pre-fitted Jacobian lens on Qwen3.5-4B (RTX 4090).

Replicates two readouts from the walkthrough / paper:
  1. "currency used in the country shaped like a boot" -> silent 'Italy' concept
  2. "number of legs on the animal that spins webs"    -> silent 'spider' concept

Usage:
  HF_HOME=~/Code/jlens-experiment/.cache/huggingface \
  ~/Code/jlens-experiment/.venv/bin/python smoke_test.py
"""

import torch
import transformers

import jlens

jlens.configure_logging()

MODEL_NAME = "Qwen/Qwen3.5-4B"
LENS_REPO = "neuronpedia/jacobian-lens"
LENS_REVISION = "qwen-n1000"
LENS_FILE = "qwen3.5-4b/jlens/Salesforce-wikitext/Qwen3.5-4B_jacobian_lens_n1000.pt"

print(f"Loading {MODEL_NAME} (bf16, cuda)...")
hf_model = transformers.AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, dtype=torch.bfloat16
).cuda()
tokenizer = transformers.AutoTokenizer.from_pretrained(MODEL_NAME)
model = jlens.from_hf(hf_model, tokenizer)

print("Loading pre-fitted lens...")
lens = jlens.JacobianLens.from_pretrained(
    LENS_REPO, filename=LENS_FILE, revision=LENS_REVISION
)

layers = [
    model.n_layers // 4,
    model.n_layers // 2,
    model.n_layers // 4 * 3,
    model.n_layers - 2,
]


def top5(logits):
    return [tokenizer.decode([t]) for t in logits.topk(5).indices]


PROMPTS = [
    ("boot-country currency", "Fact: The currency used in the country shaped like a boot is"),
    ("web-spinner legs", "Fact: The number of legs on the animal that spins webs is"),
]

for name, prompt in PROMPTS:
    print(f"\n=== {name}: {prompt!r} (position -2) ===")
    jlens_logits, model_logits, _ = lens.apply(
        model, prompt, layers=layers, positions=[-2]
    )
    logit_lens, _, _ = lens.apply(
        model, prompt, layers=layers, positions=[-2], use_jacobian=False
    )
    for layer in layers:
        print(f"L{layer:>3} logit-lens: {top5(logit_lens[layer][0])}")
        print(f"L{layer:>3} J-lens:     {top5(jlens_logits[layer][0])}")
    print(f"model next-token: {top5(model_logits[0])}")

print(f"\npeak VRAM: {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB")
print("SMOKE TEST COMPLETE")
