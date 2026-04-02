"""
Benchmark Qwen3-VL offline throughput on mini-sglang.

Usage:
    python benchmark/offline/bench_qwen3_vl.py [--model PATH] [--num-seqs N] [--max-output-len L]

Metrics reported
  - Prefill (VL): includes vision encoder + language prefill
  - Decode throughput: output tokens / decode time
  - End-to-end throughput: total output tokens / wall time
"""

import argparse
import tempfile
import time
from pathlib import Path
from random import randint, seed

from PIL import Image

from minisgl.core import SamplingParams
from minisgl.llm import LLM


# --------------- helpers ---------------

IMAGE_SIZES = [
    (224, 224),   # small  – 64 image tokens
    (448, 448),   # medium – 196 image tokens
    (672, 672),   # large  – 441 image tokens
]


def make_random_image(width: int, height: int) -> str:
    """Create a random-colored image and return its path."""
    img = Image.new("RGB", (width, height), color=(randint(0, 255), randint(0, 255), randint(0, 255)))
    path = tempfile.mktemp(suffix=".png")
    img.save(path)
    return path


PROMPTS = [
    "Describe this image in detail.",
    "What objects do you see?",
    "What is the main color of this image?",
    "Summarize what is shown.",
    "Is there anything unusual about this image?",
    "What emotions does this image evoke?",
    "Describe the composition of this image.",
    "What is the dominant element in this image?",
]


def print_len_stats(name: str, lengths: list) -> None:
    if not lengths:
        print(f"  {name}: no data")
        return
    arr = sorted(lengths)
    n = len(arr)
    print(
        f"  {name}: count={n}, min={arr[0]}, "
        f"p50={arr[int(0.50 * n)]}, p90={arr[int(0.90 * n)]}, max={arr[-1]}"
    )


# --------------- main ---------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark Qwen3-VL offline throughput")
    parser.add_argument("--model", type=str, default="/data/models/Qwen/Qwen3-VL-2B-Instruct/")
    parser.add_argument("--num-seqs", type=int, default=8, help="Number of VL requests")
    parser.add_argument("--max-output-len", type=int, default=128, help="Max output tokens per request")
    parser.add_argument("--image-size", type=str, default="mix",
                        choices=["small", "medium", "large", "mix"],
                        help="Image size: small(224), medium(448), large(672), or mix")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    seed(args.seed)

    # --- Prepare images and prompts ---
    print("Preparing requests...")
    image_paths = []
    prompts = []
    for i in range(args.num_seqs):
        if args.image_size == "mix":
            w, h = IMAGE_SIZES[i % len(IMAGE_SIZES)]
        else:
            idx = {"small": 0, "medium": 1, "large": 2}[args.image_size]
            w, h = IMAGE_SIZES[idx]
        image_paths.append(make_random_image(w, h))
        prompts.append(PROMPTS[i % len(PROMPTS)])

    sampling_params = [
        SamplingParams(temperature=0.0, max_tokens=args.max_output_len)
        for _ in range(args.num_seqs)
    ]

    # --- Load model ---
    print(f"Loading model: {args.model}")
    llm = LLM(args.model)

    # --- Warmup ---
    print("Warming up...")
    llm.generate(
        ["Warmup."],
        SamplingParams(temperature=0.0, max_tokens=4),
        images=[[image_paths[0]]],
    )

    # --- Benchmark: VL requests ---
    print(f"\n{'='*60}")
    print(f"Benchmark: {args.num_seqs} VL requests, max_output={args.max_output_len}")
    print(f"Image size: {args.image_size}")
    print(f"{'='*60}")

    images = [[p] for p in image_paths]
    t_start = time.time()
    results = llm.generate(prompts, sampling_params, images=images)
    t_total = time.time() - t_start

    output_lens = [len(r["token_ids"]) for r in results]
    total_output_tokens = sum(output_lens)

    print_len_stats("Output length", output_lens)
    print(f"\n  Requests:       {args.num_seqs}")
    print(f"  Total output:   {total_output_tokens} tokens")
    print(f"  Wall time:      {t_total:.2f}s")
    print(f"  Throughput:     {total_output_tokens / t_total:.2f} tok/s")
    print(f"  Avg latency:    {t_total / args.num_seqs * 1000:.1f} ms/req")

    # --- Benchmark: text-only baseline (same model) ---
    print(f"\n{'='*60}")
    print(f"Baseline: {args.num_seqs} text-only requests, max_output={args.max_output_len}")
    print(f"{'='*60}")

    text_prompts = prompts  # same prompts, no images
    t_start = time.time()
    results_text = llm.generate(text_prompts, sampling_params)
    t_text = time.time() - t_start

    output_lens_text = [len(r["token_ids"]) for r in results_text]
    total_text_tokens = sum(output_lens_text)

    print_len_stats("Output length", output_lens_text)
    print(f"\n  Requests:       {args.num_seqs}")
    print(f"  Total output:   {total_text_tokens} tokens")
    print(f"  Wall time:      {t_text:.2f}s")
    print(f"  Throughput:     {total_text_tokens / t_text:.2f} tok/s")
    print(f"  Avg latency:    {t_text / args.num_seqs * 1000:.1f} ms/req")

    # --- Cleanup ---
    for p in image_paths:
        Path(p).unlink(missing_ok=True)

    print(f"\n{'='*60}")
    print("Done.")


if __name__ == "__main__":
    main()
