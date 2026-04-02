"""
Example: Qwen3-VL inference with mini-sglang (offline mode).

Usage:
    python examples/qwen3_vl_example.py --model /path/to/Qwen3-VL-2B-Instruct

    # With a custom image:
    python examples/qwen3_vl_example.py --model /path/to/Qwen3-VL-2B-Instruct --image /path/to/image.jpg
"""

import argparse
import tempfile
from pathlib import Path

from minisgl.core import SamplingParams
from minisgl.llm import LLM


def create_test_image() -> str:
    """Create a simple test image and return its path."""
    from PIL import Image, ImageDraw

    img = Image.new("RGB", (320, 240), color=(135, 206, 235))
    draw = ImageDraw.Draw(img)
    # Sun
    draw.ellipse([220, 20, 300, 100], fill=(255, 215, 0))
    # Grass
    draw.rectangle([0, 170, 320, 240], fill=(34, 139, 34))
    # House body
    draw.rectangle([100, 110, 200, 170], fill=(178, 34, 34))
    # Roof
    draw.polygon([(90, 110), (150, 60), (210, 110)], fill=(139, 69, 19))
    # Door
    draw.rectangle([135, 130, 165, 170], fill=(101, 67, 33))

    path = tempfile.mktemp(suffix=".png")
    img.save(path)
    print(f"Created test image: {path}")
    return path


def main():
    parser = argparse.ArgumentParser(description="Qwen3-VL example with mini-sglang")
    parser.add_argument(
        "--model",
        type=str,
        default="/data/models/Qwen/Qwen3-VL-2B-Instruct/",
        help="Path to the Qwen3-VL model",
    )
    parser.add_argument("--image", type=str, default=None, help="Path to an image file")
    parser.add_argument("--max-tokens", type=int, default=128, help="Max output tokens")
    args = parser.parse_args()

    # Prepare image
    image_path = args.image if args.image else create_test_image()
    assert Path(image_path).exists(), f"Image not found: {image_path}"

    # Load model
    print(f"Loading model: {args.model}")
    llm = LLM(args.model)

    sp = SamplingParams(max_tokens=args.max_tokens, temperature=0.0)

    # --- Test 1: VL (image + text) ---
    print("\n=== Test 1: Image description ===")
    results = llm.generate(
        ["Describe what you see in this image."],
        sp,
        images=[[image_path]],
    )
    print(f"Output: {results[0]['text']}")
    print(f"Tokens: {len(results[0]['token_ids'])}")

    # --- Test 2: Text-only (verify VL model handles pure text) ---
    print("\n=== Test 2: Text-only ===")
    results = llm.generate(["What is the capital of France? Answer briefly."], sp)
    print(f"Output: {results[0]['text']}")
    print(f"Tokens: {len(results[0]['token_ids'])}")

    # --- Test 3: Batch with mixed VL and text ---
    print("\n=== Test 3: Batch (VL + text) ===")
    results = llm.generate(
        [
            "What colors do you see in this image?",
            "List three famous landmarks.",
        ],
        sp,
        images=[[image_path], None],
    )
    for i, r in enumerate(results):
        print(f"  [{i}] {r['text'][:120]}")

    print("\nDone.")


if __name__ == "__main__":
    main()
