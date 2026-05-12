"""
FaultSense — CWRU Data Augmentation
Reads the train split JSONL and generates 4x more instruction-tuning examples.

Techniques:
  1. 4 different instruction phrasings per example
  2. Small Gaussian noise on feature values (2% of each feature's std)

Input : train_split.json   (the 1840-example train split saved by the notebook)
Output: augmented_train.json (~7,360 examples)

Usage: python augment_cwru.py
"""

import json
import random
import numpy as np

random.seed(42)
np.random.seed(42)

INPUT_FILE     = "train_split.json"
OUTPUT_FILE    = "augmented_train.json"
AUGMENT_FACTOR = 4

FEATURE_NAMES = ["Max", "Min", "Mean", "StdDev", "RMS",
                 "Skewness", "Kurtosis", "CrestFactor", "FormFactor"]

INSTRUCTIONS = [
    "Analyze bearing vibration features and identify fault type and severity",
    "Given these vibration measurements, classify the bearing fault type and severity level",
    "Diagnose the bearing condition based on the following time-domain vibration features",
    "What is the fault type and severity of this bearing based on its vibration signature?",
    "Using the vibration statistics below, determine the bearing fault category",
    "Identify whether this bearing has a fault and classify its type and severity",
    "Based on the extracted vibration features, what fault condition does this bearing show?",
    "Classify the bearing health state from the following sensor measurements",
    "Examine these bearing vibration statistics and provide a fault diagnosis",
    "From the vibration feature values below, detect and classify any bearing fault present",
]


def parse_features(input_str: str) -> list:
    """Extract float values from 'Max=0.35, Min=-0.41, ...' string."""
    vals = []
    for name in FEATURE_NAMES:
        segment = input_str.split(f"{name}=")[1]
        vals.append(float(segment.split(",")[0]))
    return vals


def format_features(vals: list) -> str:
    return ", ".join(f"{name}={val:.4f}" for name, val in zip(FEATURE_NAMES, vals))


def main():
    with open(INPUT_FILE) as f:
        examples = [json.loads(line) for line in f]

    print(f"Loaded {len(examples)} examples from {INPUT_FILE}")

    # Per-feature noise std = 2% of each feature's std across the train split
    all_vals = [parse_features(ex["input"]) for ex in examples]
    arr = np.array(all_vals)
    noise_std = arr.std(axis=0) * 0.02

    augmented = []
    for ex in examples:
        base_vals = parse_features(ex["input"])
        for i in range(AUGMENT_FACTOR):
            instruction = INSTRUCTIONS[i % len(INSTRUCTIONS)]
            if i == 0:
                input_str = ex["input"]          # first variant: no noise (original)
            else:
                noisy = base_vals + np.random.normal(0, noise_std)
                input_str = format_features(noisy)
            augmented.append({
                "instruction": instruction,
                "input":       input_str,
                "output":      ex["output"],
            })

    random.shuffle(augmented)

    with open(OUTPUT_FILE, "w") as f:
        for ex in augmented:
            f.write(json.dumps(ex) + "\n")

    print(f"Augmented examples: {len(augmented)}  ({AUGMENT_FACTOR}x)")
    print(f"Saved to: {OUTPUT_FILE}")

    from collections import Counter
    counts = Counter(ex["output"] for ex in augmented)
    print("\nClass distribution:")
    for label, count in sorted(counts.items()):
        print(f"  {label:<28} {count}")


if __name__ == "__main__":
    main()
