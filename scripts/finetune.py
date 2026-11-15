"""Fine-tune Chronos-2 with LoRA on the training years (CPU)."""

import argparse
import time

import numpy as np
from chronos import Chronos2Pipeline

from gridcast.cleaning import clean_load
from gridcast.config import get_settings
from gridcast.data import load_hourly_load

TRAIN_END = "2018-01-01"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-steps", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--name", default="chronos2-lora-trial")
    args = parser.parse_args()

    settings = get_settings()
    df = clean_load(load_hourly_load())
    train = df[df["timestamp"] < TRAIN_END]
    inputs = [group["target"].to_numpy(dtype=np.float32) for _, group in train.groupby("item_id")]

    pipeline = Chronos2Pipeline.from_pretrained(settings.model_id, device_map=settings.device)
    output_dir = settings.artifacts_dir / args.name

    start = time.perf_counter()
    pipeline.fit(
        inputs=inputs,
        prediction_length=settings.prediction_length,
        finetune_mode="lora",
        context_length=settings.context_length,
        learning_rate=args.learning_rate,
        num_steps=args.num_steps,
        batch_size=args.batch_size,
        output_dir=output_dir,
        logging_steps=max(1, args.num_steps // 10),
    )
    elapsed = time.perf_counter() - start

    print(f"{args.num_steps} steps in {elapsed:.0f} s ({elapsed / args.num_steps:.1f} s per step)")
    print(f"Saved to {output_dir}")


if __name__ == "__main__":
    main()
