"""Precompute and save RoBERTa embeddings for 1-hop training questions.

Place this script alongside train.py in models/paper2, then run:
    python save_roberta_question_embeddings.py

The cache can be reused across hyperparameter trials because RoBERTa is frozen.
"""

import argparse
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = PROJECT_ROOT / "models"
sys.path.insert(0, str(MODELS_DIR))

from base import load_qa
from question_embedding_module import question_embedding_module


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=PROJECT_ROOT / "data")
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "models" / "paper2" / "robert_question_embeddings" / "1-hop_train",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--no-cuda", action="store_true")
    args = parser.parse_args()

    if args.batch_size < 1:
        parser.error("--batch-size must be positive")

    device = torch.device(
        "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu"
    )

    training = load_qa(args.root, 1, "train")
    unique_questions = list(dict.fromkeys(example.question for example in training))
    print(f"Loaded {len(training):,} examples, {len(unique_questions):,} unique questions")
    print(f"Computing RoBERTa embeddings on {device}...")

    embeddings, question_to_idx = question_embedding_module.convert_questions_roberta(
        questions=unique_questions,
        device=device,
        batch_size=args.batch_size,
        verbose=True,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "roberta_question_embeddings": embeddings.cpu(),
            "question_to_idx": question_to_idx,
        },
        args.output,
    )
    print(f"Saved {embeddings.shape[0]:,} embeddings of dimension "
          f"{embeddings.shape[1]} to: {args.output}")


if __name__ == "__main__":
    main()
