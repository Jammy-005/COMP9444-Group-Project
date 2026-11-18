
from __future__ import annotations

import argparse
import random
from pathlib import Path
from collections.abc import Sequence

import torch

from base import Example, KnowledgeGraph, load_qa
from paper1 import VRN
from paper3 import PullNet


def train(model: VRN | PullNet, examples: Sequence[Example], optimizer: torch.optim.Optimizer, hops: int, epoch: int) -> float:
    model.train()
    total_loss = torch.zeros((), device=next(model.parameters()).device)
    trained = 0

    for index, example in enumerate(examples, 1):
        optimizer.zero_grad(set_to_none=True)

        loss = model.loss(example, hops)
        if loss is None:
            continue
        loss.backward()     # pyright: ignore [reportUnknownMemberType]
        optimizer.step()

        # Keep running total on device; avoid synchronising CPU/GPU every question.
        total_loss += loss.detach()
        trained += 1

        if index % 100 == 0 or index == len(examples):
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, index, len(examples),
                100. * index / len(examples), loss.item()
            ))

    return float((total_loss / trained).item()) if trained else float("nan")


def test(model: VRN | PullNet, examples: Sequence[Example], hops: int) -> float:
    model.eval()
    test_loss = 0.0
    evaluated = 0
    correct = 0

    with torch.no_grad():
        for example in examples:
            candidates, logits = model(example, hops)
            prediction = candidates[int(logits.argmax().item())]
            correct += int(prediction in example.answers)

            loss = model.validation_loss(example, candidates, logits)
            if loss is not None:
                test_loss += loss.item()
                evaluated += 1


    accuracy = correct / len(examples)
    print('\nValidation set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss / evaluated if evaluated else float("nan"),
        correct, len(examples), 100. * accuracy
    ))
    return accuracy


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default = "paper1")
    parser.add_argument("--root", type=Path, default=Path("data"))
    parser.add_argument("--hops", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--train", type=int, default=None)
    parser.add_argument("--test", type=int, default=None)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--no_cuda", action="store_true", default=False)
    parser.add_argument("--max-training-pulls", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)

    model_type = str(args.model)
    root = Path(args.root)
    hops = int(args.hops)
    epochs = int(args.epochs)
    train_samples = args.train
    test_samples = args.test
    lr = float(args.lr)

    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")

    match model_type:
        case "paper1":
            Model = VRN
        case "paper3":
            Model = PullNet
        case _:
            raise ValueError("provided model is not supported")

    graph = KnowledgeGraph(root / "kb.txt")

    training = load_qa(root, hops, "train")
    if train_samples is not None and train_samples <= len(training):
        training = random.sample(training, train_samples)

    validation = load_qa(root, hops, "dev")
    if test_samples is not None and test_samples <= len(validation):
        validation = random.sample(validation, test_samples)

    model = Model(graph, training).to(device)
    if isinstance(model, PullNet):
        if args.max_training_pulls is not None and args.max_training_pulls <= 0:
            parser.error("--max-training-pulls must be positive")
        model.max_training_pulls = args.max_training_pulls
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(1, epochs + 1):
        random.shuffle(training)

        train(model, training, optimizer, hops, epoch)
        test(model, validation, hops)

if __name__ == "__main__":
    main()
