
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
    total_loss = 0.0
    trained = 0

    for index, example in enumerate(examples, 1):
        optimizer.zero_grad()

        loss = model.loss(example, hops)
        if loss is None:
            continue
        loss.backward()     # pyright: ignore [reportUnknownMemberType]
        optimizer.step()

        total_loss += loss.item()
        trained += 1

        if index % 100 == 0 or index == len(examples):
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, index, len(examples),
                100. * index / len(examples), loss.item()
            ))

    return total_loss / trained if trained else float("nan")


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
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--limit", type=int, default = 1000)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--no_cuda", action="store_true", default=False)
    args = parser.parse_args()

    model_type = str(args.model)
    root = Path(args.root)
    hops = int(args.hops)
    epochs = int(args.epochs)
    limit = int(args.limit)
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
    training = load_qa(root, hops, "train")[:limit]
    validation = load_qa(root, hops, "dev")[:limit]


    model = Model(graph, training).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(1, epochs + 1):
        random.shuffle(training)

        train(model, training, optimizer, hops, epoch)
        test(model, validation, hops)

if __name__ == "__main__":
    main()
