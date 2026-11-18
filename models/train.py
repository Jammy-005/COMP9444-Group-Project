from __future__ import annotations

import argparse
import random
from collections.abc import Sequence
from pathlib import Path

import torch

from base import Example, KnowledgeGraph, load_qa
from paper1 import VRN
from paper3_1 import PullNet as PullNet1
from paper3_2 import PullNet as PullNet2
from paper3_3 import PullNet as PullNet3


type Model = VRN | PullNet1 | PullNet2 | PullNet3


def train(model: Model, examples: Sequence[Example], optimizer: torch.optim.Optimizer, hops: int, epoch: int) -> None:
    model.train()
    for index, example in enumerate(examples, 1):
        optimizer.zero_grad(set_to_none=True)
        loss = model.loss(example, hops)
        if loss is None:
            continue
        loss.backward()
        optimizer.step()

        if index % 100 == 0 or index == len(examples):
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, index, len(examples), 100. * index / len(examples), loss.item()
            ), flush=True)


def test(model: Model, examples: Sequence[Example], hops: int) -> float:
    model.eval()
    test_loss = 0.0
    evaluated = 0
    correct = 0
    recalled = 0
    with torch.no_grad():
        for example in examples:
            candidates, logits = model(example, hops)
            prediction = candidates[int(logits.argmax().item())]
            correct += int(prediction in example.answers)
            recalled += int(any(entity in example.answers for entity in candidates))
            loss = model.validation_loss(example, candidates, logits)
            if loss is not None:
                test_loss += loss.item()
                evaluated += 1

    accuracy = correct / len(examples)
    recall = recalled / len(examples)
    print('\nValidation set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)'.format(
        test_loss / evaluated if evaluated else float('nan'),
        correct, len(examples), 100. * accuracy
    ))
    print('Candidate recall: {}/{} ({:.2f}%)\n'.format(
        recalled, len(examples), 100. * recall
    ))
    return accuracy


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=('paper1', 'paper3_1', 'paper3_2', 'paper3_3'), default='paper3_1')
    parser.add_argument('--root', type=Path, default=Path('data'))
    parser.add_argument('--hops', type=int, default=3)
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--train', type=int, default=None)
    parser.add_argument('--test', type=int, default=None)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--no_cuda', action='store_true')
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() and not args.no_cuda else 'cpu')
    models: dict[str, type[Model]] = {
        'paper1': VRN,
        'paper3_1': PullNet1,
        'paper3_2': PullNet2,
        'paper3_3': PullNet3,
    }

    graph = KnowledgeGraph(args.root / 'kb.txt')
    training = load_qa(args.root, args.hops, 'train')
    validation = load_qa(args.root, args.hops, 'dev')
    if args.train is not None and 0 < args.train < len(training):
        training = random.sample(training, args.train)
    if args.test is not None and 0 < args.test < len(validation):
        validation = random.sample(validation, args.test)

    model = models[args.model](graph, training).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    print(f'Model: {args.model}; hops: {args.hops}; train: {len(training)}; validation: {len(validation)}; seed: {args.seed}', flush=True)

    for epoch in range(1, args.epochs + 1):
        random.shuffle(training)
        train(model, training, optimizer, args.hops, epoch)
        test(model, validation, args.hops)


if __name__ == '__main__':
    main()
