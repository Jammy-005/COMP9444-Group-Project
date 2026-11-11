
from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from base import INVERSE, Example, KnowledgeGraph


class VRN(nn.Module):
    def __init__(self, graph: KnowledgeGraph, train_examples: Sequence[Example], hidden_dim: int = 128) -> None:
        super().__init__()
        self.graph: KnowledgeGraph = graph

        relations = sorted({
            relation.removeprefix(INVERSE)
            for adjacency in graph.edges.values()
            for relation in adjacency
        })
        self.relation_ids: dict[str, int] = {relation: i for i, relation in enumerate(relations)}

        vocabulary = sorted({
            word
            for example in train_examples
            for word in example.template.split()
        })
        self.word_ids: dict[str, int] = {word: i for i, word in enumerate(vocabulary, 1)}

        self.projection = nn.Linear(hidden_dim + len(relations), hidden_dim, bias=False)
        self.question_embeddings = nn.Embedding(len(vocabulary) + 1, hidden_dim)

    def forward(self, example: Example, hops: int) -> tuple[list[str], Tensor]:
        if hops < 0:
            raise ValueError("hops must be non-negative")

        device = self.projection.weight.device

        # f_qt(q)
        token_ids = torch.tensor(
            [self.word_ids.get(w, 0) for w in example.template.split()],
            dtype=torch.long,
            device=device,
        )
        question: Tensor = self.question_embeddings(token_ids).sum(dim=0)

        # g(G_y->y) = 0
        embeddings: dict[str, Tensor] = {
            example.topic: self.projection.weight.new_zeros(
                self.projection.out_features
            )
        }
        layer: list[str] = [example.topic]

        # Equation 6
        for _ in range(hops):
            incoming: dict[str, list[Tensor]] = {}

            for parent in layer:
                for relation, targets in self.graph.edges.get(parent, {}).items():
                    unseen = [child for child in targets if child not in embeddings]
                    if not unseen:
                        continue

                    relation_id = self.relation_ids[relation.removeprefix(INVERSE)]
                    one_hot = F.one_hot(
                        torch.tensor(relation_id, device=device),
                        num_classes=len(self.relation_ids),
                    ).to(self.projection.weight.dtype)

                    message = F.relu(
                        self.projection(torch.cat((embeddings[parent], one_hot)))
                    )

                    for child in unseen:
                        incoming.setdefault(child, []).append(message)

            if not incoming:
                break

            layer = sorted(incoming)
            for entity in layer:
                embeddings[entity] = torch.stack(incoming[entity]).mean(dim=0)

        # Equation 5
        candidates = list(embeddings)
        logits = torch.stack(list(embeddings.values())) @ question
        return candidates, logits

    def loss(self, example: Example, hops: int) -> Tensor | None:
        candidates, logits = self.forward(example, hops)

        answers = [
            i for i, entity in enumerate(candidates)
            if entity in example.answers
        ]
        if not answers:
            return None

        return -F.log_softmax(logits, dim=0)[answers].mean()
