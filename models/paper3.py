
from __future__ import annotations

from collections import deque
from collections.abc import Sequence

import torch
from torch import nn, Tensor
from torch.nn import functional as F

from base import INVERSE, Example, KnowledgeGraph


class PullNet(nn.Module):
    def __init__(self, graph: KnowledgeGraph, train_examples: Sequence[Example], hidden_dim: int = 128) -> None:
        super().__init__()
        self.graph: KnowledgeGraph = graph

        relations = sorted(graph.get_all_relations(include_inverse=False))
        self.relation_ids: dict[str, int] = {relation: i for i, relation in enumerate(relations)}

        vocabulary = sorted({
            word
            for example in train_examples
            for word in example.template.split()
        })
        self.word_ids: dict[str, int] = {word: i for i, word in enumerate(vocabulary, 1)}

        self.word_embeddings = nn.Embedding(len(vocabulary) + 1, hidden_dim)
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        self.relation_embeddings = nn.Embedding(len(relations), hidden_dim)

        self.entity_ids = {entity: i for i, entity in enumerate(sorted(graph.get_all_entities()))}
        self.entity_embeddings = nn.Embedding(len(self.entity_ids), hidden_dim)

        self.fact_update = nn.Linear(3 * hidden_dim, hidden_dim)
        self.entity_update = nn.Linear(3 * hidden_dim, hidden_dim)

        self.pull_classifier = nn.Linear(hidden_dim, 1)
        self.answer_classifier = nn.Linear(hidden_dim, 1)

    # h_q = LSTM(w_1, ..., w_|q|)
    def _encode_question(self, question: Example) -> Tensor:
        device = self.word_embeddings.weight.device
        words = torch.tensor(
            [self.word_ids.get(word, 0) for word in question.template.split()],
            dtype=torch.long, device=device
        ).unsqueeze(0)

        _, (embedding, _) = self.lstm(self.word_embeddings(words))
        question_embedding = embedding[-1, 0]
        return question_embedding

    # pull_facts(v_e, q): retrieve top N_f facts using S(r, q)
    def _pull_facts(self, entity: str, question_embedding: Tensor, num_facts: int) -> list[tuple[str, str, str]]:
        facts: list[tuple[str, str, str]] = []
        relation_ids: list[int] = []

        # v_f = (v_s, r, v_o), where v_s, v_o are entities
        for relation, targets in self.graph.edges.get(entity, {}).items():
            connection = relation.removeprefix(INVERSE)

            for target in targets:
                if relation.startswith(INVERSE):
                    facts.append((target, connection, entity))
                else:
                    facts.append((entity, connection, target))

                relation_ids.append(self.relation_ids[connection])

        if not facts or num_facts <= 0:
            return []

        relations = torch.tensor(relation_ids, dtype=torch.long, device=question_embedding.device)
        relations_embedding = self.relation_embeddings(relations)

        # S(r, q) = sigmoid(h_r^T h_q)
        similarities = torch.sigmoid(relations_embedding @ question_embedding)

        _, indices = torch.topk(similarities, min(num_facts, len(facts)))
        return [facts[int(indices[i])] for i in range(indices.numel())]

    # G_q = (V, E), where V = V_entity U V_fact (KB-only)
    def _graph_cnn(self, entities: set[str], facts: set[tuple[str, str, str]], question_embedding: Tensor) -> dict[str, Tensor]:
        entities_embedding = {
            entity: self.entity_embeddings(
                torch.tensor(self.entity_ids[entity], dtype=torch.long, device=question_embedding.device)
            )
            for entity in entities
        }

        # v_e connects to v_f iff v_e is its subject or object
        incoming: dict[str, list[Tensor]] = {entity: [] for entity in entities}

        for subject, relation, object in facts:
            relation_embedding = self.relation_embeddings(
                torch.tensor(self.relation_ids[relation], dtype=torch.long, device=question_embedding.device)
            )
            fact_embedding = torch.relu(self.fact_update(
                torch.cat((entities_embedding[subject], relation_embedding, entities_embedding[object]))
            ))

            incoming[subject].append(fact_embedding)
            incoming[object].append(fact_embedding)

        # aggregate connected fact embeddings
        updated: dict[str, Tensor] = {}

        for entity in entities:
            if incoming[entity]:
                fact_embedding = torch.stack(incoming[entity]).mean(dim=0)
            else:
                fact_embedding = torch.zeros_like(question_embedding)

            combined = torch.cat((entities_embedding[entity], fact_embedding, question_embedding))
            updated[entity] = torch.relu(self.entity_update(combined))

        return updated

    # classify_pullnodes(G_q, k): select top k entities to expand
    def _classify_pullnodes(self, entity_embeddings: dict[str, Tensor], k: int, epsilon: float) -> list[str]:
        entities = list(entity_embeddings)

        logits = torch.stack([
            self.pull_classifier(entity_embeddings[entity]).squeeze(-1)
            for entity in entities
        ])
        probabilities = torch.sigmoid(logits)

        if self.training:
            indices = torch.nonzero(probabilities > epsilon).flatten()
        else:
            _, indices = torch.topk(probabilities, min(k, len(entities)))
        return [entities[int(indices[i])] for i in range(indices.numel())]

    # classify_answer(G_q): score candidate answer entities
    def _classify_answer(self, entity_embeddings: dict[str, Tensor]) -> tuple[list[str], Tensor]:
        entities = list(entity_embeddings)

        logits = torch.stack([
            self.answer_classifier(entity_embeddings[entity]).squeeze(-1)
            for entity in entities
        ])
        return entities, logits


    def _run(
        self,
        question: Example,
        hops: int,
        k: int = 5,
        num_facts: int = 50,
        epsilon: float = 0.5,
        supervise: bool = False,
    ) -> tuple[list[str], Tensor, Tensor | None]:
        question_embedding = self._encode_question(question)
        shortest = self._shortest_path_entities(question, hops) if supervise else {}

        # 1. Initialise G_q^0
        entities: set[str] = {question.topic}
        facts: set[tuple[str, str, str]] = set()
        expanded: set[str] = set()

        pull_losses: list[Tensor] = []
        fact_losses: list[Tensor] = []

        # 2. For t = 0, ..., T-1
        for t in range(hops):
            entity_embeddings = self._graph_cnn(entities, facts, question_embedding)
            available = {
                entity: embedding
                for entity, embedding in entity_embeddings.items()
                if entity not in expanded
            }

            # 3. Train retrieval classifiers
            if supervise and available:
                pull_losses.append(self._pull_loss(available, shortest, t))

                for entity in available:
                    fact_loss = self._fact_loss(entity, question_embedding, shortest, t)
                    if fact_loss is not None:
                        fact_losses.append(fact_loss)

            # 4. Select entities for expansion
            selected = self._classify_pullnodes(available, k, epsilon) if available else []

            # 5. Pull facts from selected entities
            for entity in selected:
                expanded.add(entity)
                entity_facts = self._pull_facts(entity, question_embedding, num_facts)

                # 6. Extract head/tail and update G_q
                for subject, relation, object in entity_facts:
                    entities.update((subject, object))
                    facts.add((subject, relation, object))

            # 7. Ensure shortest-path facts and entities are present
            if supervise:
                for entity, distance in shortest.items():
                    if distance != t:
                        continue

                    for relation, targets in self.graph.edges.get(entity, {}).items():
                        for target in targets:
                            if shortest.get(target) != t + 1:
                                continue

                            connection = relation.removeprefix(INVERSE)

                            if relation.startswith(INVERSE):
                                facts.add((target, connection, entity))
                            else:
                                facts.add((entity, connection, target))

                            entities.update((entity, target))

            if not selected and not supervise:
                break

        # 8. Classify answers in G_q^T
        entity_embeddings = self._graph_cnn(entities, facts, question_embedding)
        candidates, logits = self._classify_answer(entity_embeddings)

        if not supervise:
            return candidates, logits, None

        if not any(entity in question.answers for entity in candidates):
            return candidates, logits, None

        # 9. Combine training objectives
        total_loss = self._answer_loss(candidates, logits, question)

        if pull_losses:
            total_loss = total_loss + torch.stack(pull_losses).mean()

        if fact_losses:
            total_loss = total_loss + torch.stack(fact_losses).mean()

        return candidates, logits, total_loss

    def forward(self, question: Example, hops: int) -> tuple[list[str], Tensor]:
        candidates, logits, _ = self._run(question, hops)
        return candidates, logits

    def loss(self, question: Example, hops: int) -> Tensor | None:
        _, _, total_loss = self._run(question, hops, supervise=True)
        return total_loss

    def _shortest_path_entities(self, question: Example, hops: int) -> dict[str, int]:
        distances: dict[str, int] = {question.topic: 0}
        parents: dict[str, set[str]] = {question.topic: set()}
        queue: deque[str] = deque([question.topic])

        while queue:
            entity = queue.popleft()
            if distances[entity] >= hops:
                continue

            for targets in self.graph.edges.get(entity, {}).values():
                for target in targets:
                    distance = distances[entity] + 1
                    if target not in distances:
                        distances[target] = distance
                        parents[target] = {entity}
                        queue.append(target)
                    elif distances[target] == distance:
                        parents[target].add(entity)

        shortest: dict[str, int] = {}
        pending = [answer for answer in question.answers if answer in distances]
        while pending:
            entity = pending.pop()
            if entity in shortest:
                continue

            shortest[entity] = distances[entity]
            pending.extend(parents.get(entity, set()))

        return shortest

    def _pull_loss(self, entity_embeddings: dict[str, Tensor], shortest: dict[str, int], t: int) -> Tensor:
        logits: list[Tensor] = []
        labels: list[float] = []

        for entity, embedding in entity_embeddings.items():
            logits.append(self.pull_classifier(embedding).squeeze(-1))

            positive = False
            for targets in self.graph.edges.get(entity, {}).values():
                if any(shortest.get(target) == t + 1 for target in targets):
                    positive = True
                    break

            labels.append(float(positive))

        scores = torch.stack(logits)
        targets = torch.tensor(labels, dtype=scores.dtype, device=scores.device)

        return F.binary_cross_entropy_with_logits(scores, targets)

    def _fact_loss(self, entity: str, question_embedding: Tensor, shortest: dict[str, int], t: int) -> Tensor | None:
        labels: dict[str, float] = {}

        for relation, targets in self.graph.edges.get(entity, {}).items():
            connection = relation.removeprefix(INVERSE)
            positive = any(shortest.get(target) == t + 1 for target in targets)
            labels[connection] = max(labels.get(connection, 0), float(positive))
        if not labels:
            return None

        relation_ids = torch.tensor(
            [self.relation_ids[relation] for relation in labels],
            dtype=torch.long, device=question_embedding.device
        )
        relations_embedding = self.relation_embeddings(relation_ids)
        logits = relations_embedding @ question_embedding

        targets = torch.tensor(list(labels.values()), dtype=logits.dtype, device=logits.device)
        return F.binary_cross_entropy_with_logits(logits, targets)

    def _answer_loss(self, entities: list[str], logits: Tensor, question: Example) -> Tensor:
        labels = torch.tensor(
            [float(entity in question.answers) for entity in entities],
            dtype=logits.dtype, device=logits.device
        )
        return F.binary_cross_entropy_with_logits(logits, labels)

    def validation_loss(self, example: Example, candidates: list[str], logits: Tensor) -> Tensor | None:
        if not any(entity in example.answers for entity in candidates):
            return None
        return self._answer_loss(candidates, logits, example)