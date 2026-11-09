import torch
import torch.nn as nn
import random
from base import KnowledgeGraph
from collections.abc import Iterable

class ComplEx(nn.Module):
    """ComplEx knowledge graph embedding model for triple scoring.

    Learns complex-valued embeddings for entities and relations
    and computes a compatibility score for each (head, relation,
    tail) triple.
    """

    def __init__(
        self,
        kg: KnowledgeGraph,
        embedding_dim: int = 100
    ) -> None:
        """Initialise the ComplEx embedding model.

        Args:
            kg (Knowledge Graph): the knowledge graph for the embeddings
            embedding_dim (int): Number of complex dimensions per embedding.
                Each dimension has a real and imaginary component, so each
                embedding stores 2 * embedding_dim real values.

                For example, embedding_dim=3 represents the complex vector
                [1+23j, -7-4j, 0-6j] as [1, -7, 0, 23, -4, -6],
                where the first half stores real components and the second
                half stores imaginary components.

        Returns:
            None: Initialises entity and relation embedding layers and their string-id mappings.

        
        """
        super().__init__()

        self.kg = kg

        # Map entities and relations to unique integer IDs
        self.entity_to_idx = {
            entity: i for i, entity in enumerate(sorted(kg.get_all_entities()))
        }
        self.relation_to_idx = {
            relation: i for i, relation in enumerate(sorted(kg.get_all_relations()))
        }

        self.entity_embeddings = nn.Embedding(
            len(self.entity_to_idx), 2 * embedding_dim
        )
        self.relation_embeddings = nn.Embedding(
            len(self.relation_to_idx), 2 * embedding_dim
        )

        nn.init.xavier_normal_(self.entity_embeddings.weight)
        nn.init.xavier_normal_(self.relation_embeddings.weight)


    def forward(
        self,
        batch: Iterable[tuple[str, str, str]]
    ) -> torch.Tensor:
        """Compute raw ComplEx scores for a batch of string triples.

        Args:
            batch (Iterable[tuple[str, str, str]]): A collection of
            (head, relation, tail) triples represented as strings.
            Accepts a list, tuple, or set of triples. The size of this iterable
            is batch_size.        

        Returns:
            torch.Tensor: Raw compatibility scores with shape
                (batch_size,). Higher scores indicate greater
                compatibility between the head, relation and tail.
                Scores are unbounded real values.

        If there are 1000 entities and 10 relations, and we want to find the ComplEx score of the string
        triple (s, r, t), then we call model.forward([(s, r, t)]). But we need a way to find the embeddings
        of the two entities s and t, and the relation r. This requires entities and relations to be
        mapped to a unique entity and relation index respectively, where this index is used to lookup the
        embedding - this is mapping is created when initialising the class, and the lookup is conducted in this
        method.
        """
        batch = list(batch)

        if not batch:
            raise ValueError("Batch cannot be empty")

        device = self.entity_embeddings.weight.device

        # Convert the list of string triples to a list of integer triples, which represent indices
        indexed_batch = [
            (
                self.entity_to_idx[h],
                self.relation_to_idx[r],
                self.entity_to_idx[t]
            )
            for h, r, t in batch
        ]

        indices = torch.tensor(
            indexed_batch, dtype=torch.long, device=device
        )

        # Retrieve embeddings
        head = self.entity_embeddings(indices[:, 0])
        relation = self.relation_embeddings(indices[:, 1])
        tail = self.entity_embeddings(indices[:, 2])

        h_re, h_im = torch.chunk(head, 2, dim=-1)
        r_re, r_im = torch.chunk(relation, 2, dim=-1)
        t_re, t_im = torch.chunk(tail, 2, dim=-1)

        re_score = h_re * r_re - h_im * r_im
        im_score = h_re * r_im + h_im * r_re

        # In the ComplEx score formula, we use the conjugate of the tail entity's embedding, hence
        # the "+ im_score * t_im" below, not "- im_score * t_im"
        score = (
            re_score * t_re + im_score * t_im
        ).sum(dim=-1)

        return score


    def predict_proba(
        self,
        batch: Iterable[tuple[str, str, str]]
    ) -> torch.Tensor:
        """Compute sigmoid-transformed scores for string triples.

        Args:
            batch (Iterable[tuple[str, str, str]]): A collection of
                (head, relation, tail) triples represented as strings.
                Accepts a list, tuple, or set of triples. The size of this iterable
                is batch_size.

        Returns:
            torch.Tensor: Sigmoid-transformed scores between 0 and 1,
                with shape (batch_size,). Higher values indicate
                greater predicted plausibility. These values are
                not necessarily calibrated probabilities.
        """
        return torch.sigmoid(
            self.forward(batch)
        )


    def fit(
        self,
        epochs: int = 100,
        batch_size: int = 128,
        learning_rate: float = 0.001,
        negatives_per_positive: int = 1
    ) -> list[float]:
        """Train ComplEx embeddings using knowledge graph triples.

        The model learns embeddings for entities and relations in the KG such that the raw ComplEx
        score for a triple that exists in the KG is more than zero, and the score for a non-existent
        triple is less than zero. Equivalently, when a sigmoid transformation is applied to the raw
        ComplEx score, true triples should have sigmoid scores close to one and false triples should
        have sigmoid scores close to zero.

        The trainable model parameters are the complex embeddings in self.entity_embeddings and
        self.relation_embeddings. The model's input is a string triple (h, r, t), and the lookups for the
        embeddings of each entity/relation use a string to index mapping stored in this class' attributes.
        The target output is one for true triples, and zero for false triples.

        The calculated raw ComplEx score for a triple (h, r, t) is found with the self.forward(h, r, t)
        method, and this score undergoes a sigmoid transformation (as described in the previous paragraph).
        The loss function between the transformed score and the target output is binary cross entropy (BCE)
        loss. The BCEWithLogitsLoss class handles the sigmoid and BCE loss at once. It also computes loss for
        batch training.

        Args:
            epochs (int): Number of training epochs.
            batch_size (int): Number of positive triples per batch.
            learning_rate (float): Adam optimizer learning rate.
            negatives_per_positive (int): Number of negative triples
                sampled for each positive triple.

        Returns:
            list[float]: Average training loss for each epoch.
        """
        if epochs < 1 or batch_size < 1 or negatives_per_positive < 1:
            raise ValueError("Training parameters must be positive")

        entities = sorted(self.kg.get_all_entities())
        positive_triples = list(self.kg.get_all_triples())

        if not positive_triples:
            raise ValueError("No trainable triples found")

        device = self.entity_embeddings.weight.device
        losses = []

        criterion = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)

        for epoch in range(epochs):
            self.train()
            random.shuffle(positive_triples)

            total_loss = 0.0
            total_samples = 0

            for start in range(0, len(positive_triples), batch_size):
                batch_positive = positive_triples[start:start + batch_size]

                # Generate negative triples by corrupting tails, while keeping head and relation the
                # same
                batch_negative = []
                for h, r, _ in batch_positive:
                    for _ in range(negatives_per_positive):
                        new_t = random.choice(entities)
                    
                        # Avoid sampling an existing positive/true triple
                        attempt = 0
                        while self.kg.has_triple((h, r, new_t)):
                            if attempt > 100:
                                raise ValueError("Could not generate false triple!")
                            new_t = random.choice(entities)
                            attempt += 1
                    
                        batch_negative.append((h, r, new_t))

                # Combine positive and negative triples, and do forward pass
                batch = batch_positive + batch_negative
                scores = self.forward(batch)

                # Calculate loss           
                labels = torch.tensor(
                    [1.0] * len(batch_positive)
                    + [0.0] * len(batch_negative),
                    dtype=torch.float32,
                    device=device
                )
                loss = criterion(scores, labels)

                # Backpropagation
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item() * len(batch)
                total_samples += len(batch)

            avg_loss = total_loss / total_samples
            losses.append(avg_loss)

            if (epoch + 1) % 10 == 0:
                print(
                    f"Epoch {epoch + 1}/{epochs}, "
                    f"Loss: {avg_loss:.4f}"
                )

        self.eval()
        return losses