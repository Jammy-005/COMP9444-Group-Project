import torch
from collections.abc import Iterable
from kg_embedding_module import kg_embedding_module
from question_embedding_module import question_embedding_module

class answer_selection_module():
    """Answer selection module, which uses learnt embeddings from the KG embedding module
    and the learnt question embedding process from the question embedding module.
    
    Answer selection involves the ComplEx scoring function, where for a fixed topic entity
    string h and question string q, we pass the question triple (h, q, a) into the
    question_embedding_module.predict_proba(), for all possible answer entities a in the knowledge graph.
    We use predict_proba() to give us a number between 0 to 1 for each question triple, that
    we can interpret as the triple's probability of correctness, but this is equivalent to
    question_embedding_module.forward() because the sigmoid function is always increasing. Finally,
    the most likely answer entity is returned.
    """

    def __init__(self, kg_embedding_module: kg_embedding_module, question_embedding_module: question_embedding_module):
        """Initialise the ComplEx embedding model.
        
        Args:
            kg_embedding_module (kg_embedding_module class): the trained embedding model for entities and
                relations, containing the learned embeddings for all entities and relations in the knowledge graph
            question_embedding_module (question_embedding_module class): the trained question embedding module
                containing the learnt 4-layer neural network to convert question strings into complex vector embeddings
        
        Returns:
            None: Initialises the learnt embedding classes/models
        """

        self.kg_embedding_module = kg_embedding_module
        self.question_embedding_module = question_embedding_module

    @torch.inference_mode()
    def select_answer(self, batch: Iterable[tuple[str, str]]) -> list[str]:
        pairs = list(batch)
        if not pairs:
            return []

        kg = self.kg_embedding_module
        qem = self.question_embedding_module
        qem.eval()

        question_to_idx = qem.question_to_idx
        roberta = qem.roberta_question_embeddings

        if question_to_idx is None or roberta is None:
            raise ValueError("Precomputed question embeddings required")

        device = kg.entity_embeddings.weight.device

        # Embed each question exactly once.
        question_indices = torch.tensor(
            [question_to_idx[question] for _, question in pairs],
            dtype=torch.long,
            device=roberta.device,
        )
        question_embeddings = qem.linear_relu_stack(
            roberta[question_indices].to(device)
        )

        # Look up the topic entities.
        topic_indices = torch.tensor(
            [kg.entity_to_idx[topic] for topic, _ in pairs],
            dtype=torch.long,
            device=device,
        )
        heads = kg.entity_embeddings(topic_indices)

        # All possible answer entities.
        tails = kg.entity_embeddings.weight

        h_re, h_im = heads.chunk(2, dim=-1)
        r_re, r_im = question_embeddings.chunk(2, dim=-1)
        t_re, t_im = tails.chunk(2, dim=-1)

        # Same ComplEx scores, calculated using matrix multiplication.
        scores = (
            (h_re * r_re - h_im * r_im) @ t_re.T
            + (h_re * r_im + h_im * r_re) @ t_im.T
        )

        best_indices = scores.argmax(dim=1).tolist()
        entities = list(kg.entity_to_idx)

        return [entities[index] for index in best_indices]

        
