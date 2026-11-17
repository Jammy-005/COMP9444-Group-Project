import torch
from collections.abc import Iterable
from kg_embedding_module import kg_embedding_module
from question_embedding_module import question_embedding_module

class answer_selection_module():
    """Answer selection module, which uses learnt embeddings from the KG embedding module
    and the learnt question embedding process from the question embedding module.
    
    Answer selection involves the ComplEx scoring function, where for a fixed topic entity
    string h and question string q, we pass the question triple (h, q, a) into the
    question_embedding_module.forward(), for all possible answer entities a in the knowledge graph.
    (((We use predict_proba() to give us a number between 0 to 1 for each question triple, that
    we can interpret as the triple's probability of correctness, but this is equivalent to
    question_embedding_module.forward() because the sigmoid function is always increasing))). Finally,
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

    def select_answer(self, batch: Iterable[tuple[str, str]]) -> list[str]:
        """Selects the answer for each question in the batch Iterable, using the method described
        at the top of this class

        Args:
            batch (Iterable[tuple[str, str]]): A collection of
                (topic entity, question) tuples represented as strings.
                Accepts a list, tuple, or set of tuples. The size of this iterable
                is batch_size.
                
        Returns:
            list[str]: A list containing the predicted answer entity for each question. The length
                of the returned list is batch_size.
                
        """
        entities = list(self.kg_embedding_module.get_all_entities())
        batch = list(batch)

        # Generate candidates for every question
        candidates = [
            (topic, question, entity)
            for topic, question in batch
            for entity in entities
        ]

        # Predict scores in one batch
        probabilities = self.question_embedding_module.forward(
            candidates
        )

        # Reshape scores into (number of questions, number of entities)
        probabilities = probabilities.reshape(len(batch), len(entities))

        # Find the highest-scoring entity for each question
        best_indices = torch.argmax(probabilities, dim=1).tolist()

        return [entities[index] for index in best_indices]

        
