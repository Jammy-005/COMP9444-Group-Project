import torch
import torch.nn as nn
import random
import copy
from base import Example
from collections.abc import Iterable
from kg_embedding_module import kg_embedding_module
from transformers import RobertaTokenizer, RobertaModel

class question_embedding_module(nn.Module):
    """Question embeddings learnt using the complEx scoring function. 

    Learns complex-valued embeddings for questions
    and computes a compatibility ComplEx score for each (topic, question,
    answer) triple.
    """

    def __init__(
        self,
        kg_embedding_module : kg_embedding_module,
        question_training_data: list[Example],
        question_to_idx: dict[str, int] | None = None,
        roberta_question_embeddings:  torch.Tensor | None = None,
        first_hidden_layer_dim: int = 600,
        second_hidden_layer_dim: int = 450,
        third_hidden_layer_dim: int = 300,
        dropout_rate: float = 0
    ) -> None:
        """Initialise the question embedding model.

        This module embeds a natural language question q to a fixed dimension vector, where the 
        dimensionality is the same as the embeddings of the entities and relations supplied in the
        kg_embedding_module. This model constructs embeddings using a feed-forward neural network that
        first embeds the question q using RoBERTa (Liu et al., 2019) into a 768-dimensional vector. 
        This is then passed through 3 fully connected layers, excluding the input layer, with ReLU
        activation and finally projected onto the complex space C. The model's input is a question triple
        (topic entity, question, answer entity) abbreviated as (h, q, a), its output in self.forward() is a raw ComplEx
        score measuring the likelihood that the question triple is correct (that is, the answer is correct
        for the question). The learnable model parameters are the weights in the linear layers which determine
        question embedding in complex space - the embeddings of the entitites are fixed and provided
        through the kg_complex embedding object parameter.

        Unlike the kg_complex class and its static, provided entities and relations, questions are not static and we
        must be able to embed any question passed into the forward method. The question_embedding_module
        learns the process to produce an embedding for any question, including those it hasn't encountered
        before. However, because the RoBERTa model is fixed, we temporarily store question embeddings for questions
        in self.question_training_data when fitting the model.

        Args:
            kg_embedding_module (kg_embedding_module class): the trained embedding model for entities and relations,
                containing the learned embeddings for all entities and relations in the knowledge graph
            question_training_data (list[Example]): a list of questions, their topic entity, the template and
                possible answers. This tells the class what the right answers are to a certain question, which
                determines the target output for a given question triple.
            question_to_idx (dict[str, int] | None): a dictionary mapping all questions in the question_training_data
                to an integer between 0 and (self.roberta_question_embeddings.shape[0] - 1). This maps a question string
                to the index of its 768-dimensional RoBERTa embedding in the self.roberta_question_embeddings tensor.
            roberta_question_embeddings (torch.Tensor | None): a tensor of size (len(question_training_data) or greater, 768) containing
                the RoBERTa question embeddings for all questions in the training data. This is useful if the question
                embeddings are pre-computed for the entire batch, to avoid running the fixed/frozen RoBERTa model again.
                The number of rows in this tensor can be less than len(question_training_data) if there are duplicate questions
                across multiple examples. If not provided, this class will calculate the embeddings and generate its own
                question_to_idx dictionary.
            first_hidden_layer_dim (int): number of hidden nodes in the second layer (first hidden layer)
            second_hidden_layer_dim (int): number of hidden nodes in the third layer (second hidden layer)
            third_hidden_layer_dim (int): number of hidden nodes in the fourth layer (third hidden layer)
            dropout_rate (int): Randomly switch off nodes with probability dropout_rate
        Returns:
            None
        
        """
        super().__init__()

        self.kg_embedding_module = kg_embedding_module
        self.question_training_data = question_training_data
        self.question_to_idx = question_to_idx
        self.roberta_question_embeddings = roberta_question_embeddings

        # We use double the embedding dimensionality of the knowledge graph entities/relations since it is a 
        # complex vector embedding
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(768, first_hidden_layer_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(first_hidden_layer_dim, second_hidden_layer_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(second_hidden_layer_dim, third_hidden_layer_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(third_hidden_layer_dim, 2 * self.kg_embedding_module.embedding_dim),
        )      

        # We freeze the parameters in the KG embedding module
        for param in self.kg_embedding_module.parameters():
            param.requires_grad = False


    def forward(
        self,
        batch: Iterable[tuple[str, str, str]],
        question_to_idx: dict[str, int] | None = None,
        roberta_question_embeddings: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Compute raw ComplEx scores for a batch of question triples.

        If we want to find the ComplEx score of the question triple (h, q, a),
        then we call model.forward([(h, q, a)]). Here, the batch_size is one.
        
        Args:
            batch (Iterable[tuple[str, str, str]]): A collection of
                (topic entity, question, answer entity) triples represented as strings.
                Accepts a list, tuple, or set of triples. The size of this iterable
                is batch_size.
            question_to_idx (dict[str, int] | None): a dictionary mapping all questions in the batch
                to an integer between 0 and (roberta_question_embeddings.shape[0] - 1). This maps a question string
                to its 768-dimensional RoBERTa embedding in the roberta_question_embeddings tensor.
            roberta_question_embeddings (torch.Tensor | None): A tensor of size (batch_size or greater, 768)
                that contains the RoBERTa question embeddings for at least each question in the batch.
                This is useful if the question embeddings are pre-computed for the entire batch,
                to avoid running the fixed/frozen RoBERTa model again.
            
            If either of the latter two parameters are None, this method will attempt to source them from the
            class' parameters. If at least one of them is still None, or if there is a question in the batch that
            isn't a key in the question_to_idx dictionary, then the method will compute the RoBERTa question embeddings
            and question_to_idx dictionary on its own.
        
        Returns:
            torch.Tensor: Raw compatibility scores with shape
                (batch_size,). Higher scores indicate greater
                compatibility between the topic entity, question and answer entity.
                Scores are unbounded real values.
        
        """
        device = next(self.parameters()).device
        
        batch = list(batch)
        if not batch:
            raise ValueError("Batch cannot be empty")

        topic_entities, questions, answer_entities = map(list, zip(*batch))

        if question_to_idx is None and roberta_question_embeddings is None:
            question_to_idx = self.question_to_idx
            roberta_question_embeddings = self.roberta_question_embeddings
        if (question_to_idx is None
            or roberta_question_embeddings is None
            or any(question not in question_to_idx for question in questions)
            ):
            print("Code should never go here in qem")
            roberta_question_embeddings, question_to_idx = (
                self.convert_questions_roberta(questions, device))
        
        # Look up the row indices for all questions in the batch
        assert roberta_question_embeddings is not None
        assert question_to_idx is not None
        question_indices = [question_to_idx[question] for question in questions]
        # Retrieve the corresponding precomputed RoBERTa embeddings
        roberta_question_embeddings = roberta_question_embeddings[question_indices].to(device)

        # The 768-dimensional embeddings for the questions are passed through three fully connected hidden layers
        # with ReLu activation, eventually reaching a complex embedding with dimensionality equal to the embeddings
        # in the knowledge graph
        question_embeddings = self.linear_relu_stack(roberta_question_embeddings)

        # Now we get the embeddings of the topic and answer entities
        topic_entity_indexed_batch = [
            self.kg_embedding_module.entity_to_idx[topic_entity]
            for topic_entity in topic_entities
        ]    
        topic_entity_indices = torch.tensor(
            topic_entity_indexed_batch, dtype=torch.long, device=device
        )
        answer_entity_indexed_batch = [
            self.kg_embedding_module.entity_to_idx[answer_entity]
            for answer_entity in answer_entities
        ]           
        answer_entity_indices = torch.tensor(
            answer_entity_indexed_batch, dtype=torch.long, device=device
        )
        head = self.kg_embedding_module.entity_embeddings(topic_entity_indices[:])
        tail = self.kg_embedding_module.entity_embeddings(answer_entity_indices[:])

        # Lastly, we compute the ComplEx scoring function
        h_re, h_im = torch.chunk(head, 2, dim=-1)
        r_re, r_im = torch.chunk(question_embeddings, 2, dim=-1)
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
            batch: Iterable[tuple[str, str, str]],
            question_to_idx: dict[str, int] | None = None,
            roberta_question_embeddings: torch.Tensor | None = None
        ) -> torch.Tensor:
            """Compute sigmoid-transformed scores for a batch of question triples.
            
            Args:
                batch (Iterable[tuple[str, str, str]]): A collection of
                    (topic entity, question, answer entity) triples represented as strings.
                    Accepts a list, tuple, or set of triples. The size of this iterable
                    is batch_size.
                question_to_idx (dict[str, int] | None): a dictionary mapping all questions in the batch
                    to an integer between 0 and (roberta_question_embeddings.shape[0] - 1). This maps a question string
                    to its 768-dimensional RoBERTa embedding in the roberta_question_embeddings tensor.
                roberta_question_embeddings (torch.Tensor | None): A tensor of size (batch_size or greater, 768)
                    that contains the RoBERTa question embeddings for at least each question in the batch.
                    This is useful if the question embeddings are pre-computed for the entire batch,
                    to avoid running the fixed/frozen RoBERTa model again.
                            
                If either of the latter two parameters are None, the forward method will attempt to source them from the
                class' parameters. If at least one of them is still None, or if there is a question in the batch that
                isn't a key in the question_to_idx dictionary, then the forward method will compute the RoBERTa question embeddings
                and question_to_idx dictionary on its own.
            
            Returns:
                torch.Tensor: Sigmoid-transformed scores between 0 and 1,
                    with shape (batch_size,). Higher values indicate
                    greater predicted plausibility. These values are
                    not necessarily calibrated probabilities.
            """
            return torch.sigmoid(self.forward(batch, question_to_idx, roberta_question_embeddings))

    def fit(
            self,
            epochs: int = 100,
            batch_size: int = 128,
            learning_rate: float = 0.001,
            weight_decay: float = 0,
            negatives_per_positive: int = 3
        ) -> list[float]:
        """Train question embeddings using the provided list of question examples when initialising this class.
    
        Args:
            epochs (int): Number of training epochs
            batch_size (int): Number of distinct questions per batch
            learning_rate (float): Adam optimizer learning rate
            weight_decay (float): L2 regularization coefficient applied by the Adam optimizer
            negatives_per_positive (int): Number of negative/incorrect question triples
                            sampled for each positive triple.
        Returns:
            list[float]: Average training loss for each epoch.
        """
        print("Beginning question embedding module fitting")
        
        if epochs < 1 or batch_size < 1:
            raise ValueError("Training parameters must be positive")

        question_training_data = copy.deepcopy(self.question_training_data)
        entities = list(self.kg_embedding_module.get_all_entities())
        device = next(self.parameters()).device
        losses = []
        
        criterion = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate, weight_decay=weight_decay)

        for epoch in range(epochs):
            print(f"Beginning epoch {epoch + 1}")
            self.train()
            random.shuffle(question_training_data)
        
            total_loss = 0.0
            total_samples = 0
        
            for start in range(0, len(question_training_data), batch_size):
                # Each example in the question training data can have multiple answers, and we treat each question
                # (topic-question-answer) triple its own sample. Paper 2 does not require a template in the question
                # embedding module, so we leave out this field when constructing a batch.
                batch_examples = question_training_data[start:start + batch_size]
                batch_positive = [
                    (example.topic, example.question, answer)
                    for example in batch_examples
                    for answer in example.answers
                ]
        
                # Generate negative/incorrect question triples by corrupting the answer entity, while keeping head (topic)
                # entity and question the same
                batch_negative = []
                for example in batch_examples:
                    num_negatives = len(example.answers) * negatives_per_positive

                    for _ in range(num_negatives):
                        attempt = 0
                        while True:
                            if attempt > 100:
                                raise ValueError("Could not generate false question triple!")
                            answer = random.choice(entities)
                            attempt += 1
                            if answer not in example.answers:
                                break

                        batch_negative.append(
                            (example.topic, example.question, answer)
                        )
                
                # Combine positive and negative question triples, and do forward pass
                batch = batch_positive + batch_negative

                scores = self.forward(
                    batch,
                    question_to_idx=self.question_to_idx,
                    roberta_question_embeddings=self.roberta_question_embeddings)
        
                # Calculate loss           
                labels = torch.tensor(
                    [1.0] * len(batch_positive)
                    + [0.0] * len(batch_negative),
                    dtype=torch.float32,
                    device=scores.device
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

    @staticmethod
    def convert_questions_roberta( 
            questions: list[str],
            device: torch.device,
            batch_size: int = 32,
            verbose: bool = False
        ) -> tuple[torch.Tensor, dict[str, int]]:
        """Convert a batch of questions into 768-dimensional RoBERTa embeddings.

        Uses a frozen RoBERTa model with mean pooling over non-padding tokens.

        Args:
            questions (list[str]): List of natural language questions of length batch_size.
            device (torch.device): Device on which to perform computation.
            batch_size (int): the number of questions that have their RoBERTa embedding computing
                in each call to the frozen RoBERTa model
            verbose (bool): If true, calling this function will print updates after each batch

        Returns:
            tuple[torch.Tensor, dict[str, int]]: Embeddings of shape
                (batch_size, 768) and a dictionary mapping each question
                to its corresponding tensor row index.
        """
        roberta_model = RobertaModel.from_pretrained('roberta-base').to(device)
        roberta_model.eval()
        tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
        roberta_question_embeddings = torch.empty(
            (len(questions), 768),
            dtype=torch.float32
        )
        question_to_idx = {
            question: idx
            for idx, question in enumerate(questions)
        }
        total_batches = (len(questions) + batch_size - 1) // batch_size

        for start in range(0, len(questions), batch_size):
            batch_questions = questions[start:start + batch_size]
            # First, we tokenize the questions into tokens, then extract the numeric tokenID that
            # corresponds to each token in the vocabulary. Any padded tokens should not be considered
            # by the RobertaModel, so we use an attention_mask which has entries that are either 0 or 1. 
            # input_ids and attention_mask are a (batch_size, 64) dimension tensors. We pad every question
            # to 64 tokens to ensure consistency.
            questions_tokenized = tokenizer(
                batch_questions,
                padding=True,
                truncation=True,
                max_length=64,
                return_tensors="pt"
            )  
            input_ids = questions_tokenized["input_ids"].to(device)
            attention_mask = questions_tokenized["attention_mask"].to(device)
            
            # Then, we pass the input_ids of the tokenized questions into the RoBERTa model, along with the
            # attention_mask. The token sequence length (second dimension) of these two variables must be
            # constant for all questions (first dimension), and in the previous few lines we ensured it was
            # always 64. 
            # 
            # For each token in a question, the final hidden layer of the default RoBERTa model has 768 nodes.
            # The first line below returns a tensor with dimensions (batch_size, sequence_length=64, hidden_size=768),
            # where for each question in the batch, and each integer input token id out of the 64 tokens in the
            # question, the real values at the 768 hidden nodes are given. These 768 real values are the embedding for
            # one token from one question. We then average the embeddings for all non-padded tokens (i.e. where the attention
            # mask is 1), for every question in the batch. This gives a (batch_size, 768) dimensional question_embeddings
            # variable with the 768-dimensional embedding for all questions in the batch.
            with torch.no_grad():
                roberta_last_hidden_states = roberta_model(
                    input_ids,
                    attention_mask=attention_mask
                ).last_hidden_state
                mask = attention_mask.unsqueeze(-1).float()
                question_embeddings = torch.sum(roberta_last_hidden_states * mask, dim=1) / torch.clamp(mask.sum(dim=1), min=1e-9)

            roberta_question_embeddings[start:start + len(batch_questions)] = (
                question_embeddings.cpu()
            )

            if verbose:
                batch_number = start // batch_size + 1
                print(
                    f"RoBERTa batch {batch_number}/{total_batches} | "
                    f"Processed {start + len(batch_questions)}/{len(questions)} questions",
                    flush=True
                )

        return roberta_question_embeddings, question_to_idx