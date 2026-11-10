import torch
import torch.nn as nn
import random
from base import KnowledgeGraph, Example
from collections.abc import Iterable
from kg_embedding_module.kg_complex import kg_complEx
from transformers import RobertaTokenizer, RobertaModel

def pad_sequence(self, arr, max_len=128):
        num_to_add = max_len - len(arr)
        for _ in range(num_to_add):
            arr.append('<pad>')
        return arr

class question_complEx(nn.Module):
    """Question embeddings learnt using the complEx scoring function. 

    Learns complex-valued embeddings for questions
    and computes a compatibility ComplEx score for each (topic, question,
    answer) triple.
    """

    def __init__(
        self,
        kg_complex: kg_complEx,
        question_training_data: list[Example],
        second_layer_dim: int = 500,
        third_layer_dim: int = 300
    ) -> None:
        """Initialise the question embedding model.

        This module embeds a natural language question q to a fixed dimension vector, where the 
        dimensionality is the same as the embeddings of the entities and relations supplied in the
        kg_complEx model. This model constructs embeddings using a feed-forward neural network that
        first embeds the question q using RoBERTa (Liu et al., 2019) into a 768-dimensional vector. 
        This is then passed through 4 fully connected linear layers with ReLU activation and finally
        projected onto the complex space C. The model's input is a question triple (topic entity,
        question, answer entity) abbreviated as (h, q, a), its output in self.forward() is a raw ComplEx
        score measuring the likelihood that the question triple is correct (that is, the answer is correct
        for the question). The learnable model parameters are the weights in the linear layers which determine
        question embedding in complex space - the embeddings of the entitites are fixed and provided
        through the kg_complex embedding object parameter.

        Unlike the kg_complex class, questions are not static and we do not store embeddings like we do for
        the static, given entities and relations in the knowledge graph. The question_complEx module instead
        learns the process to produce an embedding for any question, including those it hasn't encountered
        before.

        Args:
            kg_complex (kg_complEx embedding class): the trained embedding model for entities and relations,
                containing the learned embeddings for all entities and relations in the knowledge graph
            question_training_data (list[Example]): a list of questions, their topic entity, the template and
                possible answers. This tells the class what the right answers are to a certain question, which
                determines the target output for a given question triple.
            second_layer_dim (int): number of hidden nodes in the second layer (first hidden layer)
            third_layer_dim (int): number of hidden nodes in the third layer (second hidden layer)
        Returns:
            None: initialises
        
        """
        super().__init__()

        self.kg_complex = kg_complex
        self.question_training_data = question_training_data

        self.linear_relu_stack = nn.Sequential(
            nn.Linear(768, second_layer_dim),
            nn.ReLU(),
            nn.Linear(second_layer_dim, third_layer_dim),
            nn.ReLU(),
            nn.Linear(third_layer_dim, self.kg_complex.embedding_dim),
        )

        self.tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
        self.roberta_model = RobertaModel.from_pretrained('roberta-based')


    def forward(
        self,
        batch: Iterable[tuple[str, str, str]]
    ) -> torch.Tensor:
        """Compute raw ComplEx scores for a batch of question triples.

        If we want to find the ComplEx score of the question triple (h, r, t),
        then we call model.forward([(h, r, t)]). Here, the batch_size is one.
        
        Args:
            batch (Iterable[tuple[str, str, str]]): A collection of
                (topic entity, question, answer entity) triples represented as strings.
                Accepts a list, tuple, or set of triples. The size of this iterable
                is batch_size.        
        
        Returns:
            torch.Tensor: Raw compatibility scores with shape
                (batch_size,). Higher scores indicate greater
                compatibility between the topic entity, question and answer entity.
                Scores are unbounded real values.
        
        """
        
        batch = list(batch)
        if not batch:
            raise ValueError("Batch cannot be empty")
        
        device = self.linear_relu_stack.weight.device

        topic_entities, questions, answer_entities = map(list, zip(*batch))

        # Convert the question into a 768-dimensional vector embedding

        # First, we tokenize the questions into tokens, then extract the numeric tokenID that
        # corresponds to each token in the vocabulary. Any padded tokens should not be considered
        # by the RobertaModel, so we use an attention_mask which has entries that are either 0 or 1. 
        # input_ids and attention_mask are a (batch_size, 64) dimension tensors. We pad every question
        # to 64 tokens to ensure consistency. 
        questions_tokenized = self.tokenizer(
            questions,
            padding="max_length",
            truncation=True,
            max_length=64,
            return_tensors="pt"
        )  
        input_ids = questions_tokenized["input_ids"]
        attention_mask = questions_tokenized["attention_mask"]

        # Then, we pass the input_ids of the tokenized questions into the RoBERTa model, along with the
        # attention_mask. The token sequence length (second dimension) of these two variables must be
        # constant for all questions (first dimension), and in the last few lines we ensure it is
        # always 64. 
        # 
        # For each token in a question, the final hidden layer of the default RoBERTa model has 768 nodes.
        # The first line below returns a tensor with dimensions (batch_size, sequence_length=64, hidden_size=768),
        # where for each question in the batch, and each integer input token id out of the 64 tokens in the
        # question, the real values at the 768 hidden nodes are given. These 768 real values are the embedding for
        # one token from one question. We then average the embeddings for all non-padded tokens (where the attention
        # mask is 1), for every question in the batch. This gives a (batch_size, 768) dimensional question_embeddings
        # variable with the 768-dimensional embedding for all questions in the batch.
        roberta_last_hidden_states = self.roberta_model(input_ids, attention_mask=attention_mask).last_hidden_state
        mask = attention_mask.unsqueeze(-1).float()
        question_embeddings = torch.sum(roberta_last_hidden_states * mask, dim=1) / torch.clamp(mask.sum(dim=1), min=1e-9)

                
        
        # h_re, h_im = torch.chunk(head, 2, dim=-1)
        # r_re, r_im = torch.chunk(relation, 2, dim=-1)
        # t_re, t_im = torch.chunk(tail, 2, dim=-1)
        
        # re_score = h_re * r_re - h_im * r_im
        # im_score = h_re * r_im + h_im * r_re
        
        # # In the ComplEx score formula, we use the conjugate of the tail entity's embedding, hence
        # # the "+ im_score * t_im" below, not "- im_score * t_im"
        # score = (
        #     re_score * t_re + im_score * t_im
        # ).sum(dim=-1)
        
        return torch.ones(3, 4)

