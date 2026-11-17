"""Small CPU regressions; uses cached synthetic vectors without downloading RoBERTa.

Run: python -m unittest discover -s models/paper2 -p 'test_*.py'
"""

import contextlib
import io
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from base import Example, KnowledgeGraph
from answer_selection_module import answer_selection_module
from kg_embedding_module import kg_embedding_module
from question_embedding_module import question_embedding_module


class InferenceTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "kb.txt"
            path.write_text("a|r|b\na|r|c\n", encoding="utf-8")
            graph = KnowledgeGraph(path)
        self.kg = kg_embedding_module(graph, embedding_dim=3)
        self.examples = [Example("question", "a", "template", frozenset({"c"}))]
        self.model = question_embedding_module(
            self.kg,
            self.examples,
            question_to_idx={"question": 0, "another": 1},
            roberta_question_embeddings=torch.randn(2, 768),
            first_hidden_layer_dim=8,
            second_hidden_layer_dim=7,
            third_hidden_layer_dim=6,
            dropout_rate=0.5,
        )
        self.selector = answer_selection_module(self.kg, self.model)

    def test_matrix_scores_match_triple_scores(self):
        self.model.eval()
        batch = [("a", "question"), ("b", "another")]
        entities = sorted(self.kg.entity_to_idx, key=self.kg.entity_to_idx.__getitem__)
        triples = [(h, q, t) for h, q in batch for t in entities]
        expected = self.model(triples).reshape(len(batch), len(entities))
        original_argmax = torch.argmax
        with patch("answer_selection_module.torch.argmax", wraps=original_argmax) as argmax:
            predictions = self.selector.select_answer(batch)
        actual = argmax.call_args.args[0]
        torch.testing.assert_close(actual, expected)
        self.assertEqual(predictions, [entities[i] for i in expected.argmax(dim=1).tolist()])

    def test_ranks_large_logits_without_sigmoid_ties(self):
        self.model.linear_relu_stack = nn.Linear(768, 6)
        with torch.no_grad():
            self.kg.entity_embeddings.weight.zero_()
            self.kg.entity_embeddings.weight[:, 0] = torch.tensor([1., 2., 3.])
            self.model.linear_relu_stack.weight.zero_()
            self.model.linear_relu_stack.bias.zero_()
            self.model.linear_relu_stack.bias[0] = 20.
        scores = self.model([("a", "question", t) for t in ("a", "b", "c")])
        self.assertEqual(scores.argmax().item(), 2)
        self.assertEqual(torch.sigmoid(scores).argmax().item(), 0)
        self.assertEqual(self.selector.select_answer([("a", "question")]), ["c"])

    def test_inference_is_deterministic_and_keeps_weights_and_gradients(self):
        self.model.train()
        before = {name: value.clone() for name, value in self.model.state_dict().items()}
        gradients = {name: parameter.grad for name, parameter in self.model.named_parameters()}
        batch = [("a", "question"), ("b", "another")]
        with patch.object(self.model.linear_relu_stack, "forward", wraps=self.model.linear_relu_stack.forward) as encoder:
            first = self.selector.select_answer(batch)
            self.assertEqual(encoder.call_args.args[0].shape, (2, 768))
        for _ in range(3):
            self.assertEqual(self.selector.select_answer(batch), first)
        self.assertEqual([self.selector.select_answer([item])[0] for item in batch], first)
        self.assertFalse(self.model.training)
        for name, value in self.model.state_dict().items():
            torch.testing.assert_close(value, before[name], rtol=0, atol=0)
        for name, parameter in self.model.named_parameters():
            self.assertIs(parameter.grad, gradients[name])

    def test_question_training_updates_projection_and_keeps_kg_frozen(self):
        kg_before = {name: value.clone() for name, value in self.kg.state_dict().items()}
        projection_before = [parameter.clone() for parameter in self.model.linear_relu_stack.parameters()]
        with contextlib.redirect_stdout(io.StringIO()):
            self.model.fit(epochs=1, batch_size=1, negatives_per_positive=1)
        for name, value in self.kg.state_dict().items():
            torch.testing.assert_close(value, kg_before[name], rtol=0, atol=0)
        self.assertTrue(all(not parameter.requires_grad for parameter in self.kg.parameters()))
        self.assertTrue(any(
            not torch.equal(before, after)
            for before, after in zip(projection_before, self.model.linear_relu_stack.parameters())
        ))

    def test_empty_batch(self):
        self.assertEqual(self.selector.select_answer([]), [])


if __name__ == "__main__":
    unittest.main()
