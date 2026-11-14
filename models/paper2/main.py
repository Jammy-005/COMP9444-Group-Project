import torch
import argparse
import random
import math
from base import KnowledgeGraph, load_qa
from pathlib import Path
from itertools import product
from kg_embedding_module import kg_embedding_module
from question_embedding_module import question_embedding_module
from answer_selection_module import answer_selection_module

"""
Usage:
    python train.py
    python train.py --root data --hops 1 --epochs 100 --limit 1000
    python train.py --hops 2 --epochs 50 --limit 5000
    python train.py --no_cuda

Description:
    Trains a knowledge graph question-answering model using random-search
    hyperparameter tuning.

    Evaluates different combinations of:
        - KG entity/relation and question complex embedding dimensions
        - Learning rates for both the KG and question embedding modules
        - Batch sizes
        - Negative samples per positive KG triple in the KG embedding module
        - Question encoder hidden layer dimensions
        - Question encoder weight decay
        - Question encoder node dropout

    Selects and prints the best hyperparameters based on validation accuracy.
"""

if __name__ == "__main__":
    # Setup Paper 2 model training and validation
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("data"))
    parser.add_argument("--hops", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--limit", type=int, default = 1000)
    parser.add_argument("--trials", type=int, default=30,
                    help="Number of random hyperparameter configurations to evaluate")
    parser.add_argument("--no_cuda", action="store_true", default=False)
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")

    root = Path(args.root)
    hops = int(args.hops)
    epochs = int(args.epochs)
    limit = int(args.limit)
        
    random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    
    kb_path = Path("data/kb.txt")
    if not kb_path.is_file():
        raise FileNotFoundError(f"kb file not found at path: kb_path={kb_path}")
    graph = KnowledgeGraph(kb_path)
    training = load_qa(root, hops, "train")[:limit]
    validation = load_qa(root, hops, "dev")[:limit]
    test = load_qa(root, hops, "test")[:limit]

    # Hyperparameter search space
    embedding_dims = (20, 200)
    kgem_learning_rates = (1e-5, 1e-2)
    qem_learning_rates = (1e-5, 1e-2)
    batch_sizes = (32, 256)
    negative_samples = (1, 15)
    dropout_rates = (0.0, 0.5)
    weight_decays = (1e-8, 1e-3)

    hidden_dims = [
        (1024, 768, 512),
        (768, 512, 256),
        (768, 550, 400),
        (768, 500, 300),
        (600, 400, 200),
        (512, 256, 128),
        (300, 200, 100)
    ]

    best_accuracy = 0
    best_params = None

    for trial in range(args.trials):
        # Sample hyperparameters from the search space
        embedding_dim = random.randint(*embedding_dims)
        batch_size = random.randint(*batch_sizes)
        negatives = random.randint(*negative_samples)
        kgem_learning_rate = 10 ** random.uniform(
            math.log10(kgem_learning_rates[0]),
            math.log10(kgem_learning_rates[1])
        )
        qem_learning_rate = 10 ** random.uniform(
            math.log10(qem_learning_rates[0]),
            math.log10(qem_learning_rates[1])
        )
        hidden = random.choice(hidden_dims)
        dropout_rate = random.uniform(*dropout_rates)
        weight_decay = 10 ** random.uniform(
            math.log10(weight_decays[0]),
            math.log10(weight_decays[1])
        )

        print(f"\nTrial {trial + 1}/{args.trials}")
        print(
            f"Embedding dimension: {embedding_dim}, "
            f"KG learning rate: {kgem_learning_rate:.6g}, "
            f"Question learning rate: {qem_learning_rate:.6g}, "
            f"Batch size: {batch_size}, "
            f"Negatives per positive KG triple: {negatives}, "
            f"Hidden layer dimensions in question embedding module: {hidden}",
            f"Dropout: {dropout_rate:.3f}, "
            f"Weight decay: {weight_decay:.6g}"
        )

        # Train KG embedding module
        kg_model = kg_embedding_module(
            kg=graph,
            embedding_dim=embedding_dim
        ).to(device)

        kg_model.fit(
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=kgem_learning_rate,
            negatives_per_positive=negatives,
        )

        # Train question embedding module
        question_model = question_embedding_module(
            kg_embedding_module=kg_model,
            question_training_data=training,
            first_hidden_layer_dim=hidden[0],
            second_hidden_layer_dim=hidden[1],
            third_hidden_layer_dim=hidden[2],
            dropout_rate=dropout_rate
        ).to(device)

        question_model.fit(
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=qem_learning_rate,
            weight_decay=weight_decay
        )

        # Validation, using the answer selection module
        answer_selector = answer_selection_module(
            kg_model,
            question_model
        )

        correct = 0
        for example in validation:
            prediction = answer_selector.select_answer(
                [(example.topic, example.question)]
            )[0]

            # Any correct answer entity in the example's set of possible answers will do
            correct += int(prediction in example.answers)

        accuracy = correct / len(validation)

        print(f"Validation accuracy: {accuracy:.4f}")

        if accuracy > best_accuracy:
            best_accuracy = accuracy

            best_params = {
                "embedding_dimension": embedding_dim,
                "kg embedding module learning_rate": kgem_learning_rate,
                "question embedding module learning_rate": qem_learning_rate,
                "batch_size": batch_size,
                "negatives per positive KG triple": negatives,
                "question embedding module hidden_dimensions": hidden,
                "dropout_rate": dropout_rate,
                "weight_decay": weight_decay
            }

    print("\nBest hyperparameters:")
    print(best_params)
    print(f"Best validation accuracy: {best_accuracy:.4f}")