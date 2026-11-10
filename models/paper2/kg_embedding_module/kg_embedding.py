import torch
from base import KnowledgeGraph
from pathlib import Path
from models.paper2.kg_embedding_module.kg_complex import kg_complEx

if __name__ == "__main__":
    kb_path = Path("data/kb.txt")
    if not kb_path.is_file():
        raise FileNotFoundError(f"kb file not found at path: kb_path={kb_path}")
    graph = KnowledgeGraph(kb_path)

    complex_model = kg_complEx(kg=graph, embedding_dim=200)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    complex_model = complex_model.to(device)
    losses = complex_model.fit()
