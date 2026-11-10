
from pathlib import Path
from base import KnowledgeGraph, parse_question, load_qa


if __name__ == "__main__":
    kb_path = Path("data/kb.txt")
    if not kb_path.is_file():
        raise FileNotFoundError(f"kb file not found at path: kb_path={kb_path}")
    graph = KnowledgeGraph(kb_path)
    training_data = load_qa(Path("data"), 1, "train")
    