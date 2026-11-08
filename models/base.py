from __future__ import annotations

from dataclasses import dataclass
from collections import defaultdict

from pathlib import Path
from typing import Final
import re


INVERSE: Final = "inverse_"
ENTITY_REGEX: Final = re.compile(r"\[([^\]]+)\]")
WORD_REGEX: Final = re.compile(r"[a-z0-9]+")


@dataclass(frozen=True)
class Example:
    question: str
    topic: str
    template: str
    answers: frozenset[str]

class KnowledgeGraph:
    def __init__(self, kb_path: Path) -> None:
        self.edges: dict[str, dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))
        self.triples = 0

        with kb_path.open(encoding="utf-8-sig") as file:
            for line_number, data in enumerate(file, 1):
                data = data.strip()
                if not data:
                    continue

                triple = data.split("|")
                if len(triple) != 3:
                    raise ValueError(f"Expected SOURCE|RELATION|TARGET at {kb_path}:{line_number}")

                source, relation, target = (part.strip() for part in triple)

                self.edges[source][relation].add(target)
                self.edges[target][INVERSE + relation].add(source)
                self.triples += 1

    def follow(self, topic: str, path: tuple[str, ...]) -> set[str]:
        nodes: set[str] = {topic}
        for relation in path:
            next_nodes = set[str]()
            for node in nodes:
                targets = self.edges.get(node, {}).get(relation, ())
                for target in targets:
                    next_nodes.add(target)
            nodes = next_nodes
            if not nodes:
                break
        return nodes

    def all_paths(self, topic: str, hops: int) -> dict[tuple[str, ...], set[str]]:
        states: dict[tuple[str, ...], set[str]] = {(): {topic}}
        for _ in range(hops):
            next_states: dict[tuple[str, ...], set[str]] = defaultdict(set)
            for path, nodes in states.items():
                for node in nodes:
                    for relation, targets in self.edges.get(node, {}).items():
                        next_states[path + (relation,)].update(targets)
            states = next_states
            if not states:
                break
        return states

def parse_question(question: str, answers: str) -> Example:
    match = ENTITY_REGEX.search(question)
    if not match:
        raise ValueError(f"Missing a [topic entity] in question: {question}")
    topic = match.group(1).strip()
    template = ENTITY_REGEX.sub(" ENTITY ", question, count=1).lower()
    template = " ".join(WORD_REGEX.findall(template))
    answer_set = frozenset(answer.strip() for answer in answers.split("|"))
    if not answer_set:
        raise ValueError(f"Question has no answers: {question}")
    return Example(question, topic, template, answer_set)

def load_qa(root: Path, hop: int, split: str) -> list[Example]:
    path = root / f"{hop}-hop" / "vanilla" / f"qa_{split}.txt"
    if not path.is_file():
        raise FileNotFoundError(f"QA file could not be found at path={path}")
    result: list[Example] = []
    with path.open(encoding="utf-8-sig") as file:
        for line_number, data in enumerate(file, 1):
            data = data.strip()
            if not data:
                continue
            try:
                question, answers = data.split("\t", maxsplit=1)
            except ValueError as e:
                raise ValueError(r"Expected QUESTION\tANSWERS at" + f"{path}:{line_number}") from e
            result.append(parse_question(question, answers))
    return result

