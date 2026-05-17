import os
import pickle
import networkx as nx
from app.utils.logger import get_logger

logger = get_logger("graph_store")

class GraphStore:
    def __init__(self, graph_path: str):
        self.graph_path = graph_path
        if os.path.exists(graph_path):
            with open(graph_path, "rb") as f:
                self.G = pickle.load(f)
            logger.info(f"Graph loaded from {graph_path} ({self.G.number_of_nodes()} nodes)")
        else:
            self.G = nx.DiGraph()
            logger.info("New empty graph created")

    def add_edges(self, triples: list):
        for source, relation, target in triples:
            self.G.add_node(source)
            self.G.add_node(target)
            self.G.add_edge(source, target, relation=relation)

    def get_ego_graph(self, entity: str, radius: int = 2) -> list:
        if entity not in self.G:
            return []
        sub = nx.ego_graph(self.G, entity, radius=radius)
        triples = []
        for u, v, data in sub.edges(data=True):
            triples.append((u, data.get("relation", "linked_to"), v))
        return triples

    def get_relations_for_entities(self, entities: list) -> list:
        all_triples = []
        seen = set()
        for entity in entities:
            for triple in self.get_ego_graph(entity):
                key = (triple[0], triple[1], triple[2])
                if key not in seen:
                    seen.add(key)
                    all_triples.append(triple)
        return all_triples[:10]

    def node_exists(self, node: str) -> bool:
        return node in self.G

    def save(self):
        os.makedirs(os.path.dirname(self.graph_path), exist_ok=True)
        with open(self.graph_path, "wb") as f:
            pickle.dump(self.G, f)
        logger.debug(f"Graph saved ({self.G.number_of_nodes()} nodes, {self.G.number_of_edges()} edges)")