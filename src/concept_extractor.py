
import json
from pathlib import Path
from collections import defaultdict


class ConceptExtractor:

    # Main concepts defining the research domain
    DOMAIN_SEEDS = [
        "Fraud Detection",
        "Financial Risk",
        "Financial Security",
        "Credit Risk Assessment",
        "Financial Risk Assessment Task",
        "Financial Security Task",
        "Financial Fraud Detection Workflow",
        "Fraud Detection System",
        "Big Data",
        "Machine Learning",
        "Anomaly Detection",
        "Supervised Learning",
        "Model Explainability",
    ]

    # Generic concepts retained for graph context,
    # but excluded from the searchable vocabulary.
    GENERIC_CONCEPTS = {
        "Task",
        "System",
        "Resource",
        "AI System",
        "Workflow",
        "Function",
    }

    def __init__(self, ontology_path):
        self.ontology_path = Path(ontology_path)
        self.nodes = {}
        self.relations = []
        self.concepts = {}
        self.seed_ids = set()

    def load_ontology(self):
        """Load nodes and relations."""

        with self.ontology_path.open(
            "r", encoding="utf-8"
        ) as file:
            data = json.load(file)

        self.nodes = {
            node["id"]: node
            for node in data["nodes"]
        }

        self.relations = data["relations"]

        print(f"Loaded nodes: {len(self.nodes)}")
        print(f"Loaded relations: {len(self.relations)}")

    def find_seed_concepts(self):
        """Find the main domain concepts."""

        name_to_id = {
            node["name"].casefold(): node_id
            for node_id, node in self.nodes.items()
        }

        missing = []

        for name in self.DOMAIN_SEEDS:
            node_id = name_to_id.get(name.casefold())

            if node_id:
                self.seed_ids.add(node_id)
            else:
                missing.append(name)

        if missing:
            print("Warning: seeds not found:", missing)

        print(f"Main domain concepts found: {len(self.seed_ids)}")

    def explore_relations(self):
        """Explore direct relations of domain concepts."""

        adjacency = defaultdict(list)

        for relation in self.relations:

            source = relation["source_node_id"]
            target = relation["destination_node_id"]

            adjacency[source].append({
                "neighbor_id": target,
                "relation": relation["name"],
                "direction": "outgoing"
            })

            adjacency[target].append({
                "neighbor_id": source,
                "relation": relation["name"],
                "direction": "incoming"
            })

        # Begin with the selected domain concepts.
        selected_ids = set(self.seed_ids)

        for seed_id in self.seed_ids:

            for edge in adjacency[seed_id]:
                selected_ids.add(edge["neighbor_id"])

        # Build the concept dictionary.
        for node_id in selected_ids:

            node = self.nodes[node_id]
            name = node["name"]

            if name in self.GENERIC_CONCEPTS:
                continue

            self.concepts[node_id] = {
                "id": node_id,
                "name": name,
                "type": (
                    "core"
                    if node_id in self.seed_ids
                    else "related"
                ),
                "namespace": node.get("namespace")
            }

        return adjacency

    def get_concept_relations(self):
        """Return relations connecting selected concepts."""

        selected_ids = set(self.concepts.keys())

        relevant_relations = []

        for relation in self.relations:

            source = relation["source_node_id"]
            target = relation["destination_node_id"]

            if source in selected_ids and target in selected_ids:

                relevant_relations.append({
                    "source": self.nodes[source]["name"],
                    "relation": relation["name"],
                    "target": self.nodes[target]["name"]
                })

        return relevant_relations

    def export(self, output_path):

        output_path = Path(output_path)
        output_path.parent.mkdir(
            parents=True, exist_ok=True
        )

        result = {
            "domain": "Financial Risk and Fraud Detection",
            "core_concepts": [
                c for c in self.concepts.values()
                if c["type"] == "core"
            ],
            "related_concepts": [
                c for c in self.concepts.values()
                if c["type"] == "related"
            ],
            "relations": self.get_concept_relations()
        }

        with output_path.open(
            "w", encoding="utf-8"
        ) as file:

            json.dump(
                result,
                file,
                indent=4,
                ensure_ascii=False
            )

        print(f"\nExport completed: {output_path}")

    def display_results(self):

        print("\n========== CORE CONCEPTS ==========")

        for concept in self.concepts.values():

            if concept["type"] == "core":
                print(f"- {concept['name']}")

        print("\n========== RELATED CONCEPTS ==========")

        for concept in self.concepts.values():

            if concept["type"] == "related":
                print(f"- {concept['name']}")

        print("\n========== STATISTICS ==========")
        print(f"Core concepts: {sum(c['type'] == 'core' for c in self.concepts.values())}")
        print(f"Related concepts: {sum(c['type'] == 'related' for c in self.concepts.values())}")
        print(f"Total selected concepts: {len(self.concepts)}")


if __name__ == "__main__":

    root = Path(__file__).resolve().parent.parent

    ontology_path = (
        root / "data" / "ontology"
        / "ali_knowledge_graph_reorganized_v2.ont"
    )

    output_path = (
        root / "data" / "output"
        / "domain_concepts.json"
    )

    extractor = ConceptExtractor(ontology_path)

    extractor.load_ontology()
    extractor.find_seed_concepts()
    extractor.explore_relations()
    extractor.display_results()
    extractor.export(output_path)
