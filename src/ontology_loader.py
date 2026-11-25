
import json
from pathlib import Path


class OntologyLoader:

    def __init__(self, ontology_path):
        self.ontology_path = Path(ontology_path)
        self.data = None

    def load(self):
        """Load the ontology from the JSON-based .ont file."""

        if not self.ontology_path.exists():
            raise FileNotFoundError(
                f"Ontology file not found: {self.ontology_path}"
            )

        with self.ontology_path.open(
            "r", encoding="utf-8"
        ) as file:
            self.data = json.load(file)

        if not isinstance(self.data, dict):
            raise ValueError("Invalid ontology structure.")

        if "nodes" not in self.data or "relations" not in self.data:
            raise ValueError(
                "Ontology must contain nodes and relations."
            )

        return self.data

    def get_concepts(self):
        """Extract ontology nodes."""

        if self.data is None:
            self.load()

        concepts = []

        for node in self.data["nodes"]:
            name = node.get("name", "").strip()

            if name:
                concepts.append({
                    "id": node.get("id"),
                    "name": name,
                    "namespace": node.get("namespace")
                })

        return concepts

    def get_relations(self):
        """Extract ontology relationships."""

        if self.data is None:
            self.load()

        return self.data["relations"]

    def display_summary(self):

        concepts = self.get_concepts()
        relations = self.get_relations()

        print("\n========== ONTOLOGY SUMMARY ==========")
        print(f"File: {self.ontology_path.name}")
        print(f"Number of nodes: {len(concepts)}")
        print(f"Number of relations: {len(relations)}")

        print("\n========== CONCEPTS ==========")

        for concept in concepts:
            print(
                f"[{concept['id']}] {concept['name']}"
            )

        print("\n========== END ==========")


if __name__ == "__main__":

    project_root = Path(__file__).resolve().parent.parent

    ontology_path = (
        project_root
        / "data"
        / "ontology"
        / "ali_knowledge_graph_reorganized_v2.ont"
    )

    ontology = OntologyLoader(ontology_path)

    ontology.load()
    ontology.display_summary()
