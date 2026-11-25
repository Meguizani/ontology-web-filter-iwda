
import json
import re
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent

ONTOLOGY_PATH = (
    ROOT_DIR
    / "data"
    / "ontology"
    / "ali_knowledge_graph_reorganized_v2.ont"
)

# Selected ontology node IDs, grouped by research dimension.
CONCEPT_GROUPS = {
    "fraud": ["24", "82", "113", "134", "150", "156", "224"],

    "financial_risk": [
        "71", "73", "74", "151", "159", "212", "219", "220"
    ],

    "machine_learning": [
        "80", "83", "158", "182", "218"
    ],

    "algorithms": ["94", "98", "100", "104", "109"],

    "big_data": ["107", "112", "137", "217"],

    "data_preparation": ["105", "227", "232"],

    "explainability": ["106", "111", "223"],

    "datasets": ["11", "13", "127", "134"],
}

# Aliases of existing ontology concepts.
# These expressions do not create new ontology nodes.
ALIASES = {
    "24": [
        "financial fraud detection",
        "payment fraud detection",
        "transaction fraud detection",
        "fraud monitoring",
        "fraud detection"
    ],

    "73": [
        "credit risk analysis",
        "credit risk assessment"
    ],

    "219": [
        "financial risk management",
        "financial risk assessment",
        "financial risk"
    ],

    "218": [
        "machine learning algorithms",
        "machine learning techniques",
        "machine learning"
    ],

    "217": [
        "large-scale data processing",
        "big data analytics",
        "big data"
    ],

    "94": ["random forest algorithm", "random forest"],
    "98": ["extreme gradient boosting", "xgboost"],
    "100": ["logistic regression"],
    "104": ["isolation forest"],
    "107": ["apache spark", "spark framework"],
    "137": ["pyspark"],
    "105": [
        "synthetic minority oversampling technique",
        "smote"
    ],
    "106": [
        "shap values",
        "shapley additive explanations",
        "shap"
    ],
    "227": ["data cleaning"],
    "232": ["feature engineering"],
    "223": ["model interpretability", "model explainability"],
}

# Contextual expressions that are NOT ontology concepts.
# They are used as additional evidence for document analysis.
CONTEXTUAL_PATTERNS = {
    "financial_domain": [
        "financial sector",
        "financial services",
        "financial institutions",
        "financial institution",
        "banking sector",
        "banking industry",
        "banking system",
        "banks",
        "bank",
        "payment systems",
        "payment system",
        "financial transactions",
        "financial markets",
        "financial stability",
        "credit card",
        "payment services"
    ],

    "cybersecurity": [
        "cybersecurity",
        "cyber security",
        "cyber events",
        "cyber event",
        "cyber incidents",
        "cyber incident",
        "cyber threats",
        "cyber threat",
        "cyber attacks",
        "cyber attack",
        "digital security",
        "information security",
        "network security",
        "security breaches",
        "security breach"
    ],

    "digital_fraud": [
        "digital fraud",
        "online fraud",
        "electronic fraud",
        "payment fraud",
        "card fraud",
        "identity fraud",
        "fraudulent transactions",
        "fraudulent payment",
        "financial crime",
        "financial crimes"
    ],

    "artificial_intelligence_context": [
        "artificial intelligence",
        "AI systems",
        "AI technology",
        "AI applications",
        "foundation model",
        "foundation models",
        "deep learning",
        "predictive analytics",
        "automated decision making"
    ]
}

class OntologyMatcher:

    def __init__(self, ontology_path=ONTOLOGY_PATH):

        self.ontology_path = Path(ontology_path)

        with open(self.ontology_path, "r", encoding="utf-8") as file:
            self.ontology = json.load(file)

        self.nodes = {
            str(node["id"]): node
            for node in self.ontology.get("nodes", [])
        }

        self.relations = self.ontology.get("relations", [])

        self.concepts = self._build_catalogue()

        print(f"Ontology nodes loaded: {len(self.nodes)}")
        print(f"Ontology relations loaded: {len(self.relations)}")
        print(f"Selected concepts: {len(self.concepts)}")
        print(
            "Contextual categories loaded:",
            len(CONTEXTUAL_PATTERNS)
        )

    def _build_catalogue(self):

        catalogue = {}

        for category, node_ids in CONCEPT_GROUPS.items():

            for node_id in node_ids:

                node = self.nodes.get(node_id)

                if not node:
                    print(
                        f"WARNING: Ontology node {node_id} not found"
                    )
                    continue

                if node_id not in catalogue:

                    catalogue[node_id] = {
                        "id": node_id,
                        "name": node["name"],
                        "categories": [],
                        "aliases": set()
                    }

                if category not in catalogue[node_id]["categories"]:
                    catalogue[node_id]["categories"].append(category)

        for node_id, aliases in ALIASES.items():

            if node_id in catalogue:
                catalogue[node_id]["aliases"].update(aliases)

        for concept in catalogue.values():
            concept["aliases"] = sorted(concept["aliases"])

        return catalogue

    @staticmethod
    def _find_phrase(text, phrase):
        """
        Match a complete phrase, ignoring case and
        allowing variable whitespace between words.
        """

        words = phrase.strip().split()

        if not words:
            return False

        pattern = (
            r"(?<!\w)"
            + r"\s+".join(re.escape(word) for word in words)
            + r"(?!\w)"
        )

        return (
            re.search(pattern, text, flags=re.IGNORECASE)
            is not None
        )

    def match_text(self, text):
        """
        Extract concepts that actually exist in the ontology.
        """

        if not isinstance(text, str) or not text.strip():
            return []

        results = []

        for node_id, concept in self.concepts.items():

            expressions = [concept["name"]]
            expressions.extend(concept["aliases"])

            evidence = sorted({
                expression
                for expression in expressions
                if self._find_phrase(text, expression)
            })

            if evidence:

                results.append({
                    "id": node_id,
                    "name": concept["name"],
                    "categories": concept["categories"],
                    "evidence": evidence,
                    "source": "ontology"
                })

        return results

    def match_context(self, text):
        """
        Extract contextual indicators.

        These indicators are not ontology nodes and must
        not be assigned invented ontology identifiers.
        """

        if not isinstance(text, str) or not text.strip():
            return []

        results = []

        for category, expressions in CONTEXTUAL_PATTERNS.items():

            evidence = sorted({
                expression
                for expression in expressions
                if self._find_phrase(text, expression)
            })

            if evidence:

                results.append({
                    "category": category,
                    "evidence": evidence,
                    "source": "contextual_dictionary",
                    "ontology_concept": False
                })

        return results

    def get_ontology_relations(self, matched_concepts):
        """
        Return actual ontology relations whose two endpoints
        are both among the detected concepts.

        A relation in the ontology does not automatically mean
        that the document explicitly expresses that relation.
        """

        matched_ids = {
            str(concept["id"])
            for concept in matched_concepts
        }

        results = []

        for relation in self.relations:

            source_id = str(relation["source_node_id"])
            target_id = str(relation["destination_node_id"])

            if (
                source_id in matched_ids
                and target_id in matched_ids
            ):

                results.append({
                    "source_id": source_id,
                    "source": self.nodes[source_id]["name"],
                    "relation": relation["name"],
                    "target_id": target_id,
                    "target": self.nodes[target_id]["name"],
                    "source_type": "ontology"
                })

        return results

    def match_document(self, text):

        concepts = self.match_text(text)
        contextual_indicators = self.match_context(text)

        return {
            "concepts": concepts,
            "contextual_indicators": contextual_indicators,
            "ontology_relations": self.get_ontology_relations(
                concepts
            )
        }

if __name__ == "__main__":

    matcher = OntologyMatcher()

    sample_text = """
    The financial sector faces increasing cybersecurity risks.

    Digital fraud and cyber attacks threaten financial services.

    Machine learning algorithms such as Random Forest and XGBoost
    can be applied to financial fraud detection.

    Apache Spark supports large-scale data processing.
    """

    result = matcher.match_document(sample_text)

    print("\n=== ONTOLOGY CONCEPTS ===")

    for concept in result["concepts"]:

        print(
            f"- [{concept['id']}] {concept['name']}"
            f" | Categories: {', '.join(concept['categories'])}"
            f" | Evidence: {', '.join(concept['evidence'])}"
        )

    print("\n=== CONTEXTUAL INDICATORS ===")

    for indicator in result["contextual_indicators"]:

        print(
            f"- {indicator['category']}: "
            f"{', '.join(indicator['evidence'])}"
        )

    print("\n=== ONTOLOGY RELATIONS ===")

    for relation in result["ontology_relations"]:

        print(
            f"- {relation['source']} "
            f"--{relation['relation']}--> "
            f"{relation['target']}"
        )