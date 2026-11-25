
import json
import logging
from itertools import combinations
from pathlib import Path

from src.text_processor import TextProcessor
from src.ontology_matcher import OntologyMatcher
from src.relevance_classifier import RelevanceClassifier


# ---------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------

ROOT_DIR = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT_DIR / "data" / "output"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------
# SEMANTIC ANALYZER
# ---------------------------------------------------------

class SemanticAnalyzer:
    """
    Semantic analysis pipeline for the Intelligent Web Data Analysis
    project.

    Main components:
        1. TextProcessor:
           Linguistic preprocessing using spaCy.

        2. OntologyMatcher:
           Detection of concepts defined in the research ontology.

        3. RelevanceClassifier:
           Classification of documents according to the research topic.

    The analyzer combines linguistic information, ontology concepts,
    sentence-level analysis and document classification.
    """

    def __init__(self):
        logger.info("Initializing Semantic Analyzer...")

        # Initialize linguistic processor
        self.text_processor = TextProcessor()

        # Initialize ontology matcher
        self.ontology_matcher = OntologyMatcher()

        # Initialize relevance classifier
        self.relevance_classifier = RelevanceClassifier()

        logger.info("Semantic Analyzer initialized successfully.")

    # -----------------------------------------------------
    # HELPER METHODS
    # -----------------------------------------------------

    @staticmethod
    def _extract_text(item):
        """
        Extract text from a string or a dictionary.
        """
        if isinstance(item, str):
            return item.strip()

        if isinstance(item, dict):
            return (
                item.get("text")
                or item.get("sentence")
                or item.get("name")
                or ""
            ).strip()

        return ""

    @staticmethod
    def _extract_concepts(matching_result):
        """
        Extract detected concepts from the ontology matcher result.

        Supports the common result keys used by the matcher.
        """
        if not isinstance(matching_result, dict):
            return []

        return (
            matching_result.get("concepts")
            or matching_result.get("matched_concepts")
            or matching_result.get("detected_concepts")
            or []
        )

    @staticmethod
    def _concept_identifier(concept):
        """
        Return a stable identifier for a detected concept.
        """
        if not isinstance(concept, dict):
            return str(concept)

        return (
            concept.get("id")
            if concept.get("id") is not None
            else concept.get("name", "")
        )

    @staticmethod
    def _concept_name(concept):
        """
        Return the readable name of a detected concept.
        """
        if isinstance(concept, dict):
            return concept.get("name", str(concept))

        return str(concept)

    # -----------------------------------------------------
    # SENTENCE-LEVEL ANALYSIS
    # -----------------------------------------------------

    def analyze_sentence(self, sentence_text, sentence_id=None):
        """
        Analyze an individual sentence.

        Operations:
            - Linguistic processing
            - Ontology concept matching
            - Extraction of detected concepts
        """

        if not sentence_text or not sentence_text.strip():
            return None

        # Linguistic analysis
        linguistic_result = self.text_processor.process(sentence_text)

        # Ontology matching
        ontology_result = self.ontology_matcher.match_document(
            sentence_text
        )

        concepts = self._extract_concepts(ontology_result)

        return {
            "sentence_id": sentence_id,
            "sentence": sentence_text,
            "linguistic_analysis": linguistic_result,
            "detected_concepts": concepts,
            "ontology_relations": ontology_result.get(
                "ontology_relations", []
            ) if isinstance(ontology_result, dict) else []
        }

    # -----------------------------------------------------
    # TEXTUAL COOCCURRENCE ANALYSIS
    # -----------------------------------------------------

    def find_sentence_cooccurrences(self, sentence_results):
        """
        Identify pairs of ontology concepts appearing in the same
        sentence.

        IMPORTANT:
        Co-occurrence does not prove a semantic relationship.

        For this reason, each result explicitly contains:
            evidence_type = same_sentence_cooccurrence
            semantic_relation_confirmed = False
        """

        cooccurrences = []

        for sentence_result in sentence_results:

            concepts = sentence_result.get(
                "detected_concepts", []
            )

            sentence_text = sentence_result.get("sentence", "")

            # Avoid duplicate concepts in the same sentence
            unique_concepts = {}

            for concept in concepts:
                identifier = self._concept_identifier(concept)

                if identifier:
                    unique_concepts[identifier] = concept

            concept_values = list(unique_concepts.values())

            # Generate all possible concept pairs
            for concept_a, concept_b in combinations(
                concept_values, 2
            ):

                cooccurrences.append({
                    "sentence_id": sentence_result.get(
                        "sentence_id"
                    ),
                    "source_concept": self._concept_name(
                        concept_a
                    ),
                    "target_concept": self._concept_name(
                        concept_b
                    ),
                    "evidence_sentence": sentence_text,
                    "evidence_type": "same_sentence_cooccurrence",
                    "semantic_relation_confirmed": False
                })

        return cooccurrences

    # -----------------------------------------------------
    # DOCUMENT CLASSIFICATION
    # -----------------------------------------------------

    def classify_document(self, text):
        """
        Classify a document using the RelevanceClassifier.
        """

        return self.relevance_classifier.classify(text)

    # -----------------------------------------------------
    # COMPLETE DOCUMENT ANALYSIS
    # -----------------------------------------------------

    def analyze_document(self, text):
        """
        Execute the complete semantic analysis pipeline.

        Steps:
            1. Linguistic processing
            2. Document-level ontology matching
            3. Sentence-level concept extraction
            4. Noun chunk matching
            5. Named entity matching
            6. Textual co-occurrence extraction
            7. Document classification

        Returns a structured dictionary containing the results.
        """

        if not isinstance(text, str) or not text.strip():
            raise ValueError(
                "The document text must be a non-empty string."
            )

        logger.info("Starting document semantic analysis...")

        # -------------------------------------------------
        # STEP 1: LINGUISTIC PROCESSING
        # -------------------------------------------------

        linguistic_result = self.text_processor.process(text)

        statistics = linguistic_result.get(
            "statistics", {}
        )

        sentences = linguistic_result.get(
            "sentences", []
        )

        tokens = linguistic_result.get(
            "tokens", []
        )

        noun_chunks = linguistic_result.get(
            "noun_chunks", []
        )

        entities = linguistic_result.get(
            "entities", []
        )

        dependency_relations = linguistic_result.get(
            "dependency_relations",
            linguistic_result.get("dependencies", [])
        )

        # -------------------------------------------------
        # STEP 2: DOCUMENT-LEVEL ONTOLOGY MATCHING
        # -------------------------------------------------

        document_ontology_result = (
            self.ontology_matcher.match_document(text)
        )

        document_concepts = self._extract_concepts(
            document_ontology_result
        )

        ontology_relations = (
            document_ontology_result.get(
                "ontology_relations", []
            )
            if isinstance(document_ontology_result, dict)
            else []
        )

        # -------------------------------------------------
        # STEP 3: SENTENCE-LEVEL ANALYSIS
        # -------------------------------------------------

        sentence_results = []

        for index, sentence in enumerate(sentences, start=1):

            sentence_text = self._extract_text(sentence)

            if not sentence_text:
                continue

            sentence_result = self.analyze_sentence(
                sentence_text,
                sentence_id=index
            )

            if sentence_result:
                sentence_results.append(sentence_result)

        # -------------------------------------------------
        # STEP 4: NOUN CHUNK MATCHING
        # -------------------------------------------------

        noun_chunk_matches = []

        for chunk in noun_chunks:

            chunk_text = self._extract_text(chunk)

            if not chunk_text:
                continue

            chunk_result = self.ontology_matcher.match_document(
                chunk_text
            )

            matched_concepts = self._extract_concepts(
                chunk_result
            )

            if matched_concepts:
                noun_chunk_matches.append({
                    "noun_chunk": chunk_text,
                    "matched_concepts": matched_concepts
                })

        # -------------------------------------------------
        # STEP 5: NAMED ENTITY MATCHING
        # -------------------------------------------------

        entity_matches = []

        for entity in entities:

            entity_text = self._extract_text(entity)

            if not entity_text:
                continue

            entity_result = self.ontology_matcher.match_document(
                entity_text
            )

            matched_concepts = self._extract_concepts(
                entity_result
            )

            if matched_concepts:

                entity_matches.append({
                    "entity": entity_text,
                    "entity_label": (
                        entity.get("label")
                        or entity.get("entity_type")
                        if isinstance(entity, dict)
                        else None
                    ),
                    "matched_concepts": matched_concepts
                })

        # -------------------------------------------------
        # STEP 6: TEXTUAL COOCCURRENCES
        # -------------------------------------------------

        textual_cooccurrences = (
            self.find_sentence_cooccurrences(
                sentence_results
            )
        )

        # -------------------------------------------------
        # STEP 7: DOCUMENT CLASSIFICATION
        # -------------------------------------------------

        classification_result = self.classify_document(text)

        # -------------------------------------------------
        # STEP 8: FINAL RESULT
        # -------------------------------------------------

        analysis_result = {
            "document_statistics": statistics,

            "linguistic_analysis": {
                "tokens": tokens,
                "sentences": sentences,
                "noun_chunks": noun_chunks,
                "named_entities": entities,
                "dependency_relations": dependency_relations
            },

            "ontology_analysis": {
                "detected_concepts": document_concepts,
                "ontology_relations": ontology_relations
            },

            "sentence_analysis": sentence_results,

            "noun_chunk_matching": noun_chunk_matches,

            "entity_matching": entity_matches,

            "textual_cooccurrences": textual_cooccurrences,

            "classification": classification_result
        }

        logger.info(
            "Semantic analysis completed successfully."
        )

        return analysis_result

    # -----------------------------------------------------
    # SAVE ANALYSIS RESULTS
    # -----------------------------------------------------

    @staticmethod
    def save_results(results, output_file):
        """
        Save analysis results to a JSON file.
        """

        output_path = Path(output_file)

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        with open(
            output_path,
            "w",
            encoding="utf-8"
        ) as file:

            json.dump(
                results,
                file,
                indent=4,
                ensure_ascii=False,
                default=str
            )

        logger.info(
            "Results saved to: %s",
            output_path
        )


# ---------------------------------------------------------
# TEST EXECUTION
# ---------------------------------------------------------

if __name__ == "__main__":

    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s: %(message)s"
    )

    analyzer = SemanticAnalyzer()

    sample_text = """
    Financial fraud detection is an important challenge
    for financial institutions.

    Machine learning algorithms such as Random Forest
    and XGBoost can be used to detect fraudulent transactions.

    Apache Spark supports large-scale data processing
    for financial datasets.

    SHAP helps explain predictions made by machine
    learning models.
    """

    print("\nProcessing document...\n")

    results = analyzer.analyze_document(sample_text)

    # -----------------------------------------------------
    # DISPLAY STATISTICS
    # -----------------------------------------------------

    print("\nDOCUMENT STATISTICS:")

    for key, value in results[
        "document_statistics"
    ].items():

        print(f"{key}: {value}")

    # -----------------------------------------------------
    # DISPLAY ONTOLOGY CONCEPTS
    # -----------------------------------------------------

    print("\nDETECTED ONTOLOGY CONCEPTS:")

    for concept in results[
        "ontology_analysis"
    ]["detected_concepts"]:

        if isinstance(concept, dict):
            print(
                f"- [{concept.get('id')}] "
                f"{concept.get('name')} | "
                f"Categories: {concept.get('categories', [])}"
            )
        else:
            print(f"- {concept}")

    # -----------------------------------------------------
    # DISPLAY SENTENCE ANALYSIS
    # -----------------------------------------------------

    print("\nSENTENCE LEVEL:")

    for item in results["sentence_analysis"]:

        names = [
            concept.get("name", str(concept))
            if isinstance(concept, dict)
            else str(concept)
            for concept in item["detected_concepts"]
        ]

        print(
            f"{item['sentence_id']}. "
            f"{item['sentence']}"
        )

        print(f"   Concepts: {', '.join(names) or 'None'}")

    # -----------------------------------------------------
    # DISPLAY NOUN CHUNK MATCHING
    # -----------------------------------------------------

    print("\nNOUN CHUNK MATCHING:")

    for item in results["noun_chunk_matching"]:

        names = [
            concept.get("name", str(concept))
            if isinstance(concept, dict)
            else str(concept)
            for concept in item["matched_concepts"]
        ]

        print(
            f"- {item['noun_chunk']} -> "
            f"{', '.join(names)}"
        )

    # -----------------------------------------------------
    # DISPLAY ENTITY MATCHING
    # -----------------------------------------------------

    print("\nENTITY MATCHING:")

    for item in results["entity_matching"]:

        print(
            f"- {item['entity']} "
            f"({item['entity_label']})"
        )

    # -----------------------------------------------------
    # DISPLAY ONTOLOGY RELATIONS
    # -----------------------------------------------------

    print("\nONTOLOGY RELATIONS:")

    relations = results[
        "ontology_analysis"
    ]["ontology_relations"]

    if relations:

        for relation in relations:
            print(f"- {relation}")

    else:
        print(
            "No direct ontology relations between "
            "detected concepts."
        )

    # -----------------------------------------------------
    # DISPLAY TEXTUAL COOCCURRENCES
    # -----------------------------------------------------

    print("\nTEXTUAL ASSOCIATIONS:")

    for item in results["textual_cooccurrences"]:

        print(
            f"- {item['source_concept']} + "
            f"{item['target_concept']}"
        )

        print(
            f"  Evidence: {item['evidence_type']}"
        )

    # -----------------------------------------------------
    # DISPLAY CLASSIFICATION
    # -----------------------------------------------------

    print("\nCLASSIFICATION:")

    print(
        json.dumps(
            results["classification"],
            indent=4,
            ensure_ascii=False
        )
    )

    # -----------------------------------------------------
    # SAVE RESULTS
    # -----------------------------------------------------

    output_file = (
        OUTPUT_DIR / "semantic_analysis_test.json"
    )

    analyzer.save_results(
        results,
        output_file
    )

    print(
        f"\nResults saved to: {output_file}"
    )
