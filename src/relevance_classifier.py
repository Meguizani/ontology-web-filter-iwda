
import re

from src.ontology_matcher import OntologyMatcher


class RelevanceClassifier:

    def __init__(self):
        self.matcher = OntologyMatcher()

        # Financial domain concepts
        self.financial_categories = {
            "financial_risk"
        }

        # Fraud detection concepts
        self.fraud_problem_categories = {
            "fraud"
        }

        # Financial risk assessment concepts
        self.risk_problem_categories = {
            "financial_risk"
        }

        # Technologies and methods
        self.technology_categories = {
            "machine_learning",
            "algorithms",
            "big_data",
            "data_preparation",
            "explainability"
        }

        # Financial context expressions
        self.financial_context_terms = [
            "financial sector",
            "financial services",
            "financial institutions",
            "financial institution",
            "banking sector",
            "banking industry",
            "payment systems",
            "payment system",
            "financial fraud",
            "financial transaction",
            "financial transactions",
            "payment fraud",
            "credit card fraud",
            "bank fraud",
            "banks",
            "bank"
        ]

        # Artificial intelligence context
        self.ai_context_terms = [
            "artificial intelligence",
            "deep learning",
            "predictive analytics",
            "automated decision",
            "automated decision making"
        ]

        # Cybersecurity context
        self.cyber_context_terms = [
            "cybersecurity",
            "cyber security",
            "cyber threats",
            "cyber attacks",
            "cyber attack",
            "digital security",
            "cyber events"
        ]

    @staticmethod
    def contains_phrase(text, phrase):
        """
        Detect a complete expression in text.
        Matching is case-insensitive and supports
        variable whitespace between words.
        """

        words = phrase.strip().split()

        if not words:
            return False

        pattern = (
            r"(?<!\w)"
            + r"\s+".join(
                re.escape(word) for word in words
            )
            + r"(?!\w)"
        )

        return re.search(
            pattern,
            text,
            flags=re.IGNORECASE
        ) is not None

    def find_context_evidence(self, text, expressions):
        """
        Return all contextual expressions found in a document.
        """

        return [
            expression
            for expression in expressions
            if self.contains_phrase(text, expression)
        ]

    def classify(self, text):

        if not isinstance(text, str) or not text.strip():
            raise ValueError(
                "The document text must be a non-empty string."
            )

        # --------------------------------------------------
        # STEP 1: Ontology-based concept matching
        # --------------------------------------------------

        match_result = self.matcher.match_document(text)

        concepts = match_result["concepts"]

        # --------------------------------------------------
        # STEP 2: Extract detected ontology categories
        # --------------------------------------------------

        fraud_concepts = [
            concept
            for concept in concepts
            if set(concept["categories"])
            & self.fraud_problem_categories
        ]

        risk_concepts = [
            concept
            for concept in concepts
            if set(concept["categories"])
            & self.risk_problem_categories
        ]

        financial_concepts = [
            concept
            for concept in concepts
            if set(concept["categories"])
            & self.financial_categories
        ]

        technology_concepts = [
            concept
            for concept in concepts
            if set(concept["categories"])
            & self.technology_categories
        ]

        # --------------------------------------------------
        # STEP 3: Detect contextual evidence
        # --------------------------------------------------

        financial_context = self.find_context_evidence(
            text,
            self.financial_context_terms
        )

        ai_context = self.find_context_evidence(
            text,
            self.ai_context_terms
        )

        cyber_context = self.find_context_evidence(
            text,
            self.cyber_context_terms
        )

        # --------------------------------------------------
        # STEP 4: Determine research dimensions
        # --------------------------------------------------

        # A financial risk concept is direct financial evidence.
        # Fraud concepts require financial context to establish
        # that the fraud discussed is related to finance.

        financial_detected = (
            bool(financial_concepts)
            or bool(financial_context)
        )

        fraud_detected = bool(fraud_concepts)

        risk_detected = bool(risk_concepts)

        problem_detected = (
            fraud_detected or risk_detected
        )

        technology_detected = bool(
            technology_concepts
        )

        contextual_support = (
            bool(ai_context)
            or bool(cyber_context)
        )

        # --------------------------------------------------
        # STEP 5: Relevance classification
        # --------------------------------------------------

        if (
            financial_detected
            and problem_detected
            and technology_detected
        ):

            relevance_level = "DIRECTLY_RELEVANT"

            explanation = (
                "The document contains evidence of the "
                "financial domain, a fraud detection or "
                "financial risk assessment problem, and "
                "relevant technologies or methods."
            )

        elif (
            financial_detected
            and (
                problem_detected
                or technology_detected
                or contextual_support
            )
        ):

            relevance_level = "CONTEXTUALLY_RELATED"

            explanation = (
                "The document is connected to the financial "
                "domain and contains partial evidence related "
                "to the research topic. However, all three "
                "research dimensions were not detected."
            )

        else:

            relevance_level = "NOT_RELEVANT"

            explanation = (
                "Insufficient evidence connecting the document "
                "to the financial fraud and risk research topic."
            )

        # --------------------------------------------------
        # STEP 6: Build classification result
        # --------------------------------------------------

        return {

            "classification": relevance_level,

            "relevance_level": relevance_level,

            "research_dimensions": {

                "financial_domain": {
                    "detected": financial_detected,

                    "evidence": [
                        concept["name"]
                        for concept in financial_concepts
                    ],

                    "context_evidence": financial_context
                },

                "fraud_detection": {
                    "detected": fraud_detected,

                    "evidence": [
                        concept["name"]
                        for concept in fraud_concepts
                    ]
                },

                "financial_risk_assessment": {
                    "detected": risk_detected,

                    "evidence": [
                        concept["name"]
                        for concept in risk_concepts
                    ]
                },

                "research_problem": {
                    "detected": problem_detected,

                    "evidence": [
                        concept["name"]
                        for concept in (
                            fraud_concepts + risk_concepts
                        )
                    ]
                },

                "technology": {
                    "detected": technology_detected,

                    "evidence": [
                        concept["name"]
                        for concept in technology_concepts
                    ]
                },

                "contextual_support": {
                    "ai": ai_context,
                    "cyber": cyber_context
                }
            },

            # Concepts identified in the document
            "detected_concepts": concepts,

            # Existing ontology relations between detected concepts
            "ontology_relations": match_result[
                "ontology_relations"
            ],

            # Classification explanation
            "explanation": explanation
        }


# ==========================================================
# TESTING
# ==========================================================

if __name__ == "__main__":

    classifier = RelevanceClassifier()

    test_documents = [

        {
            "name": "Fraud detection with machine learning",

            "text": """
            Financial fraud detection uses machine learning
            algorithms such as Random Forest and XGBoost.
            """
        },

        {
            "name": "Financial cybersecurity",

            "text": """
            Artificial intelligence and cybersecurity are
            transforming financial services and banking.
            """
        },

        {
            "name": "Financial risk without fraud",

            "text": """
            Financial institutions use machine learning
            to assess financial risk and credit risk.
            """
        },

        {
            "name": "Fraud without named technology",

            "text": """
            Banks are implementing fraud detection systems
            to identify suspicious financial transactions.
            """
        },

        {
            "name": "Unrelated document",

            "text": """
            Mars is the fourth planet from the Sun.
            Its surface contains mountains and craters.
            """
        }
    ]

    for document in test_documents:

        result = classifier.classify(
            document["text"]
        )

        print("\n" + "=" * 65)

        print("DOCUMENT:", document["name"])

        print("LEVEL:", result["relevance_level"])

        print("CLASS:", result["classification"])

        print("EXPLANATION:", result["explanation"])

        print("\nRESEARCH DIMENSIONS:")

        for dimension, details in result[
            "research_dimensions"
        ].items():

            print(f"- {dimension}: {details}")

        print("\nDETECTED CONCEPTS:")

        for concept in result["detected_concepts"]:

            print(
                f"- [{concept['id']}] "
                f"{concept['name']} "
                f"| Categories: {', '.join(concept['categories'])}"
            )

        print("\nONTOLOGY RELATIONS:")

        for relation in result["ontology_relations"]:

            print(
                f"- {relation['source']} "
                f"--{relation['relation']}--> "
                f"{relation['target']}"
            )
