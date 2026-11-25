
import json
import re
import unicodedata
from pathlib import Path

import scrapy

from src.semantic_analyzer import SemanticAnalyzer

# =========================================================
# PATHS
# =========================================================

ROOT_DIR = Path(__file__).resolve().parents[2]

ONTOLOGY_FILE = (
    ROOT_DIR / "data" / "output" / "domain_concepts.json"
)

URL_FILE = (
    ROOT_DIR / "data" / "input" / "urls.txt"
)

# =========================================================
# V4 CONFIGURATION
# =========================================================

DOMAIN_ANCHORS = {
    "fraud detection",
    "financial risk",
    "financial security",
    "credit risk assessment",
    "financial risk assessment task",
    "financial security task",
    "fraud detection system",
    "financial fraud detection workflow",
}

ALIASES = {
    "fraud detection": [
        "fraud detection",
        "digital fraud",
        "financial fraud",
        "digital fraud detection",
        "cyber fraud",
        "fraudulent transaction",
        "fraudulent transactions",
        "fraudulent transaction detection",
        "credit card fraud",
        "credit card fraud detection",
        "payment fraud",
        "payment fraud detection",
        "online payment fraud",
        "banking fraud",
    ],

    "financial risk": [
        "financial risk",
        "financial risks",
        "financial risk management",
        "risk in financial services",
        "financial risk assessment",
        "risk assessment in finance",
        "financial risk analysis",
    ],

    "machine learning": [
        "machine learning",
        "machine-learning",
        "ml algorithms",
        "machine learning algorithms",
        "supervised machine learning",
        "unsupervised machine learning",
    ],

    "big data": [
        "big data",
        "large-scale data",
        "large datasets",
        "big data analytics",
        "big data processing",
    ],

    "anomaly detection": [
        "anomaly detection",
        "anomalies detection",
        "outlier detection",
        "fraud anomaly detection",
        "transaction anomaly detection",
    ],

    "credit risk assessment": [
        "credit risk assessment",
        "credit risk analysis",
        "credit scoring",
        "credit risk prediction",
    ],

    "financial security": [
        "financial security",
        "financial cybersecurity",
        "cybersecurity in financial services",
        "cybersecurity in the financial sector",
        "cyber security in financial services",
        "cyber threats in finance",
        "security of financial transactions",
    ],

    "machine learning model": [
        "machine learning model",
        "machine learning models",
        "trained machine learning model",
    ],
}

RELATION_LABEL_CORRECTIONS = {
    "machine learningmodel": "Machine Learning Model",
}

FINANCIAL_CONTEXT = [
    "financial sector",
    "financial services",
    "banking sector",
    "banking services",
    "financial institutions",
    "banking transactions",
    "payment systems",
    "financial transactions",
]

AI_CONTEXT = [
    "artificial intelligence",
    "artificial-intelligence",
    "ai systems",
    "ai models",
]

CYBER_CONTEXT = [
    "cybersecurity",
    "cyber security",
    "cyber threats",
    "cyber attacks",
    "cyber events",
]

RELATION_WEIGHTS = {
    "is_a": 1.5,
    "is_instance_of": 1.5,
    "a_part_of": 2.0,
    "use_for": 2.0,
    "has": 1.0,
}

# =========================================================
# TEXT UTILITIES
# =========================================================

def normalize_text(text):
    text = unicodedata.normalize("NFKC", text or "")
    text = text.casefold()
    text = re.sub(r"\s+", " ", text)

    return text.strip()

def contains_expression(text, expression):
    expression = normalize_text(expression)

    if not expression:
        return False

    pattern = (
        r"(?<!\w)"
        + re.escape(expression)
        + r"(?!\w)"
    )

    return bool(re.search(pattern, text))

def split_sentences(text):
    """
    Basic sentence splitting.
    The advanced linguistic analysis is performed by spaCy.
    """

    return [
        sentence.strip()
        for sentence in re.split(
            r"(?<=[.!?])\s+",
            text
        )
        if sentence.strip()
    ]

# =========================================================
# SCRAPY SPIDER
# =========================================================

class OntologySpider(scrapy.Spider):

    name = "ontology"

    custom_settings = {

        # Crawling policy
        "ROBOTSTXT_OBEY": True,

        "DOWNLOAD_DELAY": 1,

        "CONCURRENT_REQUESTS": 8,

        "CONCURRENT_REQUESTS_PER_DOMAIN": 4,

        # AutoThrottle
        "AUTOTHROTTLE_ENABLED": True,

        "AUTOTHROTTLE_START_DELAY": 1,

        "AUTOTHROTTLE_MAX_DELAY": 60,

        # Retry configuration
        "RETRY_ENABLED": True,

        "RETRY_TIMES": 3,

        "RETRY_HTTP_CODES": [
            408,
            429,
            500,
            502,
            503,
            504,
        ],

        # Download timeout
        "DOWNLOAD_TIMEOUT": 30,

        # User agent
        "USER_AGENT": (
            "Mozilla/5.0 "
            "(compatible; OntologyWebResearchBot/3.0)"
        ),

        "LOG_LEVEL": "INFO",

        "MEMUSAGE_ENABLED": False,
    }

    # =====================================================
    # INITIALIZATION
    # =====================================================

    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)

        # -------------------------------------------------
        # Load V4 ontology catalogue
        # -------------------------------------------------

        if not ONTOLOGY_FILE.exists():

            raise FileNotFoundError(
                f"Ontology file not found: {ONTOLOGY_FILE}"
            )

        with open(
            ONTOLOGY_FILE,
            "r",
            encoding="utf-8"
        ) as file:

            ontology = json.load(file)

        self.core_concepts = ontology.get(
            "core_concepts", []
        )

        self.related_concepts = ontology.get(
            "related_concepts", []
        )

        self.relations = ontology.get(
            "relations", []
        )

        self.all_concepts = {}

        self.build_concept_catalogue()

        self.normalized_relations = (
            self.prepare_relations()
        )

        # -------------------------------------------------
        # Load URLs
        # -------------------------------------------------

        if not URL_FILE.exists():

            raise FileNotFoundError(
                f"URL file not found: {URL_FILE}"
            )

        with open(
            URL_FILE,
            "r",
            encoding="utf-8"
        ) as file:

            self.urls = [
                line.strip()
                for line in file
                if (
                    line.strip()
                    and not line.strip().startswith("#")
                )
            ]

        if not self.urls:

            raise ValueError(
                "No URLs found in urls.txt"
            )

        # -------------------------------------------------
        # Initialize V5 Semantic Analyzer
        # -------------------------------------------------

        self.logger.info(
            "Initializing V5 Semantic Analyzer..."
        )

        self.semantic_analyzer = SemanticAnalyzer()

        self.logger.info(
            "Semantic Analyzer initialized successfully."
        )

        # -------------------------------------------------
        # Initialization logs
        # -------------------------------------------------

        self.logger.info(
            "V4 ontology loaded: %s core concepts, "
            "%s related concepts",
            len(self.core_concepts),
            len(self.related_concepts),
        )

        self.logger.info(
            "Ontology relations loaded: %s",
            len(self.normalized_relations),
        )

        self.logger.info(
            "Unified concept catalogue: %s",
            len(self.all_concepts),
        )

        self.logger.info(
            "URLs loaded: %s",
            len(self.urls),
        )

    # =====================================================
    # ONTOLOGY CATALOGUE
    # =====================================================

    def correct_relation_label(self, name):

        return RELATION_LABEL_CORRECTIONS.get(
            normalize_text(name),
            name.strip()
        )

    def build_concept_catalogue(self):

        for concept in (
            self.core_concepts + self.related_concepts
        ):

            self.all_concepts[
                normalize_text(concept["name"])
            ] = concept

        for relation in self.relations:

            for endpoint in (
                relation.get("source"),
                relation.get("target")
            ):

                if not endpoint:
                    continue

                canonical = self.correct_relation_label(
                    endpoint
                )

                key = normalize_text(canonical)

                if key not in self.all_concepts:

                    self.all_concepts[key] = {
                        "id": None,
                        "name": canonical,
                        "type": "relation_endpoint",
                    }

    def prepare_relations(self):

        prepared = []

        for relation in self.relations:

            original_source = relation.get(
                "source", ""
            )

            original_target = relation.get(
                "target", ""
            )

            prepared.append({

                "source": self.correct_relation_label(
                    original_source
                ),

                "relation": relation.get(
                    "relation", "unknown"
                ),

                "target": self.correct_relation_label(
                    original_target
                ),

                "original_source": original_source,

                "original_target": original_target,
            })

        return prepared

    # =====================================================
    # START CRAWLING
    # =====================================================

    async def start(self):

        for url in self.urls:

            self.logger.info(
                "Scheduling URL: %s",
                url
            )

            yield scrapy.Request(

                url=url,

                callback=self.parse_page,

                errback=self.handle_error,

                meta={

                    "original_url": url,

                    "handle_httpstatus_all": True,
                },

                dont_filter=True,
            )

    # =====================================================
    # CONCEPT DETECTION - V4
    # =====================================================

    def find_concepts(
        self,
        text,
        title,
        concepts
    ):

        normalized_text = normalize_text(text)

        normalized_title = normalize_text(title)

        detected = []

        for concept in concepts:

            name = concept["name"]

            canonical = normalize_text(name)

            variants = list(dict.fromkeys(
                [name] + ALIASES.get(canonical, [])
            ))

            found_title = any(
                contains_expression(
                    normalized_title,
                    variant
                )
                for variant in variants
            )

            found_body = any(
                contains_expression(
                    normalized_text,
                    variant
                )
                for variant in variants
            )

            if found_title or found_body:

                detected.append({

                    "id": concept.get("id"),

                    "name": name,

                    "type": concept.get("type"),

                    "in_title": found_title,

                    "in_body": found_body,
                })

        return detected

    # =====================================================
    # RELATION EVIDENCE - V4
    # =====================================================

    def find_relation_evidence(
        self,
        title,
        content
    ):

        """
        Detect relation endpoints appearing in the same sentence.

        IMPORTANT:
        This is co-occurrence evidence, not full semantic
        relation extraction.
        """

        sentences = split_sentences(
            f"{title}. {content}"
        )

        evidence = []

        for relation in self.normalized_relations:

            source = relation["source"]

            target = relation["target"]

            source_variants = list(dict.fromkeys(
                [source] + ALIASES.get(
                    normalize_text(source), []
                )
            ))

            target_variants = list(dict.fromkeys(
                [target] + ALIASES.get(
                    normalize_text(target), []
                )
            ))

            for sentence in sentences:

                normalized_sentence = normalize_text(
                    sentence
                )

                source_found = any(
                    contains_expression(
                        normalized_sentence,
                        term
                    )
                    for term in source_variants
                )

                target_found = any(
                    contains_expression(
                        normalized_sentence,
                        term
                    )
                    for term in target_variants
                )

                if source_found and target_found:

                    evidence.append({

                        "source": source,

                        "relation": relation["relation"],

                        "target": target,

                        "original_source": (
                            relation["original_source"]
                        ),

                        "original_target": (
                            relation["original_target"]
                        ),

                        "evidence_sentence": sentence[:500],

                        "evidence_type": (
                            "same_sentence_cooccurrence"
                        ),
                    })

                    break

        return evidence

    # =====================================================
    # CONTEXT ANALYSIS - V4
    # =====================================================

    def analyze_context(self, title, content):

        text = normalize_text(
            title + " " + content
        )

        return {

            "financial_context": [
                term
                for term in FINANCIAL_CONTEXT
                if contains_expression(text, term)
            ],

            "ai_context": [
                term
                for term in AI_CONTEXT
                if contains_expression(text, term)
            ],

            "cyber_context": [
                term
                for term in CYBER_CONTEXT
                if contains_expression(text, term)
            ],
        }

    # =====================================================
    # CLASSIFICATION - V4
    # =====================================================

    def classify_page(self, title, content):

        core_found = self.find_concepts(
            content,
            title,
            self.core_concepts
        )

        related_found = self.find_concepts(
            content,
            title,
            self.related_concepts
        )

        all_found = self.find_concepts(
            content,
            title,
            list(self.all_concepts.values())
        )

        context = self.analyze_context(
            title,
            content
        )

        relation_evidence = self.find_relation_evidence(
            title,
            content
        )

        # -------------------------------------------------
        # Calculate concept scores
        # -------------------------------------------------

        core_score = sum(
            3 if item["in_title"] else 2
            for item in core_found
        )

        related_score = sum(
            1.5 if item["in_title"] else 1
            for item in related_found
        )

        # -------------------------------------------------
        # Context score
        # -------------------------------------------------

        context_score = 0

        if context["financial_context"]:
            context_score += 1

        if (
            context["ai_context"]
            and context["cyber_context"]
        ):
            context_score += 1

        # -------------------------------------------------
        # Relation score
        # -------------------------------------------------

        relation_score_raw = sum(
            RELATION_WEIGHTS.get(
                item["relation"], 0
            )
            for item in relation_evidence
        )

        relation_score = min(
            relation_score_raw,
            3.0
        )

        total_score = (
            core_score
            + related_score
            + context_score
            + relation_score
        )

        detected_names = {
            normalize_text(item["name"])
            for item in all_found
        }

        # -------------------------------------------------
        # Research dimensions
        # -------------------------------------------------

        financial_domain_concepts = {

            "financial risk",
            "financial security",
            "financial security task",
            "credit risk assessment",
            "financial risk assessment task",
            "fraud detection",
            "fraud detection system",
            "financial fraud detection workflow",
        }

        research_problem_concepts = {

            "fraud detection",
            "financial risk",
            "credit risk assessment",
            "financial risk assessment task",
            "anomaly detection",
            "fraud prediction",
        }

        technology_concepts = {

            "machine learning",
            "big data",
            "supervised learning",
            "apache spark",
            "random forest",
            "xgboost",
            "logistic regression",
            "isolation forest",
            "scikit-learn",
            "pyspark",
            "machine learning model",
            "anomaly detection algorithm",
        }

        financial_found = sorted(
            detected_names.intersection(
                financial_domain_concepts
            )
        )

        problem_found = sorted(
            detected_names.intersection(
                research_problem_concepts
            )
        )

        technology_found = sorted(
            detected_names.intersection(
                technology_concepts
            )
        )

        financial_dimension = bool(
            financial_found
            or context["financial_context"]
        )

        problem_dimension = bool(problem_found)

        technology_dimension = bool(
            technology_found
        )

        contextual_support = bool(
            context["ai_context"]
            or context["cyber_context"]
        )

        # -------------------------------------------------
        # Three-level relevance classification
        # -------------------------------------------------

        if (
            financial_dimension
            and problem_dimension
            and technology_dimension
        ):

            relevance_level = "DIRECTLY_RELEVANT"

        elif (
            financial_dimension
            and (
                problem_dimension
                or technology_dimension
                or contextual_support
            )
        ):

            relevance_level = "CONTEXTUALLY_RELATED"

        else:

            relevance_level = "NOT_RELEVANT"

        classification = (
            "RELEVANT"
            if relevance_level == "DIRECTLY_RELEVANT"
            else "NOT RELEVANT"
        )

        # -------------------------------------------------
        # Explanation
        # -------------------------------------------------

        if relevance_level == "DIRECTLY_RELEVANT":

            explanation = (
                "Financial domain, research problem and "
                "technology evidence were all detected."
            )

        elif relevance_level == "CONTEXTUALLY_RELATED":

            explanation = (
                "The page is connected to the financial "
                "domain, but evidence for all three research "
                "dimensions was not found."
            )

        else:

            explanation = (
                "Insufficient evidence connecting the page "
                "to the financial fraud and risk research topic."
            )

        return {

            "classification": classification,

            "relevance_level": relevance_level,

            "score": round(total_score, 2),

            "score_breakdown": {

                "core_concepts": round(core_score, 2),

                "related_concepts": round(related_score, 2),

                "context": round(context_score, 2),

                "ontology_relations": round(
                    relation_score, 2
                ),
            },

            "research_dimensions": {

                "financial_domain": {

                    "detected": financial_dimension,

                    "evidence": financial_found,

                    "context_evidence": (
                        context["financial_context"]
                    ),
                },

                "research_problem": {

                    "detected": problem_dimension,

                    "evidence": problem_found,
                },

                "technology": {

                    "detected": technology_dimension,

                    "evidence": technology_found,
                },
            },

            "domain_anchors_found": sorted(
                detected_names.intersection(
                    DOMAIN_ANCHORS
                )
            ),

            "core_concepts_found": core_found,

            "related_concepts_found": related_found,

            "all_ontology_concepts_found": all_found,

            "relation_evidence": relation_evidence,

            "context": context,

            "explanation": explanation,
        }

    # =====================================================
    # HTML CONTENT EXTRACTION
    # =====================================================

    def extract_page_content(self, response):

        title = (

            response.xpath(
                "//title/text()"
            ).get()

            or response.xpath(
                "//h1[1]//text()"
            ).get()

            or ""

        ).strip()

        main = response.xpath(
            "//main[1]"
        )

        if not main:

            main = response.xpath(
                "//article[1]"
            )

        if not main:

            main = response.xpath(
                "//body[1]"
            )

        excluded_tags = [

            "script",
            "style",
            "nav",
            "footer",
            "header",
            "aside",
            "form",
            "noscript",
        ]

        excluded_xpath = " and ".join(
            f"not(ancestor::{tag})"
            for tag in excluded_tags
        )

        content_parts = main.xpath(

            ".//text()[normalize-space() and "

            + excluded_xpath

            + "]"

        ).getall()

        content = re.sub(
            r"\s+",
            " ",
            " ".join(content_parts)
        ).strip()

        return title, content

    # =====================================================
    # PAGE PROCESSING - V4 + V5
    # =====================================================

    def parse_page(self, response):

        original_url = response.meta.get(
            "original_url",
            response.url
        )

        # -------------------------------------------------
        # HTTP status validation
        # -------------------------------------------------

        if response.status != 200:

            location = response.headers.get(
                "Location",
                b""
            ).decode("latin-1")

            self.logger.warning(
                "HTTP error %s: %s",
                response.status,
                response.url
            )

            yield {

                "url": original_url,

                "final_url": response.url,

                "status": "HTTP_ERROR",

                "http_status": response.status,

                "redirect_location": location or None,
            }

            return

        # -------------------------------------------------
        # Content type validation
        # -------------------------------------------------

        content_type = response.headers.get(
            "Content-Type",
            b""
        ).decode("latin-1").lower()

        if "text/html" not in content_type:

            yield {

                "url": original_url,

                "final_url": response.url,

                "status": "UNSUPPORTED_CONTENT_TYPE",

                "content_type": content_type,
            }

            return

        # -------------------------------------------------
        # Extract page content
        # -------------------------------------------------

        title, content = self.extract_page_content(
            response
        )

        if not content:

            yield {

                "url": original_url,

                "final_url": response.url,

                "title": title,

                "status": "EMPTY_CONTENT",
            }

            return

        # -------------------------------------------------
        # V4: Legacy classification
        # -------------------------------------------------

        result = self.classify_page(
            title,
            content
        )

        # -------------------------------------------------
        # V5: Semantic analysis
        # -------------------------------------------------

        self.logger.info(
            "Starting V5 semantic analysis: %s",
            title[:100]
        )

        semantic_analysis = None

        semantic_analysis_error = None

        try:

            semantic_analysis = (
                self.semantic_analyzer.analyze_document(
                    f"{title}\n{content}"
                )
            )

            self.logger.info(
                "V5 semantic analysis completed: %s",
                title[:100]
            )

        except Exception as error:

            semantic_analysis_error = str(error)

            self.logger.exception(
                "V5 semantic analysis failed for %s",
                original_url
            )

        # -------------------------------------------------
        # Log V4 classification
        # -------------------------------------------------

        self.logger.info(

            "[V4: %s | %s] %s | Score: %s",

            result["classification"],

            result["relevance_level"],

            title[:100],

            result["score"],
        )

        # -------------------------------------------------
        # Log V5 classification
        # -------------------------------------------------

        if semantic_analysis is not None:

            v5_classification = (
                semantic_analysis.get(
                    "classification", {}
                )
            )

            self.logger.info(

                "[V5: %s] %s",

                v5_classification.get(
                    "relevance_level",
                    "UNKNOWN"
                ),

                title[:100],
            )

        # -------------------------------------------------
        # Export results
        # -------------------------------------------------

        item = {

            "url": original_url,

            "final_url": response.url,

            "title": title,

            "status": "ANALYZED",

            # V4 results
            "v4_analysis": {

                "classification": (
                    result["classification"]
                ),

                "relevance_level": (
                    result["relevance_level"]
                ),

                "score": result["score"],

                "score_breakdown": (
                    result["score_breakdown"]
                ),

                "research_dimensions": (
                    result["research_dimensions"]
                ),

                "domain_anchors_found": (
                    result["domain_anchors_found"]
                ),

                "core_concepts_found": (
                    result["core_concepts_found"]
                ),

                "related_concepts_found": (
                    result["related_concepts_found"]
                ),

                "all_ontology_concepts_found": (
                    result["all_ontology_concepts_found"]
                ),

                "relation_evidence": (
                    result["relation_evidence"]
                ),

                "context": result["context"],

                "explanation": result["explanation"],
            },

            # V5 results
            "semantic_analysis": semantic_analysis,

            "semantic_analysis_error": (
                semantic_analysis_error
            ),

            "content_length": len(content),
        }

        yield item

    # =====================================================
    # DOWNLOAD ERROR HANDLING
    # =====================================================

    def handle_error(self, failure):

        request = failure.request

        original_url = request.meta.get(
            "original_url",
            request.url
        )

        self.logger.error(

            "Download failed: %s - %s",

            original_url,

            failure.value,
        )

        yield {

            "url": original_url,

            "status": "DOWNLOAD_FAILED",

            "error": str(failure.value),
        }