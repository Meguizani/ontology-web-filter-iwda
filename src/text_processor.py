
import re
import spacy


class TextProcessor:
    """
    Linguistic text processing using spaCy.

    Features:
    - Text normalization
    - Sentence segmentation
    - Tokenization
    - Lemmatization
    - POS tagging
    - Noun chunk extraction
    - Named Entity Recognition (NER)
    - Dependency parsing
    """

    def __init__(self, model_name="en_core_web_sm"):

        try:
            self.nlp = spacy.load(model_name)
        except OSError as exc:
            raise RuntimeError(
                f"spaCy model '{model_name}' is not installed. "
                f"Install it using: python -m spacy download {model_name}"
            ) from exc

        print(f"spaCy model loaded: {model_name}")
        print(f"Pipeline components: {self.nlp.pipe_names}")

    @staticmethod
    def clean_text(text):
        """
        Normalize whitespace while preserving sentence boundaries.
        """

        if not isinstance(text, str):
            raise TypeError("Input text must be a string.")

        text = text.replace("\xa0", " ")
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n\s*\n+", "\n\n", text)

        return text.strip()

    def process(self, text):
        """
        Process a complete document and return
        structured linguistic information.
        """

        cleaned_text = self.clean_text(text)

        if not cleaned_text:
            raise ValueError("The input document is empty.")

        # Main spaCy processing
        doc = self.nlp(cleaned_text)

        # --------------------------------------------------
        # 1. SENTENCE SEGMENTATION
        # --------------------------------------------------

        sentences = []

        for sentence_id, sentence in enumerate(doc.sents, start=1):

            sentences.append({
                "sentence_id": sentence_id,
                "text": sentence.text.strip()
            })

        # --------------------------------------------------
        # 2. TOKENIZATION, LEMMATIZATION AND POS TAGGING
        # --------------------------------------------------

        tokens = []

        for token in doc:

            if token.is_space:
                continue

            tokens.append({
                "text": token.text,
                "lemma": token.lemma_,
                "pos": token.pos_,
                "tag": token.tag_,
                "dependency": token.dep_,
                "head": token.head.text,
                "is_stopword": token.is_stop,
                "is_alpha": token.is_alpha
            })

        # --------------------------------------------------
        # 3. NOUN CHUNK EXTRACTION
        # --------------------------------------------------

        noun_chunks = []

        for chunk in doc.noun_chunks:

            noun_chunks.append({
                "text": chunk.text,
                "root": chunk.root.text,
                "root_lemma": chunk.root.lemma_,
                "root_pos": chunk.root.pos_,
                "dependency": chunk.root.dep_
            })

        # --------------------------------------------------
        # 4. NAMED ENTITY RECOGNITION
        # --------------------------------------------------

        entities = []

        for entity in doc.ents:

            entities.append({
                "text": entity.text,
                "label": entity.label_,
                "description": spacy.explain(entity.label_)
            })

        # --------------------------------------------------
        # 5. DEPENDENCY PARSING
        # --------------------------------------------------

        dependencies = []

        for token in doc:

            if token.is_space:
                continue

            if token.dep_ == "ROOT":
                continue

            dependencies.append({
                "dependent": token.text,
                "dependent_lemma": token.lemma_,
                "relation": token.dep_,
                "head": token.head.text,
                "head_lemma": token.head.lemma_
            })

        # --------------------------------------------------
        # 6. LINGUISTIC STATISTICS
        # --------------------------------------------------

        statistics = {
            "character_count": len(cleaned_text),
            "token_count": len(tokens),
            "sentence_count": len(sentences),
            "noun_chunk_count": len(noun_chunks),
            "entity_count": len(entities),
            "dependency_count": len(dependencies)
        }

        # --------------------------------------------------
        # 7. STRUCTURED OUTPUT
        # --------------------------------------------------

        return {
            "text": cleaned_text,
            "statistics": statistics,
            "sentences": sentences,
            "tokens": tokens,
            "noun_chunks": noun_chunks,
            "entities": entities,
            "dependencies": dependencies
        }


# ==========================================================
# TESTING
# ==========================================================

if __name__ == "__main__":

    import json

    processor = TextProcessor()

    sample_text = """
    Financial fraud detection is an important challenge
    for modern financial institutions.

    Machine learning algorithms such as Random Forest
    and XGBoost can identify suspicious transactions.

    Apache Spark supports large-scale data processing.
    SHAP helps explain the predictions of machine learning
    models.
    """

    result = processor.process(sample_text)

    print("\n" + "=" * 65)
    print("TEXT PROCESSING RESULTS")
    print("=" * 65)

    print("\nSTATISTICS:")

    print(json.dumps(
        result["statistics"],
        indent=4,
        ensure_ascii=False
    ))

    print("\n1. SENTENCE SEGMENTATION:")

    for sentence in result["sentences"]:
        print(
            f"[{sentence['sentence_id']}] "
            f"{sentence['text']}"
        )

    print("\n2. TOKENIZATION, LEMMATIZATION AND POS TAGGING:")

    for token in result["tokens"][:20]:
        print(
            f"Text: {token['text']:<15} "
            f"Lemma: {token['lemma']:<15} "
            f"POS: {token['pos']:<10} "
            f"Tag: {token['tag']}"
        )

    print("\n3. NOUN CHUNKS:")

    for chunk in result["noun_chunks"]:
        print(
            f"- {chunk['text']} "
            f"(root: {chunk['root']})"
        )

    print("\n4. NAMED ENTITIES:")

    for entity in result["entities"]:
        print(
            f"- {entity['text']} "
            f"| Label: {entity['label']} "
            f"| Description: {entity['description']}"
        )

    print("\n5. DEPENDENCY PARSING:")

    for dependency in result["dependencies"][:20]:
        print(
            f"- {dependency['dependent']} "
            f"--{dependency['relation']}--> "
            f"{dependency['head']}"
        )

    print("\nPROCESSING COMPLETED SUCCESSFULLY.")
