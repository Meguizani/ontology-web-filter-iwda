"""Build RDF exports from IWDA crawl results and the project ontology.

Usage from project root:
    python -m src.rdf_exporter

The graph keeps ontology assertions, document metadata, classifier outputs,
concept mentions, and textual co-occurrence evidence distinguishable.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any

from rdflib import Graph, Literal, Namespace, URIRef
from rdflib.namespace import RDF, RDFS, XSD

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CRAWL = PROJECT_ROOT / "data" / "output" / "classification_results_v5.json"
DEFAULT_ONTOLOGY = PROJECT_ROOT / "data" / "ontology" / "ali_knowledge_graph_reorganized_v2.ont"
DEFAULT_OUT = PROJECT_ROOT / "data" / "output"
IWDA = Namespace("https://w3id.org/iwda/vocabulary#")
DOC = Namespace("https://w3id.org/iwda/document/")


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def safe_local(value: Any) -> str:
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value).strip())
    return value or "unnamed"


def node_uri(node: dict[str, Any]) -> URIRef:
    namespace = str(node.get("namespace") or "https://w3id.org/iwda/ontology#")
    if not namespace.endswith(('#', '/', ':')):
        namespace += "#"
    return URIRef(namespace + "node/" + safe_local(node.get("id", node.get("name", "unknown"))))


def get_path(data: dict[str, Any], *path: str) -> Any:
    current: Any = data
    for key in path:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def doc_uri(record: dict[str, Any], index: int) -> URIRef:
    identity = str(record.get("url") or record.get("final_url") or f"record-{index}")
    digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:24]
    return URIRef(str(DOC) + digest)


def build_graph(crawl_records: list[dict[str, Any]], ontology: dict[str, Any]) -> Graph:
    graph = Graph()
    graph.bind("iwda", IWDA)
    graph.bind("doc", DOC)
    graph.bind("rdf", RDF)
    graph.bind("rdfs", RDFS)

    nodes = ontology.get("nodes", []) if isinstance(ontology, dict) else []
    relations = ontology.get("relations", []) if isinstance(ontology, dict) else []
    nodes_by_id: dict[str, dict[str, Any]] = {}

    # Export the original ontology structure without changing its assertions.
    for node in nodes:
        if not isinstance(node, dict) or node.get("id") is None:
            continue
        nodes_by_id[str(node["id"])] = node
        resource = node_uri(node)
        graph.add((resource, RDF.type, IWDA.OntologyConcept))
        graph.add((resource, RDFS.label, Literal(str(node.get("name", "")), lang="en")))
        graph.add((resource, IWDA.ontologyNodeId, Literal(str(node["id"]))))

    for relation in relations:
        if not isinstance(relation, dict):
            continue
        source = nodes_by_id.get(str(relation.get("source_node_id")))
        target = nodes_by_id.get(str(relation.get("destination_node_id")))
        if not source or not target or not relation.get("name"):
            continue
        namespace = str(relation.get("namespace") or source.get("namespace") or "https://w3id.org/iwda/ontology#")
        if not namespace.endswith(('#', '/', ':')):
            namespace += "#"
        predicate = URIRef(namespace + safe_local(relation["name"]))
        graph.add((node_uri(source), predicate, node_uri(target)))

    for index, record in enumerate(crawl_records):
        if not isinstance(record, dict):
            continue
        document = doc_uri(record, index)
        graph.add((document, RDF.type, IWDA.WebDocument))
        source_url = record.get("url") or record.get("final_url")
        if source_url:
            graph.add((document, IWDA.url, Literal(str(source_url), datatype=XSD.anyURI)))
        if record.get("final_url"):
            graph.add((document, IWDA.finalUrl, Literal(str(record["final_url"]), datatype=XSD.anyURI)))
        if record.get("title"):
            graph.add((document, RDFS.label, Literal(str(record["title"]))))
        if record.get("status"):
            graph.add((document, IWDA.crawlStatus, Literal(str(record["status"]).upper())))
        if record.get("http_status") is not None:
            graph.add((document, IWDA.httpStatusCode, Literal(int(record["http_status"]), datatype=XSD.integer)))
        if record.get("content_length") is not None:
            graph.add((document, IWDA.contentLength, Literal(int(record["content_length"]), datatype=XSD.integer)))

        analysis = record.get("v4_analysis") or {}
        semantic = record.get("semantic_analysis") or {}
        classification = analysis.get("classification")
        relevance = analysis.get("relevance_level")
        semantic_classification = semantic.get("classification") or {}
        if not classification and isinstance(semantic_classification, dict):
            classification = semantic_classification.get("classification")
        if not relevance and isinstance(semantic_classification, dict):
            relevance = semantic_classification.get("relevance_level")
        if classification:
            graph.add((document, IWDA.classification, Literal(str(classification).upper())))
        if relevance:
            graph.add((document, IWDA.relevanceLevel, Literal(str(relevance).upper())))
        if analysis.get("score") is not None:
            try:
                graph.add((document, IWDA.relevanceScore, Literal(float(analysis["score"]), datatype=XSD.decimal)))
            except (TypeError, ValueError):
                pass

        # Merge only explicit concept detections from the classifier and semantic analyzer.
        concept_candidates = []
        for candidate_list in (
            analysis.get("all_ontology_concepts_found", []),
            get_path(semantic, "ontology_analysis", "detected_concepts") or [],
            get_path(semantic, "classification", "detected_concepts") or [],
        ):
            if isinstance(candidate_list, list):
                concept_candidates.extend(candidate_list)
        seen_concepts: set[str] = set()
        for candidate in concept_candidates:
            if not isinstance(candidate, dict) or candidate.get("id") is None:
                continue
            concept_id = str(candidate["id"])
            ontology_node = nodes_by_id.get(concept_id)
            if not ontology_node or concept_id in seen_concepts:
                continue
            seen_concepts.add(concept_id)
            graph.add((document, IWDA.mentionsConcept, node_uri(ontology_node)))

        # Store classifier context indicators as literal evidence, not ontology concepts.
        context = analysis.get("context") or {}
        if isinstance(context, dict):
            for context_group, values in context.items():
                if isinstance(values, list):
                    for value in values:
                        graph.add((document, IWDA.contextIndicator, Literal(str(value))))

        dimensions = analysis.get("research_dimensions") or {}
        if isinstance(dimensions, dict):
            for dimension, result in dimensions.items():
                if isinstance(result, dict) and result.get("detected") is True:
                    graph.add((document, IWDA.detectedResearchDimension, Literal(str(dimension))))

        # Preserve co-occurrence as evidence metadata. Do not assert it as an ontology predicate.
        relation_evidence = analysis.get("relation_evidence") or []
        for evidence_index, evidence in enumerate(relation_evidence):
            if not isinstance(evidence, dict):
                continue
            evidence_uri = URIRef(str(document) + "/evidence/" + str(evidence_index + 1))
            graph.add((document, IWDA.hasRelationEvidence, evidence_uri))
            graph.add((evidence_uri, RDF.type, IWDA.RelationEvidence))
            for key, predicate in (("source", IWDA.evidenceSourceLabel), ("relation", IWDA.evidenceRelationLabel),
                                   ("target", IWDA.evidenceTargetLabel), ("evidence_type", IWDA.evidenceType),
                                   ("evidence_sentence", IWDA.evidenceSentence)):
                if evidence.get(key):
                    graph.add((evidence_uri, predicate, Literal(str(evidence[key]))))

    return graph


def export(crawl_path: Path = DEFAULT_CRAWL, ontology_path: Path = DEFAULT_ONTOLOGY,
           output_dir: Path = DEFAULT_OUT) -> tuple[Path, Path, int]:
    records = read_json(crawl_path)
    if isinstance(records, dict):
        records = records.get("items", [records])
    if not isinstance(records, list):
        raise ValueError("The crawl export must be a JSON list or an object containing an 'items' list.")
    ontology = read_json(ontology_path)
    graph = build_graph(records, ontology)
    output_dir.mkdir(parents=True, exist_ok=True)
    ttl_path = output_dir / "iwda_results.ttl"
    rdf_path = output_dir / "iwda_results.rdf"
    graph.serialize(destination=str(ttl_path), format="turtle", encoding="utf-8")
    graph.serialize(destination=str(rdf_path), format="xml", encoding="utf-8")
    return ttl_path, rdf_path, len(graph)


def main() -> None:
    parser = argparse.ArgumentParser(description="Export IWDA crawl and ontology data to RDF.")
    parser.add_argument("--crawl", type=Path, default=DEFAULT_CRAWL)
    parser.add_argument("--ontology", type=Path, default=DEFAULT_ONTOLOGY)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    ttl, rdf, count = export(args.crawl, args.ontology, args.output_dir)
    print(f"RDF export completed: {count} triples")
    print(f"Turtle: {ttl}")
    print(f"RDF/XML: {rdf}")


if __name__ == "__main__":
    main()
