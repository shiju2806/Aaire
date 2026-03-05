"""
Knowledge Graph — ES-backed entity-relationship graph for RAG disambiguation.

Stores entity nodes and their relationships in Elasticsearch.
At query time, resolves entities in the query to graph nodes, traverses the
graph to find connected chunk IDs, and merges them with vector search results.

Usage:
    from src.knowledge_graph import GraphStore, RelationshipExtractor

    graph = GraphStore(es_client)
    extractor = RelationshipExtractor(llm_provider, graph)
"""

from .graph_store import GraphStore, EntityNode, EntityRelationship
from .relationship_extractor import RelationshipExtractor
from .audit import RetrievalAudit

__all__ = [
    "GraphStore",
    "EntityNode",
    "EntityRelationship",
    "RelationshipExtractor",
    "RetrievalAudit",
]
