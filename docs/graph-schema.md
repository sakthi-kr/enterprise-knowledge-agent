# Knowledge graph schema

The local knowledge graph keeps graph structure separate from the vector index. Qdrant stores
retrievable text chunks, while Neo4j stores document/entity relationships that are useful for
multi-hop traversal.

## Nodes

### `Document`

A `Document` node represents one physical source record and is uniquely identified by `record_id`.
The original benchmark `doc_id` is also stored, but it is not used as the graph key because the
source release contains a small number of conflicting records that share a `doc_id`.

Stored properties include the source type, title, source archive, source file, and the amount of
text used during entity extraction.

### `Entity`

An `Entity` node represents a canonical typed entity produced by the local entity-extraction
pipeline and is uniquely identified by `entity_id`.

Properties include the entity type, normalized key, display name, observed aliases, mention count,
document count, source coverage, and maximum extraction confidence.

## Relationships

### `MENTIONS`

`(Document)-[:MENTIONS]->(Entity)` is created only from extracted mentions in that document. The
relationship records the number of mentions, the highest confidence, and the aliases observed in
the document.

### `CO_OCCURS_WITH`

`(Entity)-[:CO_OCCURS_WITH]->(Entity)` records that two canonical entities were mentioned in the
same document. Each pair is stored once in deterministic entity-ID order, and `document_count`
records how many source documents support the relationship.

Co-occurrence is deliberately treated as evidence of association, not as a semantic claim such as
ownership, causation, or dependency. More specific relationship types should only be added when a
separate extraction and evaluation step supports them.
