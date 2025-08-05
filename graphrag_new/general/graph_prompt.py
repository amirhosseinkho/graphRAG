# -*- coding: utf-8 -*-
"""
Graph Prompts - پرامپت‌های استخراج گراف
"""

GRAPH_PROMPTS = {
    "extract_entities": """---Role---
You are an expert at extracting entities and relationships from text to build a knowledge graph.

---Goal---
Extract entities and their relationships from the given text to create a structured knowledge graph.

---Instructions---
- Identify entities (nodes) and their types
- Identify relationships (edges) between entities
- Use standard entity types: Gene, Disease, Drug, Protein, Pathway, Cell, Tissue, Organ, Process, Function
- Use standard relationship types: regulates, interacts_with, causes, treats, expressed_in, part_of, participates_in
- Return results in JSON format

---Output Format---
{
  "entities": [
    {
      "id": "unique_id",
      "name": "entity_name",
      "type": "entity_type",
      "attributes": {}
    }
  ],
  "relationships": [
    {
      "source": "source_entity_id",
      "target": "target_entity_id", 
      "type": "relationship_type",
      "attributes": {}
    }
  ]
}

---Example---
Text: "TP53 is a tumor suppressor gene that regulates cell cycle and apoptosis. Mutations in TP53 can cause cancer."

Response: {
  "entities": [
    {"id": "TP53", "name": "TP53", "type": "Gene", "attributes": {"function": "tumor suppressor"}},
    {"id": "cancer", "name": "cancer", "type": "Disease", "attributes": {}},
    {"id": "cell_cycle", "name": "cell cycle", "type": "Process", "attributes": {}},
    {"id": "apoptosis", "name": "apoptosis", "type": "Process", "attributes": {}}
  ],
  "relationships": [
    {"source": "TP53", "target": "cell_cycle", "type": "regulates", "attributes": {}},
    {"source": "TP53", "target": "apoptosis", "type": "regulates", "attributes": {}},
    {"source": "TP53", "target": "cancer", "type": "causes", "attributes": {"condition": "mutation"}}
  ]
}

Now extract from this text:
{text}
Response:""",

    "extract_relationships": """---Role---
You are an expert at identifying relationships between entities in biomedical text.

---Goal---
Given a list of entities, identify the relationships between them based on the context.

---Instructions---
- Analyze the relationships between the given entities
- Use standard relationship types: regulates, interacts_with, causes, treats, expressed_in, part_of, participates_in
- Consider the context and domain knowledge
- Return relationships in JSON format

---Output Format---
{
  "relationships": [
    {
      "source": "source_entity",
      "target": "target_entity",
      "type": "relationship_type", 
      "confidence": 0.0-1.0,
      "evidence": "supporting text"
    }
  ]
}

---Example---
Entities: ["TP53", "cancer", "cell cycle"]
Context: "TP53 is a tumor suppressor that regulates cell cycle and prevents cancer development."

Response: {
  "relationships": [
    {
      "source": "TP53",
      "target": "cell cycle", 
      "type": "regulates",
      "confidence": 0.9,
      "evidence": "TP53 regulates cell cycle"
    },
    {
      "source": "TP53",
      "target": "cancer",
      "type": "prevents", 
      "confidence": 0.8,
      "evidence": "prevents cancer development"
    }
  ]
}

Now analyze these entities:
Entities: {entities}
Context: {context}
Response:""",

    "validate_entity": """---Role---
You are an expert at validating and categorizing biomedical entities.

---Goal---
Validate if the given entity is a legitimate biomedical entity and assign the correct type.

---Instructions---
- Check if the entity name is valid
- Assign the most appropriate entity type
- Consider synonyms and alternative names
- Return validation results in JSON format

---Entity Types---
- Gene: DNA sequences that code for proteins
- Disease: medical conditions or disorders  
- Drug: pharmaceutical compounds
- Protein: biological molecules
- Pathway: biological processes
- Cell: cellular components
- Tissue: biological tissues
- Organ: body organs
- Process: biological processes
- Function: biological functions

---Output Format---
{
  "is_valid": true/false,
  "entity_type": "type",
  "confidence": 0.0-1.0,
  "suggested_name": "standardized_name",
  "reasoning": "explanation"
}

---Examples---
Entity: "TP53"
Response: {"is_valid": true, "entity_type": "Gene", "confidence": 0.95, "suggested_name": "TP53", "reasoning": "Valid tumor suppressor gene"}

Entity: "xyz123"
Response: {"is_valid": false, "entity_type": null, "confidence": 0.1, "suggested_name": null, "reasoning": "Not a recognized biomedical entity"}

Now validate:
Entity: {entity}
Response:""",

    "enrich_entity": """---Role---
You are an expert at enriching entity information with additional attributes and context.

---Goal---
Given an entity, provide additional information and attributes to enrich the knowledge graph.

---Instructions---
- Add relevant attributes based on entity type
- Include synonyms and alternative names
- Add functional descriptions
- Consider domain-specific information
- Return enriched information in JSON format

---Output Format---
{
  "entity_id": "id",
  "name": "name",
  "type": "type",
  "attributes": {
    "synonyms": ["alt_name1", "alt_name2"],
    "description": "functional description",
    "function": "biological function",
    "location": "cellular/tissue location",
    "diseases": ["related_disease1", "related_disease2"],
    "pathways": ["related_pathway1", "related_pathway2"]
  }
}

---Example---
Entity: "TP53"
Type: "Gene"

Response: {
  "entity_id": "TP53",
  "name": "TP53", 
  "type": "Gene",
  "attributes": {
    "synonyms": ["p53", "TP53 gene", "tumor protein p53"],
    "description": "Tumor suppressor protein that regulates cell cycle and apoptosis",
    "function": "Cell cycle regulation, DNA repair, apoptosis",
    "location": "Nucleus",
    "diseases": ["cancer", "Li-Fraumeni syndrome"],
    "pathways": ["p53 signaling pathway", "cell cycle", "apoptosis"]
  }
}

Now enrich:
Entity: {entity}
Type: {entity_type}
Response:""",

    "merge_entities": """---Role---
You are an expert at merging similar entities in a knowledge graph.

---Goal---
Given two entities, determine if they should be merged and provide the canonical representation.

---Instructions---
- Compare entities for similarity
- Consider synonyms, abbreviations, and alternative names
- Determine the canonical (preferred) name
- Merge attributes appropriately
- Return merge decision in JSON format

---Output Format---
{
  "should_merge": true/false,
  "canonical_name": "preferred_name",
  "confidence": 0.0-1.0,
  "reasoning": "explanation",
  "merged_attributes": {
    "synonyms": ["all_synonyms"],
    "description": "merged_description",
    "attributes": {}
  }
}

---Examples---
Entity 1: "TP53"
Entity 2: "p53"

Response: {
  "should_merge": true,
  "canonical_name": "TP53",
  "confidence": 0.95,
  "reasoning": "TP53 and p53 are the same gene, TP53 is the official name",
  "merged_attributes": {
    "synonyms": ["p53", "TP53 gene"],
    "description": "Tumor suppressor protein",
    "attributes": {}
  }
}

Now analyze:
Entity 1: {entity1}
Entity 2: {entity2}
Response:"""
} 