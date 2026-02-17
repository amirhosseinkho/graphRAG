# -*- coding: utf-8 -*-
"""
Graph Prompts - پرامپت‌های استخراج گراف
"""

# Prompts inspired by ragflow and knowledgegraph for high recall extraction
CONTINUE_PROMPT = "MANY entities were missed in the last extraction. Add them below using the same format:\n"
LOOP_PROMPT = "It appears some entities may have still been missed. Answer Y if there are still entities that need to be added, or N if there are none. Please answer with a single letter Y or N.\n"

GRAPH_PROMPTS = {
    "extract_entities": """---Role---
You are an information extraction system that generates a DETAILED knowledge graph from text.
Primary goal: MAXIMIZE RECALL. Do not summarize or select only main facts.

---Goal---
Extract entities and their relationships as STRICT JSON aligned to Hetionet MetaNodes/MetaEdges.

---Hetionet MetaNodes---
Gene (G), Disease (D), Compound (C), Pathway (PW), Biological Process (BP), Molecular Function (MF), Anatomy (A), Symptom (S), Pharmacologic Class (PC), Cellular Component (CC)

---Hetionet MetaEdges (subset)---
- CtD (Compound–treats–Disease), CpD (Compound–palliates–Disease)
- CbG (Compound–binds–Gene), CuG/CdG (Compound–up/down–regulates–Gene)
- DaG/DuG/DdG (Disease–associates/up/down–regulates–Gene)
- GpPW/GpBP/GpMF/GpCC (Gene–participates–Pathway/Process/Function/Component)
- GiG (Gene–interacts–Gene), Gr>G (Gene>regulates>Gene), GcG (Gene–covaries–Gene)
- DlA (Disease–localizes–Anatomy), DpS (Disease–presents–Symptom)
- PCiC (Class–includes–Compound)

---Instructions---
- Extract ALL entities explicitly mentioned, including organizations, people, locations, events, technologies, projects, concepts, risks, regulations, metrics, and domains.
- Also extract meaningful noun phrases as CONCEPT entities (e.g., autonomous delivery drones, machine learning models, distributed cloud platform, ethical concerns).
- Normalize entity names consistently (same entity = same string).
- Extract EVERY explicit relation in the text, not only important ones.
- Create separate relations for actions, attributes, locations, time, quantities, impacts, and stakeholders.
- Relations must be directional and specific, using concise snake_case verb phrases.
- Do NOT collapse multiple facts into one relation.
- Avoid weak relations like 'is' or 'has' unless they express a real property.
- Include years and quantities as separate relations or edge attributes when mentioned.
- Use only facts stated in the text. Do NOT infer or guess.
- Output MUST be VALID JSON ONLY. No prose.
- Entities must use only the MetaNodes above. Prefer canonical surface forms (e.g., "BRCA1", "breast cancer", "trastuzumab").
- Each relationship MUST include a Hetionet metaedge code in field "metaedge".
- If unsure about a relationship, omit it rather than guessing.

---Output Format---
{
  "entities": [
    {"id": "unique_id", "name": "entity_name", "type": "Gene|Disease|Compound|PW|BP|MF|A|S|PC|CC", "attributes": {}}
  ],
  "relationships": [
    {"source": "source_entity_id", "target": "target_entity_id", "metaedge": "GcG|GiG|Gr>G|CtD|...", "relation": "actual_relation_name_from_text", "attributes": {}}
  ]
}

---Example---
Text: "TP53 participates in apoptosis and interacts with BRCA1. Trastuzumab treats breast cancer."

Response: {
  "entities": [
    {"id": "TP53", "name": "TP53", "type": "Gene", "attributes": {}},
    {"id": "BRCA1", "name": "BRCA1", "type": "Gene", "attributes": {}},
    {"id": "apoptosis", "name": "apoptosis", "type": "BP", "attributes": {}},
    {"id": "trastuzumab", "name": "trastuzumab", "type": "Compound", "attributes": {}},
    {"id": "breast cancer", "name": "breast cancer", "type": "Disease", "attributes": {}}
  ],
  "relationships": [
    {"source": "TP53", "target": "apoptosis", "metaedge": "GpBP", "relation": "participates in", "attributes": {}},
    {"source": "TP53", "target": "BRCA1", "metaedge": "GiG", "relation": "interacts with", "attributes": {}},
    {"source": "trastuzumab", "target": "breast cancer", "metaedge": "CtD", "relation": "treats", "attributes": {}}
  ]
}

Now extract from this text:
{text}
Response:""",

    "extract_entities_generic": """---Role---
You are a precise information extraction system that builds a KNOWLEDGE GRAPH from a SINGLE input text.

---Primary Principles---
- ONLY extract entities and relations that are EXPLICITLY mentioned in the text.
- DO NOT use outside knowledge, assumptions, or guesses.
- For every entity and relation you output, there must be clear words or phrases in the text that justify it.
- If you are not sure something is explicitly present in the text, DO NOT include it.

---Entity Types (generic, language-agnostic)---
- PERSON: individual people, characters, named persons.
- LOCATION: cities, countries, physical places, schools, buildings, regions.
- ORGANIZATION: companies, institutions, groups, teams.
- EVENT: concrete events or actions treated as things (meetings, wars, celebrations, accidents).
- OBJECT: physical things (devices, tools, vehicles, products, artifacts).
- CONCEPT: abstract or general concepts, topics, ideas, fields.
- TIME: dates, times, periods, temporal expressions.
- NUMBER: numeric quantities, counts, percentages, measures.
- OTHER: entities that clearly appear in the text but do not fit above types.

---Instructions---
- Work directly in the language of the input text (Persian/Farsi, English, or mixed).
- Normalize entity names consistently; the same real-world entity must have the same \"name\" string.
- Each entity MUST correspond to some span of text (word or phrase) that appears in the input.
- Extract ALL relations that are explicitly stated between entities (subject-verb-object, possession, location, membership, etc.).
- Relation \"relation\" field MUST be a short natural-language phrase from the text (or a very close paraphrase using the same key verb).
- Do NOT invent new entities or relations, even if they are plausible in the real world.
- If the text is very short and mentions only 1–2 things, return only those; never add extra fictional people/objects.

---Output Format (STRICT JSON)---
{
  "entities": [
    {"id": "unique_id", "name": "entity_surface_form_from_text", "type": "PERSON|LOCATION|ORGANIZATION|EVENT|OBJECT|CONCEPT|TIME|NUMBER|OTHER", "attributes": {}}
  ],
  "relationships": [
    {
      "source": "source_entity_id",
      "target": "target_entity_id",
      "relation": "short_phrase_from_text_describing_relation",
      "attributes": {
        "evidence": "exact sentence or phrase from the text that supports this relation",
        "confidence": 0.0-1.0
      }
    }
  ]
}

---Examples---
Text: "Ali went to school."
Possible output:
{
  "entities": [
    {"id": "ali", "name": "Ali", "type": "PERSON", "attributes": {}},
    {"id": "school", "name": "school", "type": "LOCATION", "attributes": {}}
  ],
  "relationships": [
    {
      "source": "ali",
      "target": "school",
      "relation": "went to",
      "attributes": {
        "evidence": "Ali went to school.",
        "confidence": 0.95
      }
    }
  ]
}

---Task---
Now extract entities and relationships from this text as STRICT JSON:
{text}
Response:""",

    "extract_entities_high_recall": """-Goal-
Given a text document that is potentially relevant to this activity and a list of entity types, identify all entities of those types from the text and all relationships among the identified entities.

-Steps-
1. Identify all entities. For each identified entity, extract the following information:
- entity_name: Name of the entity, capitalized, in language of 'Text'
- entity_type: One of the following types: [{entity_types}]
- entity_description: Comprehensive description of the entity's attributes and activities in language of 'Text'
Format each entity as ("entity"{tuple_delimiter}<entity_name>{tuple_delimiter}<entity_type>{tuple_delimiter}<entity_description>

2. From the entities identified in step 1, identify all pairs of (source_entity, target_entity) that are *clearly related* to each other.
For each pair of related entities, extract the following information:
- source_entity: name of the source entity, as identified in step 1
- target_entity: name of the target entity, as identified in step 1
- relationship_description: explanation as to why you think the source entity and the target entity are related to each other in language of 'Text'
- relationship_strength: a numeric score indicating strength of the relationship between the source entity and target entity
 Format each relationship as ("relationship"{tuple_delimiter}<source_entity>{tuple_delimiter}<target_entity>{tuple_delimiter}<relationship_description>{tuple_delimiter}<relationship_strength>)

3. Return output as a single list of all the entities and relationships identified in steps 1 and 2. Use **{record_delimiter}** as the list delimiter.

4. When finished, output {completion_delimiter}

######################
-Real Data-
######################
Entity_types: {entity_types}
Text: {input_text}
######################
Output:""",

    "extract_relationships": """---Role---
You identify Hetionet metaedges between given entities using context.

---Goal---
Given a list of entities, identify the relationships between them based on the context.

-- Output MUST be VALID JSON ONLY.
-- Choose metaedges from: GcG, GiG, Gr>G, CtD, CpD, CbG, CuG, CdG, DaG, DuG, DdG, GpPW, GpBP, GpMF, GpCC, DlA, DpS, PCiC.
-- If relationship is not supported by Hetionet, omit it.

---Output Format---
{
  "relationships": [
    {"source": "source_entity", "target": "target_entity", "metaedge": "GiG|GpBP|CtD|...", "confidence": 0.0-1.0, "evidence": "supporting text"}
  ]
}

---Example---
Entities: ["TP53", "cancer", "cell cycle"]
Context: "TP53 is a tumor suppressor that regulates cell cycle and prevents cancer development."

Response: {
  "relationships": [
    {"source": "TP53", "target": "cell cycle", "metaedge": "GpBP", "confidence": 0.9, "evidence": "TP53 participates in cell cycle regulation"}
  ]
}

Now analyze these entities:
Entities: {entities}
Context: {context}
Response:""",

    "validate_entity": """---Role---
Validate and categorize biomedical entities to Hetionet MetaNodes.

---Goal---
Validate if the given entity is a legitimate biomedical entity and assign the correct type.

---Instructions---
- Check if the entity name is valid
- Assign the most appropriate entity type
- Consider synonyms and alternative names
- Return validation results in JSON format

Use only: Gene, Disease, Compound, Pathway (PW), Biological Process (BP), Molecular Function (MF), Cellular Component (CC), Pharmacologic Class (PC), Anatomy (A), Symptom (S)

---Output Format---
{
  "is_valid": true/false,
  "entity_type": "Gene|Disease|Compound|PW|BP|MF|CC|PC|A|S",
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
Enrich entity information with Hetionet-aligned attributes.

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
  "type": "Gene|Disease|Compound|PW|BP|MF|CC|PC|A|S",
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
Decide if two entities should be merged (equivalence) with strict Hetionet semantics.

---Goal---
Given two entities, determine if they should be merged and provide the canonical representation.

-- Output MUST be VALID JSON ONLY.
-- Do NOT merge related-but-distinct (class vs instance, family vs member).
-- Prefer registry-backed canonical names (HGNC/DOID/DrugBank/UniProt).

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