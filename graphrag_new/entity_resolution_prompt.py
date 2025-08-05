# -*- coding: utf-8 -*-
"""
Entity Resolution Prompts - پرامپت‌های حل موجودیت
"""

ENTITY_RESOLUTION_PROMPTS = {
    "similarity_check": """---Role---
You are an expert entity resolution assistant that determines if two entities are the same or similar.

---Goal---
Given two entity names, determine if they represent the same or very similar concepts.

---Instructions---
- Compare the entities for:
  * Exact matches (case-insensitive)
  * Abbreviations and acronyms
  * Synonyms and alternative names
  * Common variations in spelling
  * Scientific vs common names
- Return a JSON response with:
  * "are_similar": true/false
  * "confidence": 0.0-1.0
  * "reasoning": brief explanation
  * "suggested_canonical": the preferred name if similar

---Examples---
Entity 1: "TP53"
Entity 2: "p53"
Response: {"are_similar": true, "confidence": 0.95, "reasoning": "TP53 and p53 are the same gene, p53 is the common abbreviation", "suggested_canonical": "TP53"}

Entity 1: "Cancer"
Entity 2: "Tumor"
Response: {"are_similar": true, "confidence": 0.8, "reasoning": "Cancer and tumor are related but not identical concepts", "suggested_canonical": "Cancer"}

Entity 1: "Insulin"
Entity 2: "Glucose"
Response: {"are_similar": false, "confidence": 0.1, "reasoning": "Insulin and glucose are different molecules", "suggested_canonical": null}

Now analyze:
Entity 1: "{entity1}"
Entity 2: "{entity2}"
Response:""",

    "group_similar_entities": """---Role---
You are an expert at grouping similar entities together.

---Goal---
Given a list of entities, group them by similarity and suggest canonical names.

---Instructions---
- Group entities that represent the same or very similar concepts
- For each group, suggest a canonical (preferred) name
- Consider:
  * Exact matches and abbreviations
  * Synonyms and alternative names
  * Scientific vs common names
  * Related but distinct concepts should be separate groups

---Output Format---
Return a JSON array of groups:
[
  {
    "canonical_name": "preferred name",
    "entities": ["entity1", "entity2", "entity3"],
    "confidence": 0.9,
    "reasoning": "brief explanation"
  }
]

---Example---
Entities: ["TP53", "p53", "TP53 gene", "BRCA1", "BRCA-1", "Insulin", "Insulin hormone"]
Response: [
  {
    "canonical_name": "TP53",
    "entities": ["TP53", "p53", "TP53 gene"],
    "confidence": 0.95,
    "reasoning": "All refer to the same gene"
  },
  {
    "canonical_name": "BRCA1", 
    "entities": ["BRCA1", "BRCA-1"],
    "confidence": 0.9,
    "reasoning": "Same gene with different formatting"
  },
  {
    "canonical_name": "Insulin",
    "entities": ["Insulin", "Insulin hormone"],
    "confidence": 0.85,
    "reasoning": "Same protein with descriptive suffix"
  }
]

Now group these entities:
{entities}
Response:""",

    "canonical_name_suggestion": """---Role---
You are an expert at suggesting canonical names for groups of similar entities.

---Goal---
Given a group of similar entities, suggest the best canonical name.

---Instructions---
- Choose the most standard, widely-used name
- Prefer:
  * Official gene/protein names (e.g., TP53 over p53)
  * Full names over abbreviations when equally common
  * Scientific names over common names when appropriate
  * Most descriptive name that's still concise
- Consider the domain context (biology, medicine, etc.)

---Examples---
Entities: ["TP53", "p53", "TP53 gene"]
Canonical: "TP53" (official gene name)

Entities: ["Cancer", "Malignant tumor", "Neoplasm"]
Canonical: "Cancer" (most common term)

Entities: ["Insulin", "Insulin hormone", "INS"]
Canonical: "Insulin" (standard protein name)

Now suggest canonical name for:
{entities}
Canonical:"""
} 