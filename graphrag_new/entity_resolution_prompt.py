# -*- coding: utf-8 -*-
"""
Entity Resolution Prompts - پرامپت‌های حل موجودیت
"""

ENTITY_RESOLUTION_PROMPTS = {
    "similarity_check": """---Role---
You are a biomedical entity resolution assistant for Hetionet-like KGs.

---Goal---
Given two entity names, decide if they denote the same concept. Return STRICT JSON.

---Normalization rules (apply before matching)---
- Unicode normalize (NFKC), trim, collapse spaces.
- Case-insensitive compare; strip punctuation/parentheses/hyphens if non-semantic.
- Greek letters ↔ latin transliteration (β↔beta), roman numerals ↔ arabic (II↔2).
- Handle common bio variants (e.g., “BRCA-1”→“BRCA1”, "p53 protein"→"TP53" only if context warrants gene vs protein distinction).

---Domain rules---
- Prefer exact ID matches when possible (HGNC/Entrez for genes; UniProt for proteins; DOID/MeSH for diseases; DrugBank/CAS for compounds).
- Do NOT merge class vs instance or related-but-distinct (e.g., “Cancer” vs “Tumor”): set related_but_distinct=true and are_similar=false.
- Gene family vs specific member (e.g., “MAPK” vs “MAPK1”) → not the same.
- Ambiguous acronyms (e.g., “APC”, “ER”, “AR”) → require disambiguation (NIL).
- Consider species when applicable (HGNC is human-specific; include species if non-human or ambiguous).

---Confidence calibration---
- 0.9–1.0: registry-backed synonym/ID match
- 0.6–0.9: strong alias/acronym
- 0.3–0.6: fuzzy lexical/near match
- <0.3: no match

---Output schema (JSON only)---
{
  "are_similar": true|false|null,
  "confidence": 0.0-1.0,
  "reasoning": "brief",
  "suggested_canonical": "preferred surface form or null",
  "related_but_distinct": true|false,
  "ids": {
    "left": {"namespace": "HGNC|Entrez|UniProt|DOID|MeSH|DrugBank|CAS|null", "id": "...", "species": "human|mouse|null"},
    "right": {"namespace": "HGNC|Entrez|UniProt|DOID|MeSH|DrugBank|CAS|null", "id": "...", "species": "human|mouse|null"},
    "canonical": {"namespace": "HGNC|Entrez|UniProt|DOID|MeSH|DrugBank|CAS|null", "id": "...", "species": "human|mouse|null"}
  },
  "need_disambiguation": false|true,
  "NIL": false|true
}

Now analyze:
Entity 1: "{entity1}"
Entity 2: "{entity2}"
Response:""",

    "group_similar_entities": """---Role---
Group a list of biomedical entities into equivalence clusters.

---Rules---
- Apply normalization as in similarity_check.
- Use IDs if available to cluster (HGNC/Entrez for genes; UniProt for proteins; DOID/MeSH for diseases; DrugBank/CAS for compounds).
- Do not merge related-but-distinct concepts; put them in separate clusters with related_but_distinct=true where relevant.
- Detect ambiguous strings and create NIL clusters with need_disambiguation=true.
- Output MUST be valid JSON only.

---Output (JSON only)---
[
  {
    "canonical_name": "preferred",
    "canonical_ids": [{"ns":"HGNC|Entrez|UniProt|DOID|MeSH|DrugBank|CAS", "id":"..."}],
    "entities": ["TP53","p53","TP53 gene"],
    "confidence": 0.95,
    "reasoning": "same HGNC symbol",
    "species": "human",
    "related_but_distinct": false,
    "need_disambiguation": false,
    "NIL": false
  },
  {
    "canonical_name": null,
    "canonical_ids": [],
    "entities": ["APC"],
    "confidence": 0.4,
    "reasoning": "ambiguous acronym (gene vs immune cell)",
    "species": null,
    "related_but_distinct": false,
    "need_disambiguation": true,
    "NIL": true
  }
]

Now group these entities:
{entities}
Response:""",

    "canonical_name_suggestion": """---Role---
Suggest a canonical biomedical name for a group.

---Preference order---
- Exact registry-backed label (HGNC/UniProt/DOID/MeSH/DrugBank).
- Official symbol over colloquial (TP53 over p53).
- Specific over vague; keep species if non-human.
- Keep concise, remove redundant suffixes like "gene" when symbol suffices.
- Output MUST be valid JSON only.

---Output (JSON only)---
{
  "canonical_name": "TP53",
  "canonical_ids": [{"ns":"HGNC","id":"11998"}],
  "confidence": 0.95,
  "reasoning": "official HGNC symbol",
  "species": "human"
}

Now suggest canonical name for:
{entities}
Response:"""
}