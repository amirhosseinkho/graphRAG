# Licensed under the MIT License
"""
Reference:
 - [LightRag](https://github.com/HKUDS/LightRAG)
 - [MiniRag](https://github.com/HKUDS/MiniRAG)
"""
PROMPTS = {}

PROMPTS["minirag_query2kwd"] = """---Role---
You are an assistant that maps a biomedical question to Hetionet concepts and relations.

---Goal---
Given the user query, return STRICT JSON capturing:
1) the expected answer types (chosen ONLY from the provided Answer type pool),
2) extracted entities from the query,
3) the KG intent and the corresponding Hetionet metaedges allow/deny lists.

---Hetionet cheat-sheet---
MetaNodes: Gene(G), Disease(D), Compound(C), Pathway(PW), Biological Process(BP), Molecular Function(MF), Anatomy(A), Symptom(S), Pharmacologic Class(PC), Cellular Component(CC)
MetaEdges (subset, abbreviations): 
- CtD (Compound–treats–Disease), CpD (Compound–palliates–Disease)
- CbG (Compound–binds–Gene), CuG/CdG (Compound–up/down–regulates–Gene)
- DaG/DuG/DdG (Disease–associates/up/down–regulates–Gene)
- GpPW/GpBP/GpMF/GpCC (Gene–participates–Pathway/Process/Function/Component)
- GiG (Gene–interacts–Gene), Gr>G (Gene>regulates>Gene), GcG (Gene–covaries–Gene)
- DlA (Disease–localizes–Anatomy), DpS (Disease–presents–Symptom)
- PCiC (Class–includes–Compound)
- CrC (Compound–resembles–Compound), DrD (Disease–resembles–Disease)

---Instructions---
- Output MUST be VALID JSON ONLY. No prose.
- "answer_type_keywords": pick ≤3 types PRESENT in the given Answer type pool. Do not invent new types.
- "entities_from_query": list canonical surface strings from the query (e.g., "BRCA1", "breast cancer", "trastuzumab"). Keep ≤10.
- Detect intent and fill "kg_intent" from this finite set:
  ["Gene->Gene(covaries)","Gene->Gene(interacts)","Gene->Gene(regulates)",
   "Disease->Drug","Gene->Drug","Gene->Disease","Disease->Symptom","Disease->Anatomy",
   "Drug->Target","Gene->Pathway","Mixed"]
- Provide "allow_metaedges" and "deny_metaedges" arrays consistent with the intent. Prefer minimal allowlists.
- Provide "must_include_entities": entities that MUST be present in the subgraph (e.g., the gene/drug/disease explicitly named).
- If the query is ambiguous or lacks entities, leave "must_include_entities" empty.

---Examples---
Query: "Which genes covary with BRCA1?"
Output:
{
  "answer_type_keywords": ["Gene"],
  "entities_from_query": ["BRCA1"],
  "kg_intent": "Gene->Gene(covaries)",
  "allow_metaedges": ["GcG"],
  "deny_metaedges": ["GiG","Gr>G","DaG","GpBP","GpPW","AeG","AuG","AdG","CtD","CbG","PCiC","DpS","DlA","CrC","DrD"],
  "must_include_entities": ["BRCA1"]
}

Query: "Which drugs treat breast cancer?"
Output:
{
  "answer_type_keywords": ["Compound","Pharmacologic Class"],
  "entities_from_query": ["breast cancer"],
  "kg_intent": "Disease->Drug",
  "allow_metaedges": ["CtD","CpD","PCiC"],
  "deny_metaedges": ["CbG","DaG","GiG","Gr>G","GcG","GpPW","GpBP","CrC","DrD","DpS","DlA","AeG","AuG","AdG"],
  "must_include_entities": ["breast cancer"]
}

Query: "What is the relationship between AKT3, trastuzumab and breast cancer?"
Output:
{
  "answer_type_keywords": ["Pathway","Compound","Disease","Gene"],
  "entities_from_query": ["AKT3","trastuzumab","breast cancer"],
  "kg_intent": "Mixed",
  "allow_metaedges": ["GpPW","GiG","Gr>G","CbG","CtD","DaG"],
  "deny_metaedges": ["CrC","DrD","DpS","DlA","AeG","AuG","AdG","GpMF","GpCC"],
  "must_include_entities": ["AKT3","trastuzumab","breast cancer"]
}

---Now process the following---
Query: "{query}"
Answer type pool: {TYPE_POOL}

Output:"""