#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test script for new text generation styles
"""

from graphrag_service import GraphRAGService, RetrievalMethod, GenerationModel
from dataclasses import dataclass
from typing import List

@dataclass
class GraphNode:
    id: str
    name: str
    kind: str
    depth: int = 0
    score: float = 1.0

@dataclass
class GraphEdge:
    source: str
    target: str
    relation: str
    weight: float = 1.0

@dataclass
class RetrievalResult:
    nodes: List[GraphNode]
    edges: List[GraphEdge]
    paths: List[List[str]]
    context_text: str
    method: str
    query: str

def test_new_text_styles():
    """Test the new text generation styles"""
    
    # Create a sample retrieval result
    nodes = [
        GraphNode("Metformin", "Metformin", "Drug"),
        GraphNode("Type 2 Diabetes", "Type 2 Diabetes", "Disease"),
        GraphNode("TCF7L2", "TCF7L2", "Gene"),
        GraphNode("Insulin Signaling", "Insulin Signaling", "Pathway"),
        GraphNode("Alzheimer's Disease", "Alzheimer's Disease", "Disease")
    ]
    
    edges = [
        GraphEdge("Metformin", "Type 2 Diabetes", "treats"),
        GraphEdge("Type 2 Diabetes", "TCF7L2", "associated_with"),
        GraphEdge("TCF7L2", "Insulin Signaling", "participates_in"),
        GraphEdge("Insulin Signaling", "Alzheimer's Disease", "associated_with")
    ]
    
    paths = [
        ["Metformin", "Type 2 Diabetes", "TCF7L2", "Insulin Signaling", "Alzheimer's Disease"]
    ]
    
    retrieval_result = RetrievalResult(
        nodes=nodes,
        edges=edges,
        paths=paths,
        context_text="",
        method="BFS",
        query="آیا داروی Metformin می‌تواند در درمان بیماری Alzheimer مؤثر باشد؟"
    )
    
    # Create service instance
    service = GraphRAGService()
    
    # Test each new text generation style
    styles = [
        'SCIENTIFIC_ANALYTICAL',
        'NARRATIVE_DESCRIPTIVE', 
        'DATA_DRIVEN',
        'STEP_BY_STEP',
        'CONCISE_DIRECT'
    ]
    
    print("🧪 Testing new text generation styles...")
    print("=" * 60)
    
    for style in styles:
        print(f"\n📝 Style: {style}")
        print("-" * 40)
        
        # Generate context using the style
        if style == 'SCIENTIFIC_ANALYTICAL':
            context = service._create_scientific_analytical_context(retrieval_result)
        elif style == 'NARRATIVE_DESCRIPTIVE':
            context = service._create_narrative_context(retrieval_result)
        elif style == 'DATA_DRIVEN':
            context = service._create_data_driven_context(retrieval_result)
        elif style == 'STEP_BY_STEP':
            context = service._create_step_by_step_context(retrieval_result)
        elif style == 'CONCISE_DIRECT':
            context = service._create_compact_direct_context(retrieval_result)
        
        print(context)
        print("\n" + "="*60)

if __name__ == "__main__":
    test_new_text_styles() 