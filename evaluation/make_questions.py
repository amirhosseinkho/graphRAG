"""Generate a benchmark question set directly from Hetionet edges.

Each question is produced from a template over one relation type (or a
two-hop composition), and its gold answers are read from the graph, so no
manual labelling is involved.

Usage:
    python evaluation/make_questions.py [--per-template 8] [--seed 42]
"""
import argparse
import json
import os
import random
from collections import defaultdict

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# (template id, metaedge, question template, which side the question names)
# side="source": the question names the edge source and asks for targets.
# side="target": the question names the edge target and asks for sources.
ONE_HOP = [
    ("disease_genes", "DaG", "Which genes are associated with {name}?", "source"),
    ("disease_compounds", "CtD", "Which compounds treat {name}?", "target"),
    ("compound_genes", "CbG", "Which genes does {name} bind?", "source"),
    ("disease_symptoms", "DpS", "What symptoms does {name} present?", "source"),
    ("disease_anatomy", "DlA", "In which anatomical structures does {name} localize?", "source"),
    ("compound_side_effects", "CcSE", "What side effects does {name} cause?", "source"),
]
TWO_HOP_ID = "disease_compound_genes"
TWO_HOP_TEMPLATE = "Which genes are bound by compounds that treat {name}?"


def load(nodes_path, edges_path):
    nodes = pd.read_csv(nodes_path, sep="\t", encoding="utf-8-sig")
    edges = pd.read_csv(edges_path, sep="\t")
    names = dict(zip(nodes["id"], nodes["name"]))
    kinds = dict(zip(nodes["id"], nodes["kind"]))
    return names, kinds, edges


def neighbours(edges, metaedge, side):
    sub = edges[edges["metaedge"] == metaedge]
    key, val = ("source", "target") if side == "source" else ("target", "source")
    out = defaultdict(set)
    for k, v in zip(sub[key], sub[val]):
        out[k].add(v)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nodes", default=os.path.join(ROOT, "hetionet-v1.0-nodes.tsv"))
    ap.add_argument("--edges", default=os.path.join(ROOT, "edges.sif"))
    ap.add_argument("--out", default=os.path.join(ROOT, "evaluation", "questions.jsonl"))
    ap.add_argument("--per-template", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--min-answers", type=int, default=3)
    ap.add_argument("--max-answers", type=int, default=30)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    names, kinds, edges = load(args.nodes, args.edges)
    questions = []

    for tid, metaedge, template, side in ONE_HOP:
        adj = neighbours(edges, metaedge, side)
        pool = sorted(k for k, v in adj.items() if args.min_answers <= len(v) <= args.max_answers)
        for anchor in rng.sample(pool, min(args.per_template, len(pool))):
            gold = sorted(adj[anchor])
            questions.append({
                "id": f"{tid}_{len(questions):03d}",
                "template": tid,
                "hops": 1,
                "question": template.format(name=names[anchor]),
                "anchor_id": anchor,
                "anchor_name": names[anchor],
                "answer_kind": kinds[gold[0]],
                "gold_ids": gold,
                "gold_names": [names[g] for g in gold],
            })

    # Two-hop: Disease <-treats- Compound -binds-> Gene
    treats = neighbours(edges, "CtD", "target")   # disease -> compounds
    binds = neighbours(edges, "CbG", "source")    # compound -> genes
    two_hop = {}
    for disease, compounds in treats.items():
        genes = set().union(*(binds.get(c, set()) for c in compounds))
        if args.min_answers <= len(genes) <= 2 * args.max_answers:
            two_hop[disease] = genes
    for anchor in rng.sample(sorted(two_hop), min(args.per_template, len(two_hop))):
        gold = sorted(two_hop[anchor])
        questions.append({
            "id": f"{TWO_HOP_ID}_{len(questions):03d}",
            "template": TWO_HOP_ID,
            "hops": 2,
            "question": TWO_HOP_TEMPLATE.format(name=names[anchor]),
            "anchor_id": anchor,
            "anchor_name": names[anchor],
            "answer_kind": "Gene",
            "gold_ids": gold,
            "gold_names": [names[g] for g in gold],
        })

    with open(args.out, "w", encoding="utf-8") as f:
        for q in questions:
            f.write(json.dumps(q, ensure_ascii=False) + "\n")
    print(f"Wrote {len(questions)} questions to {args.out}")


if __name__ == "__main__":
    main()
