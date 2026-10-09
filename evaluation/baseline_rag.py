"""Dense-retrieval RAG baseline over a text rendering of Hetionet.

Every node's edges are rendered as short English sentences grouped by
relation type ("hypertension (Disease). Compounds that treat it: A, B, ..."),
split into chunks of at most CHUNK_CHARS characters, and embedded with a
sentence-transformers model. A query is answered by cosine-similarity search
over these chunks, and the top chunks are concatenated until the context
budget is reached.

The chunk embeddings are cached in evaluation/cache/ (not tracked by git).
"""
import hashlib
import os
from collections import defaultdict

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE_DIR = os.path.join(ROOT, "evaluation", "cache")
CHUNK_CHARS = 400

# metaedge -> (phrase used from the source node, phrase used from the target node)
RELATIONS = {
    "AdG": ("downregulates genes", "anatomies that downregulate it"),
    "AeG": ("expresses genes", "anatomies that express it"),
    "AuG": ("upregulates genes", "anatomies that upregulate it"),
    "CbG": ("binds genes", "compounds that bind it"),
    "CcSE": ("causes side effects", "compounds that cause it"),
    "CdG": ("downregulates genes", "compounds that downregulate it"),
    "CpD": ("palliates diseases", "compounds that palliate it"),
    "CrC": ("resembles compounds", "compounds that resemble it"),
    "CtD": ("treats diseases", "compounds that treat it"),
    "CuG": ("upregulates genes", "compounds that upregulate it"),
    "DaG": ("is associated with genes", "diseases associated with it"),
    "DdG": ("downregulates genes", "diseases that downregulate it"),
    "DlA": ("localizes to anatomies", "diseases that localize to it"),
    "DpS": ("presents symptoms", "diseases that present it"),
    "DrD": ("resembles diseases", "diseases that resemble it"),
    "DuG": ("upregulates genes", "diseases that upregulate it"),
    "GcG": ("covaries with genes", "genes that covary with it"),
    "GiG": ("interacts with genes", "genes that interact with it"),
    "GpBP": ("participates in biological processes", "genes that participate in it"),
    "GpCC": ("participates in cellular components", "genes that participate in it"),
    "GpMF": ("participates in molecular functions", "genes that participate in it"),
    "GpPW": ("participates in pathways", "genes that participate in it"),
    "Gr>G": ("regulates genes", "genes that regulate it"),
    "PCiC": ("includes compounds", "pharmacologic classes that include it"),
}


def build_chunks(nodes_path, edges_path):
    nodes = pd.read_csv(nodes_path, sep="\t", encoding="utf-8-sig")
    edges = pd.read_csv(edges_path, sep="\t")
    names = dict(zip(nodes["id"], nodes["name"]))
    kinds = dict(zip(nodes["id"], nodes["kind"]))

    groups = defaultdict(list)  # (node, phrase) -> [neighbour names]
    for s, m, t in zip(edges["source"], edges["metaedge"], edges["target"]):
        fwd, rev = RELATIONS[m]
        groups[(s, fwd)].append(names[t])
        groups[(t, rev)].append(names[s])

    chunks = []
    for (node, phrase), items in groups.items():
        header = f"{names[node]} ({kinds[node]}). {phrase[0].upper()}{phrase[1:]}: "
        current = []
        for item in sorted(items):
            if current and len(header) + len(", ".join(current + [item])) + 1 > CHUNK_CHARS:
                chunks.append(header + ", ".join(current) + ".")
                current = []
            current.append(item)
        if current:
            chunks.append(header + ", ".join(current) + ".")
    return chunks


class DenseRAG:
    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2",
                 nodes_path=None, edges_path=None, batch_size=256):
        from sentence_transformers import SentenceTransformer

        nodes_path = nodes_path or os.path.join(ROOT, "hetionet-v1.0-nodes.tsv")
        edges_path = edges_path or os.path.join(ROOT, "edges.sif")
        self.model_name = model_name
        self.model = SentenceTransformer(model_name, device="cpu")
        self.chunks = build_chunks(nodes_path, edges_path)

        digest = hashlib.sha1(("\n".join(self.chunks) + model_name).encode("utf-8")).hexdigest()[:12]
        os.makedirs(CACHE_DIR, exist_ok=True)
        cache = os.path.join(CACHE_DIR, f"chunk_embeddings_{digest}.npy")
        if os.path.exists(cache):
            self.emb = np.load(cache)
        else:
            self.emb = self.model.encode(self.chunks, batch_size=batch_size, show_progress_bar=True,
                                         normalize_embeddings=True, convert_to_numpy=True).astype(np.float32)
            np.save(cache, self.emb)

    def retrieve(self, query, max_chars=2000, max_chunks=50):
        q = self.model.encode([query], normalize_embeddings=True, convert_to_numpy=True)[0]
        scores = self.emb @ q
        order = np.argsort(-scores)[:max_chunks]
        picked, used = [], 0
        for i in order:
            text = self.chunks[i]
            if used + len(text) + 1 > max_chars:
                break
            picked.append(text)
            used += len(text) + 1
        return "\n".join(picked), len(picked)
