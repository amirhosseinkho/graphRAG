"""Download Hetionet v1.0 and build the NetworkX graph used by the service.

Files are saved in the repository root, where the code expects them:
  hetionet-v1.0-nodes.tsv   node table (id, name, kind)
  edges.sif                 edge list (source, metaedge, target)
  hetionet_graph.pkl        pickled graph built by rebuild_graph.py

Usage:
    python scripts/download_data.py
"""
import gzip
import os
import shutil
import subprocess
import sys
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE = "https://github.com/hetio/hetionet/raw/main/hetnet/tsv/"
NODES = "hetionet-v1.0-nodes.tsv"
EDGES_GZ = "hetionet-v1.0-edges.sif.gz"


def download(name):
    path = os.path.join(ROOT, name)
    if os.path.exists(path) and os.path.getsize(path) > 1000:
        print(f"{name}: already present")
        return path
    print(f"Downloading {name} ...")
    urllib.request.urlretrieve(BASE + name, path)
    return path


def main():
    download(NODES)
    gz_path = download(EDGES_GZ)
    edges_path = os.path.join(ROOT, "edges.sif")
    if not os.path.exists(edges_path) or os.path.getsize(edges_path) < 1000:
        print("Decompressing edges ...")
        with gzip.open(gz_path, "rb") as src, open(edges_path, "wb") as dst:
            shutil.copyfileobj(src, dst)
    print("Building the graph (this takes a few minutes) ...")
    subprocess.run([sys.executable, "rebuild_graph.py"], cwd=ROOT, check=True)
    if not os.path.exists(os.path.join(ROOT, "hetionet_graph.pkl")):
        sys.exit("hetionet_graph.pkl was not created")
    print("Done: hetionet_graph.pkl is ready.")


if __name__ == "__main__":
    main()
