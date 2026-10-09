"""Evaluate GraphRAG retrieval against a dense-retrieval RAG baseline.

Retrieval metrics (no LLM needed), computed per question against the gold
answers in evaluation/questions.jsonl:
  - context_recall: share of gold entities whose name or Hetionet id appears in
    the full context text that would be passed to the LLM
  - context_recall_at_budget: the same, but only within the first CONTEXT_BUDGET
    characters of the context, so systems are compared at equal context size
  - context_hit:    1 if at least one gold entity appears in the context
  - node_recall:    (GraphRAG only) share of gold entities among the retrieved
                    graph nodes and edge endpoints, before the context is built
  - context_chars, latency_s
A GraphRAG question that takes longer than QUESTION_TIMEOUT seconds is stopped
and scored as an empty retrieval (recorded in the 'error' column).

Answer metrics are computed only when --llm-model is given and OPENAI_API_KEY
is set: entity-level precision/recall/F1 of the gold names mentioned in the
generated answer, with the same model, prompt and temperature 0 for every system.

Usage:
    python evaluation/run_eval.py --systems graphrag_intelligent,graphrag_bfs,dense_rag
    python evaluation/run_eval.py --summarize-only
"""
import argparse
import contextlib
import csv
import io
import json
import multiprocessing
import os
import re
import subprocess
import sys
import time
from datetime import date

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EVAL_DIR = os.path.join(ROOT, "evaluation")
RAW_DIR = os.path.join(EVAL_DIR, "results", "raw")
CONTEXT_BUDGET = 2000  # characters; equals GraphRAGService.config['max_context_length']
QUESTION_TIMEOUT = 600  # seconds per question; a timeout counts as an empty retrieval

GRAPHRAG_SYSTEMS = {
    "graphrag_intelligent": "INTELLIGENT",
    "graphrag_bfs": "BFS",
    "graphrag_adaptive": "ADAPTIVE",
}
ALL_SYSTEMS = list(GRAPHRAG_SYSTEMS) + ["dense_rag"]

ANSWER_PROMPT = (
    "Answer the biomedical question using only the context. "
    "List the answer entities separated by commas, with no explanation.\n\n"
    "Context:\n{context}\n\nQuestion: {question}\nAnswer:"
)


def mentioned(text, name, node_id=None):
    if node_id and node_id in text:
        return True
    pattern = r"(?<![\w-])" + re.escape(name.lower()) + r"(?![\w-])"
    return re.search(pattern, text.lower()) is not None


def load_questions(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def git_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:
        return "unknown"


def answer_with_llm(model, question, context):
    from openai import OpenAI

    client = OpenAI()
    resp = client.chat.completions.create(
        model=model,
        temperature=0,
        messages=[{"role": "user", "content": ANSWER_PROMPT.format(context=context, question=question)}],
    )
    return resp.choices[0].message.content or ""


def score_answer(answer, q):
    found = [n for n, gid in zip(q["gold_names"], q["gold_ids"]) if mentioned(answer, n, gid)]
    predicted = [p.strip() for p in answer.split(",") if p.strip()]
    recall = len(found) / len(q["gold_ids"])
    precision = len(found) / len(predicted) if predicted else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


def _graphrag_worker(conn, method_name):
    """Child process: load the graph once, then answer retrieval requests."""
    os.chdir(ROOT)  # the service loads its data files relative to the repo root
    sys.path.insert(0, ROOT)
    with contextlib.redirect_stdout(io.StringIO()):
        from graphrag_service import GraphRAGService, RetrievalMethod
        service = GraphRAGService()
    method = RetrievalMethod[method_name]
    conn.send("ready")
    while True:
        question = conn.recv()
        if question is None:
            return
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                r = service.retrieve_information(question, method)
            ids = {n.id for n in r.nodes} | {e.source for e in r.edges} | {e.target for e in r.edges}
            conn.send(("ok", r.context_text, ids, {"nodes": len(r.nodes), "edges": len(r.edges)}))
        except Exception as e:
            conn.send(("error", repr(e)[:300]))


class GraphRAGWorker:
    """Runs GraphRAG retrieval in a child process so a slow question can be stopped
    after QUESTION_TIMEOUT seconds; the worker is then restarted for the next question."""

    def __init__(self, method_name):
        self.method_name = method_name
        self._start()

    def _start(self):
        ctx = multiprocessing.get_context("spawn")
        self.conn, child = ctx.Pipe()
        self.proc = ctx.Process(target=_graphrag_worker, args=(child, self.method_name), daemon=True)
        self.proc.start()
        if self.conn.recv() != "ready":
            raise RuntimeError("GraphRAG worker failed to start")

    def retrieve(self, question):
        self.conn.send(question)
        if not self.conn.poll(QUESTION_TIMEOUT):
            self.proc.terminate()
            self.proc.join()
            self._start()
            raise TimeoutError(f"retrieval exceeded {QUESTION_TIMEOUT} s")
        msg = self.conn.recv()
        if msg[0] == "error":
            raise RuntimeError(msg[1])
        return msg[1], msg[2], msg[3]


def run_system(system, questions, llm_model, out_path):
    done = set()
    if os.path.exists(out_path):
        with open(out_path, encoding="utf-8") as f:
            done = {json.loads(line)["question_id"] for line in f if line.strip()}
    todo = [q for q in questions if q["id"] not in done]
    if not todo:
        print(f"[{system}] all {len(questions)} questions already done")
        return

    sys.path.insert(0, ROOT)
    if system == "dense_rag":
        sys.path.insert(0, EVAL_DIR)
        from baseline_rag import DenseRAG

        rag = DenseRAG()

        def retrieve(question):
            context, n_chunks = rag.retrieve(question, max_chars=CONTEXT_BUDGET)
            return context, None, {"chunks": n_chunks}
    else:
        worker = GraphRAGWorker(GRAPHRAG_SYSTEMS[system])
        retrieve = worker.retrieve

    with open(out_path, "a", encoding="utf-8") as out:
        for i, q in enumerate(todo, 1):
            t0 = time.time()
            try:
                context, ids, extra = retrieve(q["question"])
                error = ""
            except Exception as e:  # keep going; failures and timeouts count as empty retrieval
                context, ids, extra, error = "", set(), {}, repr(e)[:300]
            latency = time.time() - t0
            hits = [mentioned(context, n, gid) for n, gid in zip(q["gold_names"], q["gold_ids"])]
            head = context[:CONTEXT_BUDGET]
            hits_budget = [mentioned(head, n, gid) for n, gid in zip(q["gold_names"], q["gold_ids"])]
            row = {
                "question_id": q["id"],
                "template": q["template"],
                "hops": q["hops"],
                "system": system,
                "n_gold": len(q["gold_ids"]),
                "context_recall": sum(hits) / len(hits),
                "context_hit": int(any(hits)),
                "context_recall_at_budget": sum(hits_budget) / len(hits_budget),
                "node_recall": (len(ids & set(q["gold_ids"])) / len(q["gold_ids"])) if ids is not None else None,
                "context_chars": len(context),
                "latency_s": round(latency, 2),
                "error": error,
                **{f"n_{k}": v for k, v in extra.items()},
            }
            if llm_model and os.environ.get("OPENAI_API_KEY") and not error:
                answer = answer_with_llm(llm_model, q["question"], context)
                row["answer"] = answer
                row["answer_precision"], row["answer_recall"], row["answer_f1"] = score_answer(answer, q)
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
            out.flush()
            print(f"[{system}] {i}/{len(todo)} {q['id']} recall={row['context_recall']:.2f} {latency:.1f}s", flush=True)


def mean(values):
    values = [v for v in values if v is not None]
    return sum(values) / len(values) if values else None


def summarize(questions, systems, args):
    rows = []
    for system in systems:
        path = os.path.join(RAW_DIR, f"{system}.jsonl")
        if os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                rows += [json.loads(line) for line in f if line.strip()]
    qids = {q["id"] for q in questions}
    rows = [r for r in rows if r["question_id"] in qids]
    for r in rows:  # rows written before this metric existed: exact when the context fits the budget
        if "context_recall_at_budget" not in r and r["context_chars"] <= CONTEXT_BUDGET:
            r["context_recall_at_budget"] = r["context_recall"]
    if not rows:
        print("No results to summarize.")
        return

    today = date.today().isoformat()
    results_dir = os.path.join(EVAL_DIR, "results")
    fields = sorted({k for r in rows for k in r}, key=lambda k: (k not in ("question_id", "system"), k))
    with open(os.path.join(results_dir, f"{today}_results.csv"), "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    metrics = ["context_recall_at_budget", "context_recall", "context_hit", "node_recall", "context_chars", "latency_s",
               "answer_precision", "answer_recall", "answer_f1"]
    summary = {
        "run_date": today,
        "commit": git_commit(),
        "n_questions": len(questions),
        "question_seed": args.question_seed,
        "context_budget_chars": CONTEXT_BUDGET,
        "dense_model": "sentence-transformers/all-MiniLM-L6-v2",
        "llm_model": args.llm_model if os.environ.get("OPENAI_API_KEY") else None,
        "systems": {},
    }
    for system in systems:
        srows = [r for r in rows if r["system"] == system]
        if not srows:
            continue
        overall = {m: mean([r.get(m) for r in srows]) for m in metrics}
        overall["n"] = len(srows)
        overall["errors"] = sum(1 for r in srows if r.get("error"))
        by_template = {}
        for t in sorted({r["template"] for r in srows}):
            trows = [r for r in srows if r["template"] == t]
            by_template[t] = {"n": len(trows), "context_recall": mean([r["context_recall"] for r in trows]),
                              "context_recall_at_budget": mean([r["context_recall_at_budget"] for r in trows]),
                              "context_hit": mean([r["context_hit"] for r in trows])}
        summary["systems"][system] = {"overall": overall, "by_template": by_template}
    with open(os.path.join(results_dir, f"{today}_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    def fmt(v, pct=True):
        if v is None:
            return "n/a"
        return f"{100 * v:.1f}%" if pct else f"{v:.0f}" if v >= 10 else f"{v:.1f}" if v >= 1 else f"{v:.2f}"

    lines = [f"| System | n | Recall in first {CONTEXT_BUDGET} chars | Recall in full context | Hit rate (full context) "
             "| Node recall | Mean context (chars) | Mean latency (s) |",
             "|---|---|---|---|---|---|---|---|"]
    for system, s in summary["systems"].items():
        o = s["overall"]
        lines.append(f"| {system} | {o['n']} | {fmt(o['context_recall_at_budget'])} | {fmt(o['context_recall'])} | {fmt(o['context_hit'])} | "
                     f"{fmt(o['node_recall'])} | {fmt(o['context_chars'], False)} | {fmt(o['latency_s'], False)} |")
    templates = sorted({t for s in summary["systems"].values() for t in s["by_template"]})
    lines += ["", f"Recall in the first {CONTEXT_BUDGET} characters, by question template:", "",
              "| Template | " + " | ".join(summary["systems"]) + " |",
              "|---|" + "---|" * len(summary["systems"])]
    for t in templates:
        cells = [fmt(s["by_template"].get(t, {}).get("context_recall_at_budget")) for s in summary["systems"].values()]
        lines.append(f"| {t} | " + " | ".join(cells) + " |")
    table = "\n".join(lines)
    with open(os.path.join(results_dir, f"{today}_summary.md"), "w", encoding="utf-8") as f:
        f.write(table + "\n")
    print(table)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--questions", default=os.path.join(EVAL_DIR, "questions.jsonl"))
    ap.add_argument("--systems", default=",".join(ALL_SYSTEMS))
    ap.add_argument("--limit", type=int, default=None, help="evaluate only the first N questions")
    ap.add_argument("--llm-model", default=None, help="e.g. gpt-4o-mini; needs OPENAI_API_KEY")
    ap.add_argument("--question-seed", type=int, default=42, help="seed used by make_questions.py (recorded only)")
    ap.add_argument("--summarize-only", action="store_true")
    args = ap.parse_args()

    questions = load_questions(args.questions)[: args.limit]
    systems = [s.strip() for s in args.systems.split(",") if s.strip()]
    unknown = set(systems) - set(ALL_SYSTEMS)
    if unknown:
        sys.exit(f"Unknown systems: {sorted(unknown)}; choose from {ALL_SYSTEMS}")
    os.makedirs(RAW_DIR, exist_ok=True)
    if not args.summarize_only:
        for system in systems:
            run_system(system, questions, args.llm_model, os.path.join(RAW_DIR, f"{system}.jsonl"))
    summarize(questions, systems, args)


if __name__ == "__main__":
    main()
