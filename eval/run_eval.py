"""Score the pipeline on eval/controls.jsonl.

  python eval/run_eval.py --mode retrieval   # free, deterministic: is the evidence in the retrieved facts?
  python eval/run_eval.py --mode llm         # GraphRAG answers from Claude (costs API tokens)
  python eval/run_eval.py --mode baseline    # Claude with NO facts: negative control for the LLM itself

LLM grading is heuristic (citations, yes/no, numbers, "not found") and every answer is written to
eval/results/ for a human to read. Treat the automatic score as a triage aid, not a verdict.
"""
import argparse
import json
import re
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from ehr_graphrag.build_graph import load_graph  # noqa: E402
from ehr_graphrag.cli import load_config  # noqa: E402
from ehr_graphrag.retrieve import UUID_RE, GraphRetriever, Retrieval  # noqa: E402

NOT_FOUND = re.compile(r"not found|no record|does not exist|doesn't exist|no such|not in the record", re.I)


def score_retrieval(item, r):
    ctx, exp, kind = r.context(), item["expect"], item["type"]
    if kind == "unknown_patient":
        return "does not exist" in ctx
    if kind == "fake_concept":
        return not r.concepts
    if kind == "absent_condition":
        pid = UUID_RE.search(item["question"]).group().lower()
        return f"no record links patient:{pid} to" in ctx and f"[{exp['nodes'][0]}]" in ctx
    found = all(f"[{n}]" in ctx for n in exp["nodes"])
    if kind == "cohort_count":
        found = found and f"{exp['answer']} patient(s) in the graph have a record" in ctx
    return found


def score_answer(item, r, text):
    exp, kind = item["expect"], item["type"]
    if item["polarity"] == "negative" and exp.get("answer") in ("not found", "no"):
        said_no = bool(NOT_FOUND.search(text)) or re.match(r"\W*no\b", text, re.I) is not None
        return said_no
    if exp.get("answer") == "yes":
        return re.match(r"\W*yes\b", text, re.I) is not None
    if kind == "cohort_count":
        return re.search(rf"\b{exp['answer']}\b", text) is not None
    # Node-based items: the facts the answer cites must contain every expected node.
    cited = {int(n) for group in re.findall(r"\[F[\d,\sF]+\]", text) for n in re.findall(r"\d+", group)}
    cited_text = " ".join(r.facts[i - 1] for i in cited if 0 < i <= len(r.facts))
    if exp.get("names") and kind == "medication_reason":
        return all(name.lower() in text.lower() for name in exp["names"])
    return all(f"[{n}]" in cited_text for n in exp["nodes"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["retrieval", "llm", "baseline"], default="retrieval")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    cfg = load_config()
    retriever = GraphRetriever(load_graph(cfg["graph_path"]), **cfg["retrieval"])
    items = [json.loads(l) for l in (ROOT / "eval" / "controls.jsonl").read_text().splitlines()][: args.limit]
    if args.mode != "retrieval":
        from ehr_graphrag.answer import answer, assert_synthetic
        assert_synthetic(cfg["data_dir"])

    results, tally = [], defaultdict(lambda: [0, 0])
    for item in items:
        r = retriever.retrieve(item["question"])
        if args.mode == "retrieval":
            text, ok = None, score_retrieval(item, r)
        else:
            if args.mode == "baseline":
                r = Retrieval([], [], [])
            text = answer(item["question"], r, **cfg["llm"])
            ok = score_answer(item, r, text)
        tally[item["type"]][0] += ok
        tally[item["type"]][1] += 1
        results.append({**item, "linked_patients": r.patients, "linked_concepts": r.concepts,
                        "n_facts": len(r.facts), "answer": text, "pass": ok})
        print(f"{'PASS' if ok else 'FAIL'}  {item['id']:<24} {item['question'][:90]}")

    print(f"\n{'type':<20} {'polarity':<9} pass")
    polarity = {i["type"]: i["polarity"] for i in items}
    for kind, (ok, n) in tally.items():
        print(f"{kind:<20} {polarity[kind]:<9} {ok}/{n}")
    out = ROOT / "eval" / "results" / f"{args.mode}-{datetime.now():%Y%m%d-%H%M%S}.jsonl"
    out.parent.mkdir(exist_ok=True)
    out.write_text("".join(json.dumps(x) + "\n" for x in results))
    print(f"\nDetails: {out}")


if __name__ == "__main__":
    main()
