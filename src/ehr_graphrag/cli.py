"""Command line entry point.

  ehr-graphrag build                 # CSV -> graph pickle
  ehr-graphrag stats                 # node / edge counts
  ehr-graphrag ask "question" [--no-llm]
"""
import argparse
from collections import Counter
from pathlib import Path

import yaml

from .build_graph import build_graph, load_graph, save_graph
from .retrieve import GraphRetriever

ROOT = Path(__file__).resolve().parents[2]


def load_config(path=ROOT / "config.yaml"):
    cfg = yaml.safe_load(Path(path).read_text())
    cfg["data_dir"] = ROOT / cfg["data_dir"]
    cfg["graph_path"] = ROOT / cfg["graph_path"]
    return cfg


def main():
    ap = argparse.ArgumentParser(prog="ehr-graphrag")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("build")
    sub.add_parser("stats")
    ask = sub.add_parser("ask")
    ask.add_argument("question")
    ask.add_argument("--no-llm", action="store_true", help="print retrieved facts only")
    args = ap.parse_args()
    cfg = load_config()

    if args.cmd == "build":
        G = build_graph(cfg["data_dir"])
        save_graph(G, cfg["graph_path"])
        print(f"Saved graph to {cfg['graph_path']}: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
        return

    G = load_graph(cfg["graph_path"])
    if args.cmd == "stats":
        print("Nodes:", dict(Counter(d["kind"] for _, d in G.nodes(data=True))))
        print("Edges:", dict(Counter(e["rel"] for *_, e in G.edges(data=True))))
        return

    retriever = GraphRetriever(G, **cfg["retrieval"])
    r = retriever.retrieve(args.question)
    print(f"Linked patients: {r.patients}  missing: {r.missing_patients}")
    print(f"Linked concepts: {r.concepts}\n")
    if args.no_llm:
        print(r.context())
        return

    from .answer import answer, assert_synthetic
    assert_synthetic(cfg["data_dir"])
    print(answer(args.question, r, **cfg["llm"]))


if __name__ == "__main__":
    main()
