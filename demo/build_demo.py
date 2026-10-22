"""Run the real pipeline on a few questions and bake the results into the walkthrough page.

  python demo/build_demo.py [out.html]   ->  demo/graphrag_walkthrough.html by default
"""
import glob
import json
import re
import sys
from collections import Counter
from datetime import date
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
from ehr_graphrag.answer import SYSTEM_PROMPT
from ehr_graphrag.build_graph import PATIENT_COLS, clean_name, load_graph
from ehr_graphrag.cli import load_config
from ehr_graphrag.retrieve import GraphRetriever

cfg = load_config()
G = load_graph(cfg["graph_path"])
R = GraphRetriever(G, **cfg["retrieval"])
DATA = cfg["data_dir"]
PID = "1d1330ae-1be0-09c5-040e-dc59176bf0d5"
FAKE = json.loads((ROOT / "eval/controls.jsonl").read_text().splitlines()[25])["question"]

QUESTIONS = [
    ("patient", "One patient's record", f"What active disorders does patient {PID} have?"),
    ("knowledge", "Cross-patient knowledge", "Which medications are used to treat Acute bronchitis?"),
    ("cohort", "Cohort count", "How many patients have a record of Hyperlipidemia?"),
    ("absent", "Negative control: absent condition", f"Does patient {PID} have Essential hypertension?"),
    ("unknown", "Negative control: unknown patient", FAKE),
    ("fake", "Known failure: invented disease", "How many patients have Zorblaxian fever?"),
]
NODE_RE = re.compile(r"(patient|condition|medication|procedure|allergen|vaccine|lab|encounter):[\w.-]+")


def node(n, **extra):
    d = G.nodes[n]
    return {"id": n, "kind": d["kind"], "name": d.get("name") or n.split(":")[1][:8], **extra}


def view(r):
    nodes, edges = {}, []

    def add(n, **extra):
        if n not in nodes:
            nodes[n] = node(n, **extra)

    for p in r.patients:
        add(p, center=True)
        enc = 0
        for _, t, e in G.out_edges(p, data=True):
            if e["rel"] == "HAS_ENCOUNTER":
                enc += 1
                continue
            add(t)
            if not any(x["s"] == p and x["t"] == t for x in edges):
                edges.append({"s": p, "t": t, "rel": e["rel"]})
        nodes["encounters"] = {"id": "encounters", "kind": "encounter", "name": f"{enc} encounters"}
        edges.append({"s": p, "t": "encounters", "rel": "HAS_ENCOUNTER"})
    for c in r.concepts:
        add(c, center=True)
        if r.patients:
            for p in r.patients:
                if not G.has_edge(p, c):
                    edges.append({"s": p, "t": c, "rel": "NO RECORD", "missing": True})
            continue
        for s, _, e in G.in_edges(c, data=True):
            if e["rel"] in ("TREATS", "INDICATED_FOR"):
                add(s)
                edges.append({"s": s, "t": c, "rel": e["rel"], "n": e["n_patients"]})
        for _, t, e in G.out_edges(c, data=True):
            add(t)
            edges.append({"s": c, "t": t, "rel": e["rel"], "n": e["n_patients"]})
        pats = sorted({s for s in G.predecessors(c) if s.startswith("patient:")})
        for p in pats[:40]:
            add(p)
            edges.append({"s": p, "t": c, "rel": "HAS_CONDITION" if G.nodes[c]["kind"] == "condition" else "HAS"})
        if G.nodes[c]["kind"] == "condition":
            co = Counter(t for p in pats for t in set(G.successors(p)) if t != c and G.nodes[t]["kind"] == "condition")
            for t, n in sorted(co.items(), key=lambda x: (-x[1], x[0]))[:10]:
                add(t)
                edges.append({"s": c, "t": t, "rel": "CO-OCCURS", "n": n, "derived": True})
    for m in r.missing_patients:
        nodes[m] = {"id": m, "kind": "patient", "name": m.split(":")[1][:8], "center": True, "missing": True}
    return {"nodes": list(nodes.values()), "edges": edges}


questions = []
for key, label, q in QUESTIONS:
    r = R.retrieve(q)
    ctx = r.context()
    questions.append({
        "key": key, "label": label, "question": q,
        "linked_patients": r.patients, "missing_patients": r.missing_patients,
        "linked_concepts": [node(c) for c in r.concepts],
        "facts": r.facts, "fact_nodes": [sorted(set(m.group().rstrip(".") for m in NODE_RE.finditer(f))) for f in r.facts],
        "prompt_user": f"<facts>\n{ctx or '(no facts retrieved)'}\n</facts>\n\n<question>\n{q}\n</question>",
        "graph": view(r),
    })


def csv(name, n, cols=None):
    df = pd.read_csv(DATA / f"{name}.csv", dtype=str).fillna("")
    df = df[df.PATIENT == PID] if "PATIENT" in df else df
    return df.sort_values(df.columns[0], ascending=False).head(n)[cols or list(df.columns)].to_dict("records")


all_pat_cols = list(pd.read_csv(DATA / "patients.csv", nrows=0).columns)
pat = pd.read_csv(DATA / "patients.csv", dtype=str, usecols=PATIENT_COLS).fillna("")
raw = {
    "patient_columns": [{"col": c, "kept": c in PATIENT_COLS} for c in all_pat_cols],
    "patient_row": pat[pat.Id == PID].iloc[0].to_dict(),
    "conditions": csv("conditions", 5, ["START", "STOP", "PATIENT", "CODE", "DESCRIPTION"]),
    "medications": [r for r in csv("medications", 20, ["START", "STOP", "PATIENT", "CODE", "DESCRIPTION", "REASONCODE", "REASONDESCRIPTION"]) if r["REASONCODE"]][:3],
    "row_counts": {f: len(pd.read_csv(DATA / f"{f}.csv", usecols=[0])) for f in
                   ["patients", "encounters", "conditions", "medications", "procedures", "observations",
                    "allergies", "immunizations", "providers"]},
}
acet = G.get_edge_data("medication:313782", "condition:10509002")
raw["treats_example"] = {"med": node("medication:313782"), "cond": node("condition:10509002"),
                         "n_patients": list(acet.values())[0]["n_patients"]}

stats = {"nodes": dict(Counter(d["kind"] for _, d in G.nodes(data=True))),
         "edges": dict(Counter(e["rel"] for *_, e in G.edges(data=True)))}

latest = sorted(glob.glob(str(ROOT / "eval/results/retrieval-*.jsonl")))[-1]
ev = [json.loads(l) for l in open(latest)]
evals = [{"id": e["id"], "type": e["type"], "polarity": e["polarity"], "question": e["question"],
          "pass": e["pass"], "linked_concepts": e["linked_concepts"], "n_facts": e["n_facts"]} for e in ev]

out = {"questions": questions, "raw": raw, "stats": stats, "evals": evals, "system_prompt": SYSTEM_PROMPT,
       "model": cfg["llm"]["model"], "run_date": date.today().isoformat(), "fuzzy_threshold": cfg["retrieval"]["fuzzy_threshold"]}
target = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "demo" / "graphrag_walkthrough.html"
template = (ROOT / "demo" / "template.html").read_text()
target.write_text(template.replace("__DATA__", json.dumps(out).replace("</", "<\\/")))
print({q["key"]: (len(q["facts"]), len(q["graph"]["nodes"])) for q in questions})
print(f"Wrote {target} ({target.stat().st_size:,} bytes)")

# Standalone copy for opening from disk: the artifact host normally adds this skeleton at publish time.
# Without a network the Google Fonts link fails quietly and the page falls back to system fonts.
offline = target.with_name(target.stem + "_offline.html")
offline.write_text('<!doctype html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n'
                   '<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">\n'
                   "<style>body{margin:0}[hidden]{display:none!important}img{max-width:100%}</style>\n"
                   "</head>\n<body>\n" + target.read_text() + "\n</body>\n</html>\n")
print(f"Wrote {offline}")
