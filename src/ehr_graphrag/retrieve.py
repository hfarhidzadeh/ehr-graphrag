"""Graph retrieval: link a question to graph nodes, then serialize their neighbourhood as facts.

No LLM is involved here, so retrieval can be tested and evaluated for free and deterministically.
"""
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field

from rapidfuzz import fuzz

UUID_RE = re.compile(r"\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b", re.I)
CONCEPT_KINDS = ("condition", "medication", "procedure", "allergen", "vaccine", "lab")
# Question words too generic to link to a concept on their own.
STOPWORDS = {
    "patient", "patients", "which", "what", "does", "have", "having", "taking", "taken", "medication",
    "medications", "condition", "conditions", "allergic", "allergy", "allergies", "record", "records",
    "history", "treated", "treat", "treatment", "treatments", "current", "currently", "active", "there",
    "their", "about", "many", "with", "list", "other", "procedure", "procedures", "result", "results",
    "latest", "value", "level", "levels", "diagnosed", "diagnosis", "common", "most", "used", "given",
    # generic clinical head nouns: alone they would link "X syndrome" to every syndrome
    "syndrome", "disease", "disorder", "infection", "chronic", "acute", "fever", "injury",
    "syndromes", "diseases", "disorders", "infections", "injuries",
}


@dataclass
class Retrieval:
    patients: list
    missing_patients: list
    concepts: list
    facts: list = field(default_factory=list)

    def context(self):
        return "\n".join(f"[F{i}] {f}" for i, f in enumerate(self.facts, 1))


class GraphRetriever:
    def __init__(self, G, fuzzy_threshold=90, max_concepts=5, max_facts=200):
        self.G = G
        self.fuzzy_threshold = fuzzy_threshold
        self.max_concepts = max_concepts
        self.max_facts = max_facts
        self._index = [(n, d["name"].lower(), _tokens(d["name"]))
                       for n, d in G.nodes(data=True) if d["kind"] in CONCEPT_KINDS and len(d["name"]) >= 4]

    # ---- entity linking -------------------------------------------------------------------
    def link(self, question):
        ids = [u.lower() for u in UUID_RE.findall(question)]
        patients = [f"patient:{u}" for u in ids if f"patient:{u}" in self.G]
        missing = [f"patient:{u}" for u in ids if f"patient:{u}" not in self.G]

        q = UUID_RE.sub(" ", question).lower()
        q_tokens = _tokens(q) - STOPWORDS
        scored = []
        for nid, name, name_tokens in self._index:
            score = fuzz.partial_ratio(name, q) if len(name) <= len(q) else 0
            if q_tokens and name_tokens:
                score = max(score, max(fuzz.ratio(t, n) for t in q_tokens for n in name_tokens))
            if score >= self.fuzzy_threshold:
                scored.append((score, self.G.degree(nid), nid))
        concepts = [nid for _, _, nid in sorted(scored, reverse=True)[: self.max_concepts]]
        return patients, missing, concepts

    # ---- retrieval ------------------------------------------------------------------------
    def retrieve(self, question):
        patients, missing, concepts = self.link(question)
        r = Retrieval(patients, missing, concepts)
        for p in missing:
            r.facts.append(f"RECORD CHECK: {p} does not exist in the graph.")
        for p in patients:
            for c in concepts:
                r.facts.append(self._record_check(p, c))
        for p in patients:
            r.facts += self.patient_facts(p)
        for c in concepts:
            r.facts += self.concept_facts(c, include_patient_list=not patients)
        if len(r.facts) > self.max_facts:
            dropped = len(r.facts) - self.max_facts
            r.facts = r.facts[: self.max_facts] + [f"[{dropped} further facts truncated]"]
        return r

    def _record_check(self, p, c):
        edges = self.G.get_edge_data(p, c) or {}
        name = self._label(c)
        if not edges:
            return f"RECORD CHECK: no record links {p} to {name}."
        rels = Counter(e["rel"] for e in edges.values())
        return f"RECORD CHECK: {p} is linked to {name} by " + ", ".join(f"{k} x{v}" for k, v in rels.items()) + "."

    def patient_facts(self, p):
        G, d = self.G, self.G.nodes[p]
        vital = f"deceased {d['deathdate'][:10]}" if d["deathdate"] else "alive"
        facts = [f"{p} demographics: gender {d['gender']}, born {d['birthdate']}, {vital}, "
                 f"race {d['race']}, ethnicity {d['ethnicity']}."]

        by_rel = defaultdict(lambda: defaultdict(list))
        for _, tgt, e in G.out_edges(p, data=True):
            by_rel[e["rel"]][tgt].append(e)

        for c, es in _latest_first(by_rel["HAS_CONDITION"], "start"):
            active = any(e["status"] == "active" for e in es)
            facts.append(f"{p} HAS_CONDITION {self._label(c)}: {'active' if active else 'resolved'}, "
                         f"latest onset {es[0]['start'][:10]}, {len(es)} episode(s).")
        for c, es in _latest_first(by_rel["TAKES_MEDICATION"], "start"):
            e = es[0]
            stop = f"stopped {e['stop'][:10]}" if e["stop"] else "ongoing"
            reason = f", reason: {e['reason']}" if e["reason"] else ""
            facts.append(f"{p} TAKES_MEDICATION {self._label(c)}: {e['status']}, started {e['start'][:10]}, "
                         f"{stop}{reason}, {len(es)} prescription record(s).")
        for c, es in _latest_first(by_rel["ALLERGIC_TO"], "start"):
            e = es[0]
            facts.append(f"{p} ALLERGIC_TO {self._label(c)}: {e['status']}, category {e['category']}, "
                         f"reaction {e['reaction'] or 'unspecified'} ({e['severity'] or 'severity unspecified'}).")
        for c, es in _latest_first(by_rel["HAS_RESULT"], "date"):
            e = es[0]
            facts.append(f"{p} HAS_RESULT {self._label(c)} = {e['value']} {e['units']} on {e['date'][:10]} "
                         f"(latest of {e['n_results']}).")
        for c, es in list(_latest_first(by_rel["UNDERWENT"], "start"))[:25]:
            facts.append(f"{p} UNDERWENT {self._label(c)}: last {es[0]['start'][:10]}, {len(es)} time(s).")
        for c, es in _latest_first(by_rel["RECEIVED"], "date"):
            facts.append(f"{p} RECEIVED {self._label(c)}: last {es[0]['date'][:10]}, {len(es)} dose(s).")

        encounters = sorted(by_rel["HAS_ENCOUNTER"], key=lambda n: G.nodes[n]["start"], reverse=True)
        facts.append(f"{p} has {len(encounters)} encounter(s) on record.")
        for enc in encounters[:5]:
            ed = G.nodes[enc]
            specialty = [G.nodes[t]["specialty"] for _, t in G.out_edges(enc)]
            facts.append(f"{p} HAS_ENCOUNTER {enc}: {ed['start'][:10]}, {ed['encounter_class']}, {ed['name']}"
                         + (f", reason {ed['reason']}" if ed["reason"] else "")
                         + (f", provider specialty {specialty[0]}" if specialty else "") + ".")
        return facts

    def concept_facts(self, c, include_patient_list=True, top=10):
        G, d = self.G, self.G.nodes[c]
        facts = [f"{self._label(c)} is a {d['kind']} concept ({d['system']} {d['code']})."]

        patients = {s for s, _, e in G.in_edges(c, data=True) if s.startswith("patient:")}
        active = {s for s, _, e in G.in_edges(c, data=True) if s.startswith("patient:") and e.get("status") == "active"}
        facts.append(f"{len(patients)} patient(s) in the graph have a record of {d['name']}"
                     + (f"; {len(active)} with it active." if d["kind"] in ("condition", "medication", "allergen") else "."))
        if include_patient_list and patients:
            sample = sorted(active or patients)[:20]
            facts.append(f"Patients with {'active ' if active else ''}{d['name']} (up to 20): " + ", ".join(sample) + ".")

        for s, _, e in sorted(G.in_edges(c, data=True), key=lambda x: -x[2].get("n_patients", 0)):
            if e["rel"] in ("TREATS", "INDICATED_FOR"):
                facts.append(f"{self._label(s)} {e['rel']} {d['name']} (in {e['n_patients']} patient(s)).")
        for _, t, e in G.out_edges(c, data=True):
            if e["rel"] in ("TREATS", "INDICATED_FOR"):
                facts.append(f"{d['name']} {e['rel']} {self._label(t)} (in {e['n_patients']} patient(s)).")

        if d["kind"] == "condition" and patients:
            co = Counter(t for p in patients for t in set(G.successors(p))
                         if t != c and G.nodes[t]["kind"] == "condition")
            # Sort ties by node id: set iteration order varies between runs (hash seed).
            for t, n in sorted(co.items(), key=lambda x: (-x[1], x[0]))[:top]:
                facts.append(f"Co-occurring condition among patients with {d['name']}: {self._label(t)} ({n} patient(s)).")
        return facts

    def _label(self, n):
        return f"{self.G.nodes[n]['name']} [{n}]"


def _tokens(text):
    return {t for t in re.findall(r"[a-z]+", text.lower()) if len(t) >= 5}


def _latest_first(edges_by_target, date_key):
    """Yield (target, edges) with each edge list sorted newest first, targets ordered by newest edge."""
    items = [(t, sorted(es, key=lambda e: e[date_key], reverse=True)) for t, es in edges_by_target.items()]
    return sorted(items, key=lambda x: x[1][0][date_key], reverse=True)
