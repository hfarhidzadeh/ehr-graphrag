"""Generate positive and negative control questions with ground truth.

Ground truth is computed straight from the raw CSVs with pandas, deliberately NOT through the
graph code, so a bug in the graph builder or retriever shows up as a failed control.

  python eval/make_controls.py [--per-type 5] [--seed 7]   ->  eval/controls.jsonl
"""
import argparse
import json
import random
import sys
import uuid
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from ehr_graphrag.build_graph import clean_name  # noqa: E402
from ehr_graphrag.cli import load_config  # noqa: E402

# Social-determinant "findings" dominate Synthea conditions; controls focus on clinical disorders.
CLINICAL = "(disorder)"


def read(data_dir, name):
    return pd.read_csv(Path(data_dir) / f"{name}.csv", dtype=str).fillna("")


def make_controls(data_dir, per_type, seed):
    rng = random.Random(seed)
    cond, meds, allergies = read(data_dir, "conditions"), read(data_dir, "medications"), read(data_dir, "allergies")
    patients = read(data_dir, "patients").Id.tolist()
    items = []

    def add(kind, polarity, question, **expect):
        items.append({"id": f"{kind}-{len(items):03d}", "type": kind, "polarity": polarity,
                      "question": question, "expect": expect})

    # P1: active clinical conditions of a patient
    active = cond[(cond.STOP == "") & cond.DESCRIPTION.str.endswith(CLINICAL)]
    for pid in rng.sample(sorted(active.PATIENT.unique()), per_type):
        rows = active[active.PATIENT == pid].drop_duplicates("CODE")
        add("active_conditions", "positive", f"What active disorders does patient {pid} have?",
            nodes=[f"condition:{c}" for c in rows.CODE], names=[clean_name(d) for d in rows.DESCRIPTION])

    # P2: reason a patient takes a medication
    reasoned = meds[meds.REASONCODE != ""].drop_duplicates(["PATIENT", "CODE"])
    for r in reasoned.sample(per_type, random_state=seed).itertuples():
        add("medication_reason", "positive", f"Why is patient {r.PATIENT} taking {r.DESCRIPTION}?",
            nodes=[f"medication:{r.CODE}"], names=[clean_name(r.REASONDESCRIPTION)])

    # P3: allergy present
    for r in allergies.drop_duplicates(["PATIENT", "CODE"]).sample(per_type, random_state=seed).itertuples():
        add("allergy_yes", "positive", f"Is patient {r.PATIENT} allergic to {clean_name(r.DESCRIPTION)}?",
            nodes=[f"allergen:{r.CODE}"], answer="yes")

    # P4: cohort size for a clinical condition (patients with any record of it)
    clinical = cond[cond.DESCRIPTION.str.endswith(CLINICAL)]
    counts = clinical.groupby(["CODE", "DESCRIPTION"]).PATIENT.nunique().reset_index()
    for r in counts[counts.PATIENT >= 3].sample(per_type, random_state=seed).itertuples():
        add("cohort_count", "positive", f"How many patients have a record of {clean_name(r.DESCRIPTION)}?",
            nodes=[f"condition:{r.CODE}"], answer=str(r.PATIENT))

    # P5: medications documented as treating a condition (cross-patient knowledge edge)
    treated = meds[meds.REASONCODE != ""].groupby(["REASONCODE", "REASONDESCRIPTION"]).CODE.unique().reset_index()
    for r in treated.sample(min(per_type, len(treated)), random_state=seed).itertuples():
        add("treatments", "positive", f"Which medications are used to treat {clean_name(r.REASONDESCRIPTION)}?",
            nodes=[f"medication:{c}" for c in r.CODE])

    # N1: patient that does not exist
    for _ in range(per_type):
        add("unknown_patient", "negative", f"What medications is patient {uuid.UUID(int=rng.getrandbits(128))} taking?",
            answer="not found")

    # N2: real patient asked about a common clinical condition they have no record of
    common = counts.sort_values("PATIENT", ascending=False).head(15)
    for _ in range(per_type):
        r = common.sample(1, random_state=rng.randrange(10**6)).iloc[0]
        has = set(cond[cond.CODE == r.CODE].PATIENT)
        pid = rng.choice([p for p in patients if p not in has])
        add("absent_condition", "negative", f"Does patient {pid} have {clean_name(r.DESCRIPTION)}?",
            nodes=[f"condition:{r.CODE}"], answer="no")

    # N3: fictitious condition names; nothing should be linked
    for name in ["Zorblaxian fever", "Quentrolitis", "Myxovarian syndrome", "Hepatoplasmic drift", "Velkor disease"][:per_type]:
        add("fake_concept", "negative", f"How many patients have {name}?", nodes=[], answer="not found")
    return items


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-type", type=int, default=5)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    items = make_controls(load_config()["data_dir"], args.per_type, args.seed)
    out = ROOT / "eval" / "controls.jsonl"
    out.write_text("".join(json.dumps(i) + "\n" for i in items))
    print(f"Wrote {len(items)} control questions to {out}")


if __name__ == "__main__":
    main()
