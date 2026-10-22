"""Build a patient / clinical-concept knowledge graph from Synthea CSV exports.

Graph layout (networkx.MultiDiGraph, node ids are "<kind>:<key>"):

  patient:<uuid>  -HAS_CONDITION->     condition:<SNOMED>
                  -TAKES_MEDICATION->  medication:<RxNorm>
                  -UNDERWENT->         procedure:<SNOMED>
                  -ALLERGIC_TO->       allergen:<code>
                  -RECEIVED->          vaccine:<CVX>
                  -HAS_RESULT->        lab:<LOINC>          (latest value + count)
                  -HAS_ENCOUNTER->     encounter:<uuid> -WITH_PROVIDER-> provider:<uuid>

  Cross-patient "knowledge" edges, aggregated from REASONCODE columns:
  medication:<code> -TREATS->        condition:<code>   (n_patients)
  procedure:<code>  -INDICATED_FOR-> condition:<code>   (n_patients)
"""
import pickle
import re
from pathlib import Path

import networkx as nx
import pandas as pd

# Direct identifiers (names, SSN, address, ...) are never read, even from synthetic
# data, so the pipeline stays minimal-by-default if pointed at another export.
PATIENT_COLS = ["Id", "BIRTHDATE", "DEATHDATE", "RACE", "ETHNICITY", "GENDER"]
LAB_CATEGORIES = {"laboratory", "vital-signs"}


def _read(data_dir, name, usecols=None):
    return pd.read_csv(Path(data_dir) / f"{name}.csv", dtype=str, usecols=usecols).fillna("")


def clean_name(description):
    """'Hypertension (disorder)' -> 'Hypertension'."""
    return re.sub(r"\s*\((disorder|finding|situation|procedure|substance|qualifier value|"
                  r"regime/therapy|observable entity|morphologic abnormality)\)$", "", description).strip()


def _concept(G, kind, code, description, system=""):
    nid = f"{kind}:{code}"
    if nid not in G:
        G.add_node(nid, kind=kind, code=code, name=clean_name(description), system=system)
    return nid


def _status(stop):
    return "resolved" if stop else "active"


def build_graph(data_dir):
    G = nx.MultiDiGraph()

    for r in _read(data_dir, "patients", PATIENT_COLS).itertuples():
        G.add_node(f"patient:{r.Id}", kind="patient", birthdate=r.BIRTHDATE, deathdate=r.DEATHDATE,
                   gender=r.GENDER, race=r.RACE, ethnicity=r.ETHNICITY)

    for r in _read(data_dir, "providers", ["Id", "SPECIALITY"]).itertuples():
        G.add_node(f"provider:{r.Id}", kind="provider", specialty=r.SPECIALITY)

    enc = _read(data_dir, "encounters", ["Id", "START", "PATIENT", "PROVIDER", "ENCOUNTERCLASS",
                                         "DESCRIPTION", "REASONDESCRIPTION"])
    for r in enc.itertuples():
        e = f"encounter:{r.Id}"
        G.add_node(e, kind="encounter", start=r.START, encounter_class=r.ENCOUNTERCLASS,
                   name=r.DESCRIPTION, reason=clean_name(r.REASONDESCRIPTION))
        G.add_edge(f"patient:{r.PATIENT}", e, rel="HAS_ENCOUNTER")
        if r.PROVIDER:
            G.add_edge(e, f"provider:{r.PROVIDER}", rel="WITH_PROVIDER")

    for r in _read(data_dir, "conditions").itertuples():
        c = _concept(G, "condition", r.CODE, r.DESCRIPTION, r.SYSTEM)
        G.add_edge(f"patient:{r.PATIENT}", c, rel="HAS_CONDITION", start=r.START, stop=r.STOP,
                   status=_status(r.STOP), encounter=r.ENCOUNTER)

    meds = _read(data_dir, "medications", ["START", "STOP", "PATIENT", "ENCOUNTER", "CODE",
                                           "DESCRIPTION", "REASONCODE", "REASONDESCRIPTION"])
    for r in meds.itertuples():
        m = _concept(G, "medication", r.CODE, r.DESCRIPTION, "RxNorm")
        G.add_edge(f"patient:{r.PATIENT}", m, rel="TAKES_MEDICATION", start=r.START, stop=r.STOP,
                   status=_status(r.STOP), reason=clean_name(r.REASONDESCRIPTION), encounter=r.ENCOUNTER)
    _add_reason_edges(G, meds, "medication", "TREATS")

    procs = _read(data_dir, "procedures", ["START", "PATIENT", "ENCOUNTER", "SYSTEM", "CODE",
                                           "DESCRIPTION", "REASONCODE", "REASONDESCRIPTION"])
    for r in procs.itertuples():
        p = _concept(G, "procedure", r.CODE, r.DESCRIPTION, r.SYSTEM)
        G.add_edge(f"patient:{r.PATIENT}", p, rel="UNDERWENT", start=r.START,
                   reason=clean_name(r.REASONDESCRIPTION), encounter=r.ENCOUNTER)
    _add_reason_edges(G, procs, "procedure", "INDICATED_FOR")

    for r in _read(data_dir, "allergies", ["START", "STOP", "PATIENT", "CODE", "SYSTEM", "DESCRIPTION",
                                           "CATEGORY", "DESCRIPTION1", "SEVERITY1"]).itertuples():
        a = _concept(G, "allergen", r.CODE, r.DESCRIPTION, r.SYSTEM)
        G.add_edge(f"patient:{r.PATIENT}", a, rel="ALLERGIC_TO", start=r.START, stop=r.STOP,
                   status=_status(r.STOP), category=r.CATEGORY, reaction=clean_name(r.DESCRIPTION1),
                   severity=r.SEVERITY1)

    for r in _read(data_dir, "immunizations", ["DATE", "PATIENT", "CODE", "DESCRIPTION"]).itertuples():
        v = _concept(G, "vaccine", r.CODE, r.DESCRIPTION, "CVX")
        G.add_edge(f"patient:{r.PATIENT}", v, rel="RECEIVED", date=r.DATE)

    # Observations are the bulk of the data; keep only the latest lab/vital value per patient and test.
    obs = _read(data_dir, "observations", ["DATE", "PATIENT", "CATEGORY", "CODE", "DESCRIPTION", "VALUE", "UNITS"])
    obs = obs[obs.CATEGORY.isin(LAB_CATEGORIES)].sort_values("DATE")
    counts = obs.groupby(["PATIENT", "CODE"]).size()
    for r in obs.groupby(["PATIENT", "CODE"]).tail(1).itertuples():
        lab = _concept(G, "lab", r.CODE, r.DESCRIPTION, "LOINC")
        G.add_edge(f"patient:{r.PATIENT}", lab, rel="HAS_RESULT", date=r.DATE, value=r.VALUE,
                   units=r.UNITS, n_results=int(counts[(r.PATIENT, r.CODE)]))

    return G


def _add_reason_edges(G, df, kind, rel):
    """Aggregate per-patient REASONCODE links into concept -> condition knowledge edges."""
    linked = df[df.REASONCODE != ""]
    for (code, reason_code), grp in linked.groupby(["CODE", "REASONCODE"]):
        cond = _concept(G, "condition", reason_code, grp.REASONDESCRIPTION.iloc[0], "SNOMED-CT")
        G.add_edge(f"{kind}:{code}", cond, rel=rel, n_patients=grp.PATIENT.nunique())


def save_graph(G, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(G, f)


def load_graph(path):
    with open(path, "rb") as f:
        return pickle.load(f)
