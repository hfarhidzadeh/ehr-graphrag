"""Integrity checks: the graph must reflect the CSVs exactly and carry no direct identifiers."""
from collections import Counter

import pandas as pd
import pytest

from ehr_graphrag.build_graph import build_graph
from ehr_graphrag.cli import load_config
from ehr_graphrag.retrieve import GraphRetriever

CFG = load_config()


@pytest.fixture(scope="module")
def G():
    return build_graph(CFG["data_dir"])


def csv_rows(name):
    return len(pd.read_csv(CFG["data_dir"] / f"{name}.csv", dtype=str))


@pytest.mark.parametrize("rel,csv", [("HAS_CONDITION", "conditions"), ("TAKES_MEDICATION", "medications"),
                                     ("UNDERWENT", "procedures"), ("ALLERGIC_TO", "allergies"),
                                     ("RECEIVED", "immunizations"), ("HAS_ENCOUNTER", "encounters")])
def test_one_edge_per_csv_row(G, rel, csv):
    rels = Counter(e["rel"] for *_, e in G.edges(data=True))
    assert rels[rel] == csv_rows(csv)


def test_every_patient_loaded(G):
    assert sum(d["kind"] == "patient" for _, d in G.nodes(data=True)) == csv_rows("patients")


def test_no_direct_identifiers(G):
    banned = {"FIRST", "LAST", "SSN", "ADDRESS", "first", "last", "ssn", "address", "name"}
    for n, d in G.nodes(data=True):
        if d["kind"] == "patient":
            assert not banned & set(d), n


def test_no_dangling_patient_edges(G):
    # An edge to a patient id absent from patients.csv would create a node without attributes.
    assert all("kind" in d for _, d in G.nodes(data=True))


def test_unknown_patient_is_reported(G):
    r = GraphRetriever(G).retrieve("What is patient 00000000-0000-0000-0000-000000000000 taking?")
    assert r.missing_patients and "does not exist" in r.context()
