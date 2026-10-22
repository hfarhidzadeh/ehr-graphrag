# EHR GraphRAG

Question answering over electronic health records. The records are turned into a knowledge graph, the
relevant part of the graph is retrieved, and Claude answers using only those facts, citing each one.

**Data policy:** this project uses **synthetic data only** ([Synthea](https://synthetichealth.github.io/synthea/)).
Real patient data must not be sent to AI services. The LLM step will not run unless the data folder
contains a `DATA_PROVENANCE.txt` that declares the data synthetic.

## How it works

```
Synthea CSVs ──build_graph──▶ knowledge graph ──retrieve──▶ numbered facts ──answer──▶ cited answer
 (111 pts)                    (networkx)        (no LLM)      [F1] [F2] ...   (Claude)
```

| Layer | Contents |
|---|---|
| Patient facts | `patient -HAS_CONDITION/TAKES_MEDICATION/UNDERWENT/ALLERGIC_TO/RECEIVED/HAS_RESULT-> concept`, encounters → provider |
| Concepts | conditions & procedures (SNOMED), medications (RxNorm), labs (LOINC), vaccines (CVX), allergens |
| Cross-patient knowledge | `medication -TREATS-> condition`, `procedure -INDICATED_FOR-> condition`, aggregated from REASONCODE |

Names, SSNs and addresses are never loaded. Each patient keeps only demographics and a UUID.

Retrieval is deterministic:
1. **Entity linking:** patient UUIDs are matched with a regex, and concepts with fuzzy string matching (rapidfuzz).
2. **Subgraph to facts:** for each linked patient, the record is retrieved. For each concept, the retriever gets cohort size, treatments and co-occurring conditions. When the question names both a patient and a concept, an explicit `RECORD CHECK` fact is added.

## Setup

```bash
conda env create -f environment.yml        # Python 3.11
conda activate ehr-graphrag
pip install -e .
export ANTHROPIC_API_KEY=...               # only needed for LLM answers
```

The synthetic data is already in `data/raw/synthea/`. To download it again, use
[synthea_sample_data_csv_latest.zip](https://synthetichealth.github.io/synthea-sample-data/downloads/latest/synthea_sample_data_csv_latest.zip).

## Usage

```bash
ehr-graphrag build                                   # CSV -> data/processed/ehr_graph.pkl
ehr-graphrag stats
ehr-graphrag ask "Which medications are used to treat Childhood asthma?" --no-llm   # show retrieved facts
ehr-graphrag ask "What active disorders does patient <uuid> have?"                  # Claude answer
pytest -q tests                                      # graph integrity checks
```

## Evaluation: positive and negative controls

`eval/make_controls.py` computes ground truth **directly from the CSVs with pandas**, without going through
the graph code, so a bug in the graph or retriever shows up as a failed control.

| Control | Polarity | What a correct system does |
|---|---|---|
| active_conditions, medication_reason, allergy_yes, cohort_count, treatments | positive | retrieves and answers with the true facts |
| unknown_patient (random UUID) | negative | says the patient does not exist |
| absent_condition (real patient, condition they lack) | negative | says "no" |
| fake_concept ("Quentrolitis") | negative | says not found, without inventing anything |
| `--mode baseline` (Claude with no facts) | negative | says "not found" to everything; if it answers, it is hallucinating |

```bash
python eval/make_controls.py           # 40 questions -> eval/controls.jsonl
python eval/run_eval.py --mode retrieval   # free
python eval/run_eval.py --mode llm         # ~40 Claude calls
python eval/run_eval.py --mode baseline    # ~40 Claude calls
```

Every answer is saved to `eval/results/` and **must be read by a person**. The automatic LLM grading is
only heuristic: it checks citations, yes/no, numbers and "not found".

**Current status (retrieval mode): 39/40.** The known failure is that "Zorblaxian **fever**" links to the real
concept *Fever*. Lexical linking over-matches. The LLM-mode control checks whether Claude notices
that the facts are about a different concept.

## Roadmap

1. **v0 (done):** structured graph, lexical linking, single-call grounded answer, and the control suite.
2. **Better linking:** synonyms and abbreviations (e.g. "HTN", "T2DM") through UMLS/SNOMED hierarchies, plus a local embedding index of concept names.
3. **Agentic retrieval:** expose graph queries (patient summary, cohort filter, treatment lookup, timeline) as Claude tools for multi-hop and aggregate questions.
4. **Neo4j backend:** load the same schema into Neo4j (Docker) for Cypher queries and larger cohorts (generate 1k–10k patients with Synthea).
5. **Unstructured notes:** use Synthea text exports, extract entities with an LLM, and link them into the same concept nodes. This step needs its own extraction controls.
6. **Comparison:** compare GraphRAG with plain vector RAG over the same records on the same control set.
