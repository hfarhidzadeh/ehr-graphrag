# EHR GraphRAG — notes for Claude

- **Synthetic data only.** Never read, load, or send real patient data. `answer.assert_synthetic` guards the LLM step; do not bypass it.
- Python env: `/Users/hamidfarhid/anaconda3/envs/ehr-graphrag/bin/python` (conda env `ehr-graphrag`, Python 3.11). The system python is 3.9 and lacks deps.
- Package lives in `src/ehr_graphrag/`; installed editable. Config in `config.yaml`.
- After changing `build_graph.py`, rebuild (`ehr-graphrag build`) before running eval.
- Ground truth in `eval/make_controls.py` must stay independent of the graph code (pandas on raw CSVs). Don't "fix" a failing control by changing the control; report it.
- Run `pytest -q tests` and `python eval/run_eval.py --mode retrieval` after changes; LLM-mode eval costs money — ask first.
