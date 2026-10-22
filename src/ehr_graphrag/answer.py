"""Generate a grounded answer from retrieved graph facts with Claude."""
from pathlib import Path

import anthropic

SYSTEM_PROMPT = """You answer questions about a synthetic electronic health record (EHR) dataset.
You are given numbered facts retrieved from a knowledge graph built from that dataset.

Rules:
- Use only the facts provided. Do not add medical knowledge that is not in the facts.
- Cite the facts that support each statement, e.g. [F3] or [F3, F7].
- If the facts do not contain what is needed to answer, say "Not found in the record" and state what is missing.
- If a RECORD CHECK fact says a patient does not exist or has no link to a concept, report that plainly.
- Be concise."""


def assert_synthetic(data_dir):
    """Org policy: no real patient data goes to an AI service. The data folder must declare itself synthetic."""
    marker = Path(data_dir) / "DATA_PROVENANCE.txt"
    if not marker.exists() or "synthetic" not in marker.read_text().lower():
        raise PermissionError(
            f"{marker} missing or does not declare the data synthetic; refusing to send records to the LLM.")


def answer(question, retrieval, model="claude-opus-5-5", effort="medium", max_tokens=16000, client=None):
    client = client or anthropic.Anthropic()
    user = (f"<facts>\n{retrieval.context() or '(no facts retrieved)'}\n</facts>\n\n"
            f"<question>\n{question}\n</question>")
    response = client.beta.messages.create(
        model=model,
        max_tokens=max_tokens,
        system=SYSTEM_PROMPT,
        output_config={"effort": effort},
        betas=["server-side-fallback-2026-07-01"],
        fallbacks="default",
        messages=[{"role": "user", "content": user}],
    )
    if response.stop_reason == "refusal":
        category = response.stop_details.category if response.stop_details else None
        return f"[model declined to answer; category={category}]"
    return "".join(b.text for b in response.content if b.type == "text")
