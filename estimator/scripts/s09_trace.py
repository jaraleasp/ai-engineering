"""S09 pre-work — manual RAG trace for examples/transcripts/02_ambiguous.txt.

Step 1: embed the FULL raw transcript with the same model the service uses
        (text-embedding-3-small, 1536-dim) and report dim / L2 norm / first &
        last components.
Step 2: call POST /search with the raw transcript as the query (k=5) — exactly
        how S08 exposes retrieval (the endpoint re-embeds the query internally).

Run from the `estimator/` directory:
    uv run --with openai python scripts/s09_trace.py

There is no "encode" endpoint that returns a vector, so step 1 calls OpenAI
directly. The OPENAI_API_KEY is read from estimator/.env.
"""

from __future__ import annotations

import json
import math
import urllib.request
from pathlib import Path

from openai import OpenAI

HERE = Path(__file__).resolve().parent          # estimator/scripts
ESTIMATOR_DIR = HERE.parent                       # estimator/
REPO_ROOT = ESTIMATOR_DIR.parent                  # repo root
TRANSCRIPT = REPO_ROOT / "examples" / "transcripts" / "02_ambiguous.txt"
ENV_FILE = ESTIMATOR_DIR / ".env"

MODEL = "text-embedding-3-small"
SEARCH_URL = "http://localhost:8000/search"
K = 5


def load_api_key() -> str:
    for line in ENV_FILE.read_text(encoding="utf-8").splitlines():
        if line.startswith("OPENAI_API_KEY="):
            return line.split("=", 1)[1].strip()
    raise SystemExit("OPENAI_API_KEY not found in estimator/.env")


def main() -> None:
    text = TRANSCRIPT.read_text(encoding="utf-8")
    print(f"# transcript: {TRANSCRIPT.name}")
    print(f"# chars: {len(text)} | approx words: {len(text.split())}")
    print("=" * 72)

    # ---- STEP 1: embed the whole transcript -----------------------------
    client = OpenAI(api_key=load_api_key())
    resp = client.embeddings.create(model=MODEL, input=text)
    vec = resp.data[0].embedding
    norm = math.sqrt(sum(x * x for x in vec))

    print("STEP 1 — embed full transcript")
    print(f"  model:        {MODEL}")
    print(f"  dimension:    {len(vec)}")
    print(f"  L2 norm:      {norm:.6f}")
    print(f"  prompt_tokens:{resp.usage.prompt_tokens}")
    print(f"  first 3:      {vec[:3]}")
    print(f"  last 3:       {vec[-3:]}")
    print("=" * 72)

    # ---- STEP 2: semantic search with the raw transcript ----------------
    payload = json.dumps({"query": text, "k": K}).encode("utf-8")
    req = urllib.request.Request(
        SEARCH_URL, data=payload, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req) as r:  # noqa: S310 (localhost, trusted)
        result = json.loads(r.read().decode("utf-8"))

    print(f"STEP 2 — POST /search  (k={K}, query = full transcript)")
    print(f"  search_time_ms: {result.get('search_time_ms')}")
    print("  --- raw results ---")
    for i, hit in enumerate(result["results"], 1):
        md = hit["metadata"]
        print(
            f"  [{i}] dist={hit['distance']:.4f}  "
            f"{md.get('budget_id')} / {md.get('client_sector')} / {md.get('main_technology')}  "
            f"({md.get('estimated_hours')}h)"
        )
        first_line = hit["content"].splitlines()[0]
        comp_line = next(
            (ln for ln in hit["content"].splitlines() if ln.startswith("Component:")), ""
        )
        print(f"        {first_line}")
        print(f"        {comp_line}")
    print("=" * 72)
    print("# full raw JSON of /search response:")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
