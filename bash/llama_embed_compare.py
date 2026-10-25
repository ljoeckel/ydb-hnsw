#!/usr/bin/env python3
"""Compare sentence-transformers (torch) embeddings with llama.cpp GGUF ones.

The question this answers: can `llama-embedding` (llama.cpp) replace the
torch/sentence-transformers path in `src/bert_nim.nim` without changing what the
index sees?

It encodes the same texts both ways, then reports

  * per-text cosine and max |diff| between the two vectors,
  * top-k agreement when a *llama.cpp* query vector is ranked against
    *torch* corpus vectors (the mixed case: existing index, new queries),
  * the same ranking with torch query vectors as a control.

Run it with the venv that has torch + sentence-transformers:

  ./hnsw_env/bin/python bash/llama_embed_compare.py \
      --gguf ~/models/gguf/paraphrase-multilingual-MiniLM-L12-v2-f16.gguf \
      --llama-embedding ~/git/llama.cpp/build/bin/llama-embedding

Add `--texts FILE` (one text per line) to use real article titles instead of the
built-in sample.  Keep texts within the model's 128-token truncation limit; both
sides truncate, but llama.cpp uses the GGUF's context size, which may be larger
than 128, so longer texts are compared unfairly.

The GGUF must come from a conversion of THIS model with
`bash/llama_cpp_xlmroberta.patch` applied: without it `convert_hf_to_gguf.py`
writes a WordPiece vocab for a Unigram tokenizer and every word becomes `<unk>`
(cosine ~0.14).  See LLAMA_CPP.md.
"""

import argparse
import json
import os
import subprocess
import sys

import numpy as np

DEFAULT_GGUF = os.path.expanduser("~/models/gguf/paraphrase-multilingual-MiniLM-L12-v2-f16.gguf")
DEFAULT_LLAMA = os.path.expanduser("~/git/llama.cpp/build/bin/llama-embedding")
DEFAULT_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

# Short, mixed-language, close to the article titles this index actually holds.
SAMPLE_TEXTS = [
    "Die Regierung beschließt neue Maßnahmen gegen den Klimawandel",
    "Bundestag stimmt über den Haushalt ab",
    "Neue Studie zeigt Fortschritte bei der Krebsforschung",
    "Der FC Bayern gewinnt das Spiel in letzter Minute",
    "Le gouvernement annonce une réforme des retraites",
    "El gobierno presenta un nuevo plan económico",
    "Il governo approva la legge di bilancio",
    "New vaccine shows promise in clinical trials",
    "Stock markets fall after interest rate decision",
    "Scientists discover a new species in the Amazon",
    "Правительство объявило о новых мерах",
    "الشرطة تعلن عن اعتقالات في العاصمة",
    "Wiener Opernball zieht Prominente aus ganz Europa an",
    "Slowenien und Kroatien streiten über die Grenze",
    "Tesla kündigt neue Batteriefabrik in Brandenburg an",
    "Unwetter führt zu Stromausfällen im Norden",
    "Die Inflation ist im September leicht gesunken",
    "Ein Erdbeben der Stärke 5.2 erschüttert die Region",
    "Das Museum eröffnet eine Ausstellung über moderne Kunst",
    "Der Zugverkehr ist wegen Bauarbeiten unterbrochen",
    "Researchers publish an open dataset for machine translation",
    "The city council approves a new bicycle lane network",
    "Le musée présente une exposition consacrée à la photographie",
    "La selección nacional gana la final del torneo",
    "Die Krankenhäuser warnen vor Personalengpässen",
    "Die Universität erhält einen neuen Forschungsbau",
    "A new telescope captures the sharpest image of a galaxy",
    "Der neue Roman der Autorin steht auf Platz eins",
    "Die Polizei ermittelt nach einem Einbruch in der Innenstadt",
    "Farmers protest against new environmental rules",
    "Die Bahn kündigt schnellere Verbindungen nach Berlin an",
    "Das Unternehmen meldet Rekordgewinne für das Quartal",
]


def torch_embed(model_name: str, texts: list[str]) -> np.ndarray:
    """Embedding matrix (n, dim) as sentence-transformers produces it."""
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name)
    vecs = model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
    return np.ascontiguousarray(vecs, dtype=np.float32)


def llama_embed(binary: str, gguf: str, texts: list[str], pooling: str,
                threads: int) -> np.ndarray:
    """Embedding matrix (n, dim) via llama.cpp's embedding example.

    The example splits the prompt on `--embd-separator` (default "\\n"), so the
    texts go in as one newline-separated prompt and the JSON output comes back
    one embedding per line.
    """
    if any("\n" in t for t in texts):
        sys.exit("texts must not contain newlines (the prompt is newline-split)")

    cmd = [
        binary,
        "-m", gguf,
        "-p", "\n".join(texts),
        "--pooling", pooling,
        "--embd-normalize", "2",
        "--embd-output-format", "json",
        "-t", str(threads),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        sys.stderr.write(proc.stdout[-4000:] + "\n" + proc.stderr[-4000:] + "\n")
        sys.exit(f"llama-embedding failed with exit code {proc.returncode}")

    # The JSON goes to stdout; log lines go to stderr.  A leading "{" is the
    # start of the object (the example may also print a blank line first).
    start = proc.stdout.find('{"object"')
    if start < 0:
        start = proc.stdout.find("{")
    if start < 0:
        sys.exit("no JSON object in llama-embedding output:\n" + proc.stdout[-2000:])
    doc = json.loads(proc.stdout[start:])

    vecs = [np.asarray(d["embedding"], dtype=np.float32) for d in doc["data"]]
    if len(vecs) != len(texts):
        sys.exit(f"expected {len(texts)} embeddings, got {len(vecs)}")
    return np.stack(vecs)


def rank(query: np.ndarray, corpus: np.ndarray, k: int) -> list[int]:
    """Top-k corpus indices by cosine, ties broken by index (both sides use it)."""
    sims = corpus @ query
    order = np.lexsort((np.arange(len(sims)), -sims))
    return [int(i) for i in order[:k]]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gguf", default=DEFAULT_GGUF)
    ap.add_argument("--llama-embedding", default=DEFAULT_LLAMA)
    ap.add_argument("--model", default=DEFAULT_MODEL,
                    help="sentence-transformers model name or local directory")
    ap.add_argument("--texts", help="file with one text per line (default: built-in sample)")
    ap.add_argument("--pooling", default="mean")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--threads", type=int, default=4)
    args = ap.parse_args()

    texts = SAMPLE_TEXTS
    if args.texts:
        with open(args.texts, encoding="utf-8") as f:
            texts = [line.rstrip("\n") for line in f if line.strip()]
    if not texts:
        sys.exit("no texts to compare")

    print(f"texts: {len(texts)}")
    print(f"torch model: {args.model}")
    print(f"gguf: {args.gguf}")

    ref = torch_embed(args.model, texts)
    got = llama_embed(args.llama_embedding, args.gguf, texts, args.pooling, args.threads)
    print(f"torch shape {ref.shape}, llama.cpp shape {got.shape}")
    if ref.shape != got.shape:
        sys.exit("shape mismatch - different model or dimension")

    # Both sides normalise, so the vectors should be unit length.
    print(f"norms: torch {np.linalg.norm(ref, axis=1).min():.6f}.."
          f"{np.linalg.norm(ref, axis=1).max():.6f}, "
          f"llama.cpp {np.linalg.norm(got, axis=1).min():.6f}.."
          f"{np.linalg.norm(got, axis=1).max():.6f}")

    cos = np.sum(ref * got, axis=1)
    maxdiff = np.abs(ref - got).max(axis=1)
    print("\nper-text agreement (torch vs llama.cpp):")
    print(f"  cosine      min {cos.min():.8f}  mean {cos.mean():.8f}  max {cos.max():.8f}")
    print(f"  max |diff|  min {maxdiff.min():.6f}  mean {maxdiff.mean():.6f}  max {maxdiff.max():.6f}")

    worst = int(np.argmin(cos))
    if cos.min() < 0.99999:
        print(f"  worst text (cos {cos[worst]:.8f}): {texts[worst]!r}")

    # Mixed mode: llama.cpp query against torch corpus (the existing index).
    agree = 0
    same_order = 0
    for q in range(len(texts)):
        a = rank(got[q], ref, args.k)
        b = rank(ref[q], ref, args.k)
        agree += len(set(a) & set(b)) / args.k
        same_order += a == b
    print("\nranking with a llama.cpp query against torch corpus vectors "
          f"(k={args.k}, n={len(texts)}):")
    print(f"  mean top-{args.k} overlap {agree / len(texts):.3f}, "
          f"identical order {same_order}/{len(texts)}")
    print("  (control: the same ranking with torch query vectors is 1.000 / all)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
