"""Bridge between the Python/SBERT pipeline and Nim (see `bert_nim.nim`).

The Nim side imports this module with nimpy and calls `embed()` or
`embed_numpy()` / `query()`. Keep the public API here small and boring - it is
the contract that crosses the language boundary, so renaming things breaks the
Nim build (see `bert_nim.nim`).
"""

import os

# --- threading: configured before torch (and MKL) are imported --------------
# This machine's CPU (Xeon E5-1620 v2 / Ivy Bridge) has AVX and SSE4.2 but no
# AVX2/AVX-512, while the Intel MKL bundled with torch was observed executing an
# AVX-512 kernel from a worker thread (vmovdqu64 ... %zmm13, inside
# mkl_vml_kernel_sTanh_Z0HAynn) -> SIGILL / exit 132, intermittent and only
# under concurrent load. That is why the default here is a single thread, on top
# of ~/.bashrc exporting OMP_NUM_THREADS=1 MKL_NUM_THREADS=1.
#
# More cores are there (4 cores / 8 threads) and they do help large batches -
# measured on 256 short texts: 1 thread 55 texts/s, 4 threads 193 texts/s,
# 8 threads 188 texts/s (hyperthreading on 4 physical cores does not pay off).
# Opt in deliberately, with the guardrail that pins MKL to a kernel class this
# CPU really supports so the AVX-512 path cannot be taken:
#
#   SBERT_THREADS=4 MKL_ENABLE_INSTRUCTIONS=SSE4_2 nimble runHnsw
#
# MKL_ENABLE_INSTRUCTIONS is not set from here on purpose: it would also cap MKL
# on a machine that could legitimately use AVX2/AVX-512.
THREADS = max(1, int(os.environ.get("SBERT_THREADS", "1")))
if THREADS > 1:
    # MKL prioritises MKL_NUM_THREADS over OMP_NUM_THREADS, and the shell exports
    # both as 1, so torch.set_num_threads() alone would leave MKL single
    # threaded. These have to be in place before the libraries are loaded.
    os.environ["OMP_NUM_THREADS"] = str(THREADS)
    os.environ["MKL_NUM_THREADS"] = str(THREADS)

import numpy as np
import torch
from sentence_transformers import SentenceTransformer, util
from transformers.utils import logging

logging.set_verbosity_error()   # hides the LOAD REPORT and misc warnings
torch.set_num_threads(THREADS)

#MODEL_NAME = "all-MiniLM-L6-v2"
MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"

# Loaded once at import time, so every call coming from Nim reuses the weights.
model = SentenceTransformer(MODEL_NAME)


def dim() -> int:
    """Embedding size of the loaded model (384 for all-MiniLM-L6-v2)."""
    # Renamed in sentence-transformers 6; this env is pinned to 6.0.1.
    return int(model.get_embedding_dimension())


def embed(texts) -> list[list[float]]:
    """texts: sequence[str] -> one plain-Python vector per text.

    Returning plain lists keeps the nimpy conversion trivial on the Nim side:
    `bridge.embed(texts).to(seq[seq[float32]])`.
    """
    vectors = model.encode(list(texts), convert_to_numpy=True)
    return vectors.tolist()


def embed_numpy(texts):
    """texts: sequence[str] -> (N, dim) float32 C-contiguous numpy array.

    Preferred for larger batches: the Nim side reads the buffer directly via
    nimpy's `raw_buffers`, so no Python float object is built per element.
    """
    vectors = model.encode(list(texts), convert_to_numpy=True)
    return np.ascontiguousarray(vectors, dtype=np.float32)


def query(query_vector, embeddings, texts, top_k: int = 5, skip_self: bool = True):
    """Rank `texts` by cosine similarity to `query_vector`.

    Same math as articles.py's query() - util.cos_sim over the whole corpus
    followed by a descending sort - but it returns structured hits instead of
    printing them: [(index, score, text), ...].

    `skip_self` drops the top hit, the way articles.py skips the query article
    itself (it always scores ~1.0). Set it to False when the query is not part
    of the corpus.

    Accepts nested Python lists (what nimpy sends) as well as numpy arrays.
    """
    q = np.asarray(query_vector, dtype=np.float32).reshape(1, -1)
    emb = np.asarray(embeddings, dtype=np.float32)

    scores = util.cos_sim(q, emb)[0]
    order = scores.argsort(descending=True)

    start = 1 if skip_self else 0
    hits = []
    for idx in order[start:start + top_k]:
        i = int(idx)
        hits.append((i, float(scores[i].item()), str(texts[i])))
    return hits
