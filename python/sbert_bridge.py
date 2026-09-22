"""Bridge between the Python/SBERT pipeline and Nim (see `bert_nim.nim`).

The Nim side imports this module with nimpy, then loads a model with `load()` -
which happens when `openHnsw()` opens an index, from `HnswParams.model` (see
`hnsw.nim`) - and after that calls `embed()` / `embed_numpy()` / `query()`.
Keep the public API here small and boring - it is the contract that crosses the
language boundary, so renaming things breaks the Nim build (see `bert_nim.nim`).
"""

import os

# --- threading: configured before torch (and MKL) are imported --------------
# HISTORICAL - the old box (Xeon E5-1620 v2 / Ivy Bridge) has AVX and SSE4.2 but
# no AVX2/AVX-512, while the Intel MKL bundled with torch was observed executing
# an AVX-512 kernel from a worker thread (vmovdqu64 ... %zmm13, inside
# mkl_vml_kernel_sTanh_Z0HAynn) -> SIGILL / exit 132, intermittent and only
# under concurrent load. That is why the default here is a single thread, on top
# of ~/.bashrc exporting OMP_NUM_THREADS=1 MKL_NUM_THREADS=1.
#
# The current box is a Core Ultra 7 155H (AVX2, no AVX-512) and its shell does
# not export either variable, so single threaded is now just the conservative
# default, not a necessity. Opt in deliberately, with MKL_ENABLE_INSTRUCTIONS as
# the guardrail that pins MKL to a kernel class the CPU really supports, should
# the AVX-512 crash ever come back (it is not set from here on purpose: it would
# also cap MKL on a machine that could legitimately use AVX2/AVX-512):
#
#   SBERT_THREADS=4 MKL_ENABLE_INSTRUCTIONS=SSE4_2 nimble runHnsw
#
# Threads only pay off for *batched* calls, and only modestly: a batch amortises
# the per-call overhead (Python dispatch, tokenizer, kernel launches) over many
# texts, and is the only case where the matrix work is big enough to split.
# Measured on the current box, 2000 headlines embedded one call at a time:
# 23.3 s with SBERT_THREADS=1 and 23.6 s with 4 (on the old box: 1 thread
# 55 texts/s, 4 threads 193 texts/s for batches of 256). So a caller that embeds
# one text per call cannot be fixed with a thread count - it has to batch first
# (hnsw_search.nim does, in bulks of 128: 23.3 s -> 1.4 s for the same work).
THREADS = max(1, int(os.environ.get("SBERT_THREADS", "1")))
if THREADS > 1:
    # MKL prioritises MKL_NUM_THREADS over OMP_NUM_THREADS, and the old shell
    # exported both as 1, so torch.set_num_threads() alone would leave MKL
    # single threaded. These have to be in place before the libraries load.
    os.environ["OMP_NUM_THREADS"] = str(THREADS)
    os.environ["MKL_NUM_THREADS"] = str(THREADS)

import numpy as np
import torch
from sentence_transformers import SentenceTransformer, util
from transformers.utils import logging

logging.set_verbosity_error()   # hides the LOAD REPORT and misc warnings
torch.set_num_threads(THREADS)

# --- stop tqdm from registering a multiprocessing semaphore -------------------
# The first progress bar (transformers' "Loading weights" here, and the
# "Batches" bar that encode() builds via trange) makes tqdm create a
# class-level multiprocessing.RLock - tqdm/std.py, TqdmDefaultWriteLock.
# create_mp_lock. Two consequences in this embedded host:
#
#   * Python 3.14 defaults to the "forkserver" start method, so that SemLock
#     gets a name and multiprocessing/synchronize.py:78 registers it with the
#     resource tracker; the matching unregister is deferred to interpreter
#     shutdown (util.Finalize, exitpriority=0). There is no interpreter
#     shutdown here (nimpy never finalises), so the tracker reports
#     "There appear to be 1 leaked semaphore objects ... {'/loky-...'}" at the
#     end of every run. It is only cosmetic - the tracker unlinks the semaphore
#     right after warning, /dev/shm stays empty and the exit code stays 0 - but
#     it is noise on stderr after the results.
#   * The "/loky-" prefix is misleading: joblib/loky (pulled in transitively by
#     sentence-transformers) monkey-patches the *stdlib* SemLock name factory.
#     No loky worker is ever started.
#
# A None mp_lock leaves tqdm with its threading lock only, which is all a
# single-process writer needs. transformers.utils.logging.disable_progress_bar()
# would not be enough - it does not cover sentence-transformers' own trange.
import tqdm.std

tqdm.std.TqdmDefaultWriteLock.mp_lock = None

# The model is deliberately *not* loaded here: the Nim side names it in
# `HnswParams.model` and loads it through `load()` when the index is opened, so
# which model an index holds is decided per index (and recorded in the index's
# META) instead of being baked into this module.
_model = None         # the SentenceTransformer, once load() has run
_model_name = None    # the name it was loaded under, None before that


def load(model_name: str) -> str:
    """Load the sentence-transformers model `model_name`; return the name used.

    Called once per index open, so loading the same name again is a no-op: the
    weights are already resident and every call coming from Nim reuses them.
    Loading a *different* name replaces the model - vectors are only comparable
    to vectors from the same model, which is why an index pins its model in META
    the way it pins its dimension and storage mode.
    """
    global _model, _model_name
    if _model is None or model_name != _model_name:
        _model = SentenceTransformer(model_name)
        _model_name = model_name
    return _model_name


def model_name() -> str:
    """Name the model was loaded under, or "" while nothing is loaded."""
    return _model_name or ""


def _loaded():
    if _model is None:
        raise RuntimeError(
            "no embedding model loaded - set HnswParams.model and open the "
            "index first (openHnsw loads it), or call sbert_bridge.load(name)")
    return _model


def dim() -> int:
    """Embedding size of the loaded model (384 for the MiniLM models)."""
    # Renamed in sentence-transformers 6; this env is pinned to 6.0.1.
    return int(_loaded().get_embedding_dimension())


def embed(texts) -> list[list[float]]:
    """texts: sequence[str] -> one plain-Python vector per text.

    Returning plain lists keeps the nimpy conversion trivial on the Nim side:
    `bridge.embed(texts).to(seq[seq[float32]])`.
    """
    vectors = _loaded().encode(list(texts), convert_to_numpy=True)
    return vectors.tolist()


def embed_numpy(texts):
    """texts: sequence[str] -> (N, dim) float32 C-contiguous numpy array.

    Preferred for larger batches: the Nim side reads the buffer directly via
    nimpy's `raw_buffers`, so no Python float object is built per element.
    """
    vectors = _loaded().encode(list(texts), convert_to_numpy=True)
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
