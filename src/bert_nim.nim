## Call the Python sentence-transformers pipeline from Nim, via nimpy.
##
## Build:  nimble build
## Run:    nimble runNim          (sets PYTHONPATH to the venv, see ydbhnsw.nimble)
##   or:   PYTHONPATH=/home/ljoeckel/git/ydb-hnsw/hnsw_env/lib/python3.12/site-packages ./bert_nim
##
## nimpy embeds a CPython interpreter into this binary (it dlopens libpython at
## runtime), so this process *becomes* a Python host. Everything the Python code
## needs - sentence_transformers, numpy, torch - must be importable by that
## interpreter. Since we embed the system libpython3.12, the venv's
## site-packages are supplied through PYTHONPATH.
##
## `python/sbert_bridge.py` is imported as the ordinary Python module
## `sbert_bridge` (see below), so it stays a real file: changing it takes a
## restart of the binary, not a rebuild.

import std/os

import hnsw         # `ModelLoader`: what openHnsw calls back for the model
import nimpy
import nimpy/raw_buffers   # for the zero-copy numpy path

# The bridge is imported from disk like any other Python module - no source is
# baked into the binary. Its directory goes on sys.path first, so neither the
# working directory nor PYTHONPATH has to name it; the path is resolved at
# compile time (absolute, independent of where the binary is started from), the
# same way a deployed binary would ship the .py next to itself.
const bridgeDir = currentSourcePath().parentDir.parentDir / "python"
discard pyImport("sys").path.insert(0, bridgeDir)

let bridge = pyImport("sbert_bridge")


proc loadModel*(name: string = hnsw.DefaultModel): string =
  ## Load the sentence-transformers model `name` (empty means `DefaultModel`) and
  ## return the name it is loaded under.
  ##
  ## `openHnsw` calls this through `hnsw.setModelLoader`, so `HnswParams.model`
  ## decides which weights an index embeds with - and META decides for an index
  ## that already has vectors. A program that embeds without opening an index
  ## calls it directly (see `examples/bert_demo.nim`).
  ##
  ## Loading is idempotent per name: the same name keeps the model already in
  ## memory, a different one replaces it.
  bridge.load(if name.len == 0: hnsw.DefaultModel else: name).to(string)


# hnsw must not depend on Python, so the way back is registered here: every
# `openHnsw` that names a model lands in `loadModel` above.
hnsw.setModelLoader(loadModel)


proc dim*(): int =
  ## Embedding size of the loaded model (384 for the MiniLM models), asked from
  ## Python. Needs a loaded model: `openHnsw` loads one, `loadModel` does it on
  ## request, and `sbert_bridge` raises a clear error if neither has happened.
  bridge.dim().to(int)


proc embed*(texts: seq[string]): seq[seq[float32]] =
  ## One embedding per text: `texts.len` vectors of `dim` floats.
  ##
  ## Simple path - Python returns nested lists and nimpy converts them element
  ## by element. Fine for a handful of articles or for interactive use.
  bridge.embed(texts).to(seq[seq[float32]])


proc embedFlat*(texts: seq[string]): seq[float32] =
  ## Fast path: `texts.len * dim` floats, row-major (row i starts at i * dim).
  ##
  ## Python returns a float32 numpy array and we copy its buffer directly, so
  ## no Python object is allocated per element. Use this for real batches.
  let arr = bridge.embed_numpy(texts)

  var buf: RawPyBuffer
  # Ask for a read-only C-contiguous view: unlike a plain PyBUF_READ request
  # this fills in ndim/shape/strides, and numpy refuses if the array is not
  # contiguous (so the flat copy below is always valid).
  arr.getBuffer(buf, PyBUF_C_CONTIGUOUS.cint)
  defer: buf.release()

  doAssert buf.ndim == 2, "expected a 2-D embedding matrix"
  doAssert buf.itemsize == 4, "expected float32 data"

  # buf.shape is a C array of ndim entries; Nim 2 dropped `[]` on raw `ptr`.
  # (Py_ssize_t is not re-exported by nimpy; Nim's int has the same size.)
  let shape = cast[ptr UncheckedArray[int]](buf.shape)
  let rows = shape[0].int
  let cols = shape[1].int
  let n = rows * cols

  if n == 0:
    return

  result = newSeq[float32](n)
  copyMem(result[0].addr, buf.buf, n * sizeof(float32))


type
  QueryHit* = tuple
    index: int      ## position of the article in `texts`
    score: float32  ## cosine similarity to the query vector
    text: string    ## the article text itself


proc query*(
    queryVector: seq[float32],
    embeddings: seq[seq[float32]],
    texts: seq[string],
    topK: int = 5,
    skipSelf: bool = true,
): seq[QueryHit] =
  ## Top-`topK` articles by cosine similarity to `queryVector`.
  ##
  ## Mirrors articles.py's query(): the ranking is done by Python's
  ## `util.cos_sim` over the whole corpus, we just get the hits back as Nim
  ## data instead of printed lines. `skipSelf` drops the best hit (the query
  ## article matching itself at ~1.0); pass `skipSelf = false` when the query
  ## text is not part of the corpus.
  bridge.query(queryVector, embeddings, texts, topK, skipSelf).to(seq[QueryHit])