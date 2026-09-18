## Call the Python sentence-transformers pipeline from Nim, via nimpy.
##
## Build:  nimble build
## Run:    nimble runNim          (sets PYTHONPATH to the venv, see bert.nimble)
##   or:   PYTHONPATH=/home/ljoeckel/hnsw_env/lib/python3.12/site-packages ./bert_nim
##
## nimpy embeds a CPython interpreter into this binary (it dlopens libpython at
## runtime), so this process *becomes* a Python host. Everything the Python code
## needs - sentence_transformers, numpy, torch - must be importable by that
## interpreter. Since we embed the system libpython3.12, the venv's
## site-packages are supplied through PYTHONPATH.

import bakery
import nimpy
import nimpy/raw_buffers   # for the zero-copy numpy path

# bakery registers this in sys.modules at init; take the module from there
# rather than pyImport("sbert_bridge"), so the import is a real dependency.
let bridge = bakery.bridge


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

# ------------------------------------------------------------------------

when isMainModule:
  import std/[strformat, strutils]    

  # Embedding size of the loaded model, asked from Python (384 for MiniLM-L6).
  let dim = bridge.dim().to(int)

  let texts = @[
    "MacMini has a M4 chip",
    "Apple releases new MacBook Pro with M4 chip",
    "Tesla announces price cuts across its EV lineup",
    "New MacBook Pro benchmarks show big performance gains",
    "Federal Reserve signals possible rate cut",
    "The MacMini has very high computational power with the new M4 chip",
    "Apple does not make a macmini with a M5 chip",
  ]

  let texts2 = @[
    "NVidea has a new GPU chip",
    "Putin killed by an assanin",
    "Apple shut's down russian shops",
    "M5 chip is much faster than before",
    "DGX-1 was the first AI machine from NVidea",
    "MacMini has doubled price due to memory shortness",
  ]

  echo &"python {pyImport(\"sys\").version.to(string).split(chr(10))[0]}, embedding dim = {dim}"

  # Both article groups make up one corpus. Row i of `vectors` has to line up
  # with allTexts[i]; indexing into `texts` alone breaks as soon as `texts2`
  # is appended (it is shorter, so i runs off the end of it).
  let allTexts = texts & texts2

  var vectors: seq[seq[float32]]
  # --- simple path: seq[seq[float32]] ---------------------------------------
  vectors.add(embed(texts))
  echo &"embed: {vectors.len} vectors x {vectors[0].len} floats"
  vectors.add(embed(texts2))
  echo &"embed: {vectors.len} vectors x {vectors[0].len} floats"

  for i, v in vectors:
    echo &"  {i}: [{v[0]:.4f} {v[1]:.4f} {v[2]:.4f} ...]  {allTexts[i]}"

  # --- fast path: flat float32 buffer --------------------------------------
  let flat = embedFlat(texts)
  echo &"embedFlat: {flat.len} float32 values ({flat.len div dim} rows)"

  # Sanity check: both paths must produce the same first vector.
  var dot: float64
  for j in 0 ..< dim:
    dot += float64(flat[j]) * float64(vectors[0][j])
  echo &"sanity: dot(list[0], flat[0]) = {dot:.6f}  (expected ~1.0)"

  # --- query: rank the corpus against article 0 ----------------------------
  echo "query(article 0, topK = 3):"
  for hit in query(vectors[0], vectors, allTexts, topK = 3):
    echo &"  idx: {hit.index} score: {hit.score:.4f} | {hit.text}"
