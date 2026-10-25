## Embedding model for the index, served by llama.cpp - no Python, no PyTorch.
##
## This replaces the old nimpy/sentence-transformers bridge (`bert_nim.nim` +
## `pyembed.nim`). It provides the same four entry points the rest of the repo
## uses - `loadModel`, `dim`, `embedFlat` (plus `embed`/`embedOne`/`query`) - so
## `hnsw_common`, the examples and the HTTP service are unchanged. `hnsw.nim`
## hands the model name from `HnswParams.model` / the index META to `loadModel`
## through `setModelLoader`, which this module registers at import time.
##
## The model is a GGUF file, resolved from the name the index pins:
##
##   * `HNSW_GGUF` - explicit file, wins over everything (handy when the local
##     file is not named after the model, e.g. an F16 build)
##   * a name that already is a path and ends in `.gguf`
##   * `HNSW_GGUF_DIR/<name>.gguf` (then `-f16.gguf`, `-f32.gguf`),
##     `HNSW_GGUF_DIR` defaults to `$HOME/models/gguf`
##
## Vectors are mean-pooled and L2-normalised, exactly like sentence-transformers
## produced them, and truncated to `HNSW_EMBED_MAX_TOKENS` (default 128, the
## model's `max_seq_length`) so truncation matches the old path. A conversion with
## llama.cpp's `convert_hf_to_gguf.py` reproduces the torch vectors to float32
## precision (<= 2.4e-4 per component on F16); see LLAMA_CPP.md.
##
## Concurrency: a llama_context may only be used by one thread at a time, so
## every public proc runs under `lock`. That is the whole replacement for the old
## GIL dance: the model is loaded once, and workers serialise on the same lock
## the Python calls used to serialise on implicitly.
##
## The C side lives in `llama_shim.c` (compiled in below). It exists because
## llama.h passes `llama_model_params`, `llama_context_params` and `llama_batch`
## by value and those layouts change between releases; the shim is compiled
## against the real header so Nim never has to mirror them.

import std/[algorithm, locks, math, os, strutils]

import hnsw   # DefaultModel + the ModelLoader hook

# Where llama.cpp lives. `config.nims` passes the build directory through
# `-d:llamaCppDir=<build>`; without it we assume a system-wide install.
const llamaCppDir {.strdefine.} = ""
when llamaCppDir.len > 0:
  const llamaBin = llamaCppDir / "bin"
  # llama.h pulls in ggml.h, which lives in the ggml subdirectory.
  {.passC: "-I" & (llamaCppDir / ".." / "include") &
            " -I" & (llamaCppDir / ".." / "ggml" / "include").}
  # The rpath points at the llama.cpp build tree, the same way the YottaDB block
  # in config.nims points at libyottadb; libllama carries its own RUNPATH for
  # libggml, so one entry is enough.
  {.passL: "-L" & llamaBin & " -lllama -Wl,-rpath," & llamaBin.}
else:
  {.passL: "-lllama".}

{.compile: "llama_shim.c".}

const
  DefaultCtx = 512
    ## tokens per decode; also the per-batch token budget (`n_batch`/`n_ubatch`)
  DefaultSeqs = 64
    ## sequences per decode (`n_seq_max`); a batch is also capped by the token budget
  DefaultThreads = 4
    ## llama.cpp compute threads; override with HNSW_EMBED_THREADS
  DefaultMaxTokens = 128
    ## the model's max_seq_length - the old torch path truncated here too
  ErrLen = 512
  PathLen = 1024

type
  Model = pointer   ## struct llama_model *
  Context = pointer ## struct llama_context *
  Batch = pointer   ## struct shim_batch *

  QueryHit* = tuple
    index: int      ## position of the article in `texts`
    score: float32  ## cosine similarity to the query vector
    text: string    ## the article text itself

# --- the C shim (src/llama_shim.c) ------------------------------------------

proc cBackendInit(quiet: cint) {.importc: "shim_backend_init", cdecl.}
proc cBackendFree() {.importc: "shim_backend_free", cdecl.}
proc cLoadModel(path: cstring, nGpuLayers: cint, err: cstring, errLen: csize_t): Model
  {.importc: "shim_load_model", cdecl.}
proc cFreeModel(model: Model) {.importc: "shim_free_model", cdecl.}
proc cNewContext(model: Model, nCtx, nBatch, nSeqMax, nThreads, nThreadsBatch, pooling: cint,
                 err: cstring, errLen: csize_t): Context
  {.importc: "shim_new_context", cdecl.}
proc cFreeContext(ctx: Context) {.importc: "shim_free_context", cdecl.}
proc cNEmbdOut(model: Model): int32 {.importc: "shim_n_embd_out", cdecl.}
proc cPoolingType(ctx: Context): cint {.importc: "shim_pooling_type", cdecl.}
proc cTokenize(model: Model, text: cstring, textLen: cint, outIds: ptr int32,
               outCap: cint, addSpecial, parseSpecial: cint): cint
  {.importc: "shim_tokenize", cdecl.}
proc cSeqEndToken(model: Model): cint {.importc: "shim_seq_end_token", cdecl.}
proc cBatchNew(nTokens, nSeqMax: cint): Batch {.importc: "shim_batch_new", cdecl.}
proc cBatchFree(batch: Batch) {.importc: "shim_batch_free", cdecl.}
proc cBatchReset(batch: Batch) {.importc: "shim_batch_reset", cdecl.}
proc cBatchCapacity(batch: Batch): cint {.importc: "shim_batch_capacity", cdecl.}
proc cBatchAdd(batch: Batch, token, pos, seq: cint): cint
  {.importc: "shim_batch_add", cdecl.}
proc cDecode(ctx: Context, batch: Batch, err: cstring, errLen: csize_t): cint
  {.importc: "shim_decode", cdecl.}
proc cEmbeddingsSeq(ctx: Context, seq: cint): ptr float32
  {.importc: "shim_embeddings_seq", cdecl.}

# --- state ------------------------------------------------------------------
# Only non-GC types here: the service calls the public procs from `{.gcsafe.}`
# handlers, which may not touch GC'ed globals. `loadedPath` is a plain buffer so
# no string is reachable without holding `lock`.
var
  lock: Lock
  backendReady: bool
  model: Model
  ctx: Context
  embdDim: int
  batchTokens: int
  seqMax: int
  maxTokens: int
  loadedPath: array[PathLen, char]

initLock(lock)

# --- helpers ----------------------------------------------------------------

proc cstr(buf: array[PathLen, char]): cstring {.inline.} =
  cast[cstring](unsafeAddr buf[0])

proc errStr(err: array[ErrLen, char]): string =
  $cast[cstring](unsafeAddr err[0])

proc envInt(name: string, fallback: int): int =
  let raw = getEnv(name)
  if raw.len == 0:
    return fallback
  try:
    result = parseInt(raw.strip)
  except ValueError:
    raise newException(ValueError, name & " must be an integer, got '" & raw & "'")

proc envBool(name: string, fallback: bool): bool =
  let raw = getEnv(name)
  if raw.len == 0:
    return fallback
  case raw.strip.toLowerAscii
  of "1", "true", "yes", "on": true
  of "0", "false", "no", "off": false
  else: raise newException(ValueError, name & " must be true/false, got '" & raw & "'")

proc envPooling(): cint =
  ## `llama_pooling_type` values; -1 would mean "whatever the model says", but
  ## mean is what this model was trained with.
  case getEnv("HNSW_EMBED_POOLING", "mean").strip.toLowerAscii
  of "", "mean": 1
  of "cls": 2
  of "last": 3
  of "none": 0
  else:
    raise newException(ValueError,
      "HNSW_EMBED_POOLING must be mean, cls, last or none")

proc resolveModelFile(name: string): string =
  ## Name (as pinned in the index) -> GGUF path. See the module doc for the order.
  let explicit = getEnv("HNSW_GGUF")
  if explicit.len > 0:
    if not fileExists(explicit):
      raise newException(IOError, "HNSW_GGUF points to a file that does not exist: " & explicit)
    return explicit

  let wanted = if name.len == 0: DefaultModel else: name
  if wanted.endsWith(".gguf") or DirSep in wanted:
    if fileExists(wanted):
      return wanted
    raise newException(IOError, "embedding model file not found: " & wanted)

  let dir = getEnv("HNSW_GGUF_DIR", getHomeDir() / "models" / "gguf")
  var tried: seq[string]
  for suffix in [".gguf", "-f16.gguf", "-f32.gguf"]:
    let candidate = dir / (wanted & suffix)
    if fileExists(candidate):
      return candidate
    tried.add candidate

  raise newException(IOError,
    "no GGUF file for embedding model '" & wanted & "' (tried " & tried.join(", ") & "). " &
    "Convert the model with llama.cpp's convert_hf_to_gguf.py, or set HNSW_GGUF " &
    "to the file / HNSW_GGUF_DIR to the directory that holds it.")

proc ensureLoaded() =
  if ctx == nil or model == nil:
    raise newException(ValueError,
      "no embedding model loaded - open an HNSW index (its META names the model) " &
      "or call loadModel() first")

proc normalize(vec: var seq[float32], first, last: int) =
  ## L2 normalise `vec[first ..< last]`, the way the index normalises on insert.
  var sum = 0.0'f32
  for i in first ..< last:
    sum += vec[i] * vec[i]
  if sum <= 0.0'f32:
    return
  let inv = 1.0'f32 / sqrt(sum)
  for i in first ..< last:
    vec[i] *= inv

proc tokenizeInto(text: string, ids: var seq[int32]) =
  ## Token ids for one text, truncated to `maxTokens` while keeping the closing
  ## separator (SentencePiece `</s>`), so a cut text still ends like a sentence.
  var needed = cTokenize(model, text.cstring, cint(text.len), nil, 0, 1, 0)
  if needed == cint(low(int32)):
    raise newException(ValueError, "text is too long to tokenize")
  if needed < 0:
    needed = -needed
  ids.setLen(int(needed))
  if needed == 0:
    return
  let got = cTokenize(model, text.cstring, cint(text.len), addr ids[0], needed, 1, 0)
  if got < 0:
    raise newException(ValueError,
      "tokenization failed: " & text[0 ..< min(60, text.len)])
  ids.setLen(int(got))
  if ids.len > maxTokens:
    let closing = ids[^1]
    ids.setLen(maxTokens)
    ids[^1] = closing

proc embedAll(texts: seq[string]): seq[float32] =
  ## Caller holds `lock`. Packs as many texts as fit into each decode.
  ensureLoaded()
  result = newSeq[float32](texts.len * embdDim)

  let batch = cBatchNew(cint(batchTokens), cint(seqMax))
  if batch == nil:
    raise newException(ResourceExhaustedError,
      "cannot allocate a llama_batch of " & $batchTokens & " tokens")
  defer: cBatchFree(batch)

  var err: array[ErrLen, char]
  var ids: seq[int32] = @[]
  var next = 0
  while next < texts.len:
    cBatchReset(batch)
    var used = 0
    var seqs = 0
    while next < texts.len and seqs < seqMax:
      tokenizeInto(texts[next], ids)
      if used + ids.len > batchTokens:
        if seqs == 0:
          raise newException(ValueError,
            "text " & $next & " needs " & $ids.len & " tokens, more than the " &
            $batchTokens & " the context allows; raise HNSW_EMBED_CTX or lower " &
            "HNSW_EMBED_MAX_TOKENS")
        break
      for pos, tok in ids:
        if cBatchAdd(batch, cint(tok), cint(pos), cint(seqs)) < 0:
          raise newException(ValueError, "internal error: batch overflow")
      used += ids.len
      inc seqs
      inc next

    let rc = cDecode(ctx, batch, cast[cstring](unsafeAddr err[0]), csize_t(err.len))
    if rc < 0:
      raise newException(IOError, "embedding failed: " & errStr(err))

    let firstRow = next - seqs
    for s in 0 ..< seqs:
      let src = cEmbeddingsSeq(ctx, cint(s))
      if src == nil:
        raise newException(IOError, "no embedding returned for sequence " & $s)
      let base = (firstRow + s) * embdDim
      copyMem(addr result[base], src, embdDim * sizeof(float32))
      normalize(result, base, base + embdDim)

# --- public API -------------------------------------------------------------

proc loadModel*(name: string = hnsw.DefaultModel): string =
  ## Load (or keep) the GGUF for `name` and return the name it is loaded under.
  ##
  ## `openHnsw` calls this through `hnsw.setModelLoader`, so the index's META
  ## decides which model gets loaded; a program that embeds without opening an
  ## index calls it directly. Loading the same file again is a no-op.
  let wanted = if name.len == 0: DefaultModel else: name
  acquire(lock)
  defer: release(lock)

  let path = resolveModelFile(wanted)
  if ctx != nil and cstr(loadedPath) == path.cstring:
    return wanted

  if not backendReady:
    cBackendInit(cint(if envBool("HNSW_LLAMA_LOG", false): 0 else: 1))
    backendReady = true

  var err: array[ErrLen, char]

  # Release the previous model before loading the next one: a GGUF is ~240 MB
  # and an index reopen with a different model should not hold both.
  if ctx != nil:
    cFreeContext(ctx)
    ctx = nil
  if model != nil:
    cFreeModel(model)
    model = nil
  loadedPath[0] = '\0'

  let nGpu = envInt("HNSW_EMBED_GPU_LAYERS", 0)
  let fresh = cLoadModel(path.cstring, cint(nGpu),
                         cast[cstring](unsafeAddr err[0]), csize_t(err.len))
  if fresh == nil:
    raise newException(IOError, errStr(err))

  let nCtx = envInt("HNSW_EMBED_CTX", DefaultCtx)
  let nSeqs = envInt("HNSW_EMBED_SEQS", DefaultSeqs)
  let nThreads = envInt("HNSW_EMBED_THREADS", DefaultThreads)
  if nCtx <= 0:
    cFreeModel(fresh)
    raise newException(ValueError, "HNSW_EMBED_CTX must be positive")
  if nSeqs <= 0:
    cFreeModel(fresh)
    raise newException(ValueError, "HNSW_EMBED_SEQS must be positive")

  let freshCtx = cNewContext(fresh, cint(nCtx), cint(nCtx), cint(nSeqs),
                             cint(nThreads), cint(nThreads),
                             envPooling(),
                             cast[cstring](unsafeAddr err[0]), csize_t(err.len))
  if freshCtx == nil:
    cFreeModel(fresh)
    raise newException(IOError, errStr(err))

  model = fresh
  ctx = freshCtx
  embdDim = int(cNEmbdOut(fresh))
  batchTokens = nCtx
  seqMax = nSeqs
  # The model's max_seq_length is 128; never tokenize past the batch either.
  let requested = envInt("HNSW_EMBED_MAX_TOKENS", DefaultMaxTokens)
  maxTokens = if requested <= 0: batchTokens else: min(requested, batchTokens)

  # Record the path that is actually open; written last, only on success.
  let trimmed = if path.len >= PathLen: path[0 ..< PathLen - 1] else: path
  for i, ch in trimmed:
    loadedPath[i] = ch
  loadedPath[trimmed.len] = '\0'

  result = wanted

proc dim*(): int =
  ## Embedding width of the loaded model (384 for the MiniLM default).
  acquire(lock)
  defer: release(lock)
  ensureLoaded()
  result = embdDim

proc embedFlat*(texts: seq[string]): seq[float32] =
  ## Fast path: `texts.len * dim` floats, row-major (row i starts at i * dim).
  if texts.len == 0:
    return @[]
  acquire(lock)
  defer: release(lock)
  result = embedAll(texts)

proc embed*(texts: seq[string]): seq[seq[float32]] =
  ## One vector per text.
  if texts.len == 0:
    return @[]
  acquire(lock)
  defer: release(lock)
  let flat = embedAll(texts)
  result = newSeq[seq[float32]](texts.len)
  for i in 0 ..< texts.len:
    result[i] = flat[i * embdDim ..< (i + 1) * embdDim]

proc embedOne*(text: string): seq[float32] =
  ## Embedding of a single text, callable from any thread (the HTTP service uses
  ## this for one article or one query per request).
  acquire(lock)
  defer: release(lock)
  result = embedAll(@[text])

proc query*(
    queryVector: seq[float32],
    embeddings: seq[seq[float32]],
    texts: seq[string],
    topK: int = 5,
    skipSelf: bool = true,
): seq[QueryHit] =
  ## Top-`topK` articles by cosine similarity to `queryVector`.
  ##
  ## Both sides are normalised, so this is a plain dot product - the same
  ## ordering `util.cos_sim` gave the old Python bridge. `skipSelf` drops the best
  ## hit (the query article matching itself at ~1.0); pass `skipSelf = false` when
  ## the query text is not part of the corpus.
  if embeddings.len != texts.len:
    raise newException(ValueError,
      "embeddings (" & $embeddings.len & ") and texts (" & $texts.len & ") differ in length")

  var scored = newSeq[tuple[score: float32, index: int]](embeddings.len)
  for i, vec in embeddings:
    if vec.len != queryVector.len:
      raise newException(ValueError, "vector " & $i & " has " & $vec.len &
        " dimensions, the query has " & $queryVector.len)
    var dot = 0.0'f32
    for j in 0 ..< vec.len:
      dot += vec[j] * queryVector[j]
    scored[i] = (score: dot, index: i)

  # Descending by score, ties broken by corpus position: deterministic order.
  scored.sort(proc (a, b: tuple[score: float32, index: int]): int =
    if a.score != b.score: (if a.score > b.score: -1 else: 1)
    else: cmp(a.index, b.index))

  let first = if skipSelf and scored.len > 0: 1 else: 0
  for i in first ..< min(scored.len, first + topK):
    result.add (index: scored[i].index, score: scored[i].score,
                text: texts[scored[i].index])

# hnsw must not depend on the embedding backend, so the way back is registered
# here: every `openHnsw` that names a model lands in `loadModel` above.
hnsw.setModelLoader(loadModel)
