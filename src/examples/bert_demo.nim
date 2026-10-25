## Embed texts and rank them, using the llama.cpp backend directly.
##
## No index needed: this only exercises `llama_embed` (loadModel / embed /
## embedFlat / query).
##
## Build:  nim c -d:release src/examples/bert_demo.nim
## Run:    HNSW_GGUF=~/models/gguf/<model>.gguf ./src/examples/bert_demo
##   or:   nimble runNim          (see the task in ydbhnsw.nimble)
##
## The model file is resolved by `llama_embed.resolveModelFile`: HNSW_GGUF wins,
## then a `.gguf` path, then HNSW_GGUF_DIR/<name>.gguf. No Python is involved.

#import ../ydbhnsw

import ../llama_embed


when isMainModule:
  import std/[strformat, strutils]    

  # This demo never opens an HNSW index, so no `openHnsw` loads a model for it:
  # name it here (the empty argument means `DefaultModel`). `dim()` below needs a
  # loaded model, and so do embed()/embedFlat().
  echo &"model: {loadModel()}"

  # Embedding size of the loaded model (384 for the MiniLM default).
  let embeddingDim = dim()

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

  echo &"embedding dim = {embeddingDim}"

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
  echo &"embedFlat: {flat.len} float32 values ({flat.len div embeddingDim} rows)"

  # Sanity check: both paths must produce the same first vector.
  var dot: float64
  for j in 0 ..< embeddingDim:
    dot += float64(flat[j]) * float64(vectors[0][j])
  echo &"sanity: dot(list[0], flat[0]) = {dot:.6f}  (expected ~1.0)"

  # --- query: rank the corpus against article 0 ----------------------------
  echo "query(article 0, topK = 3):"
  for hit in query(vectors[0], vectors, allTexts, topK = 3):
    echo &"  idx: {hit.index} score: {hit.score:.4f} | {hit.text}"
