## Accuracy check for `hnsw`'s int8 scalar quantization (SQ).
##
## Two vector sources, because synthetic noise and real embeddings fail
## differently: uniform random vectors have a known, flat distribution, while the
## stored SBERT vectors are what the quantizer will actually see in the index.
##
## Measures the round-trip error (max / mean cosine similarity after
## quantization) and, more importantly for search, how much the *ranking* moves:
## the top-10 overlap between exact dot products and `dotInt8` over the same
## candidates.
##
## Run: nim c -r hnsw_quant_test

import std/[algorithm, math, random, sequtils, strformat, times]
import hnsw, yottadb

const
  Dim = 384
  Global = "^HNSWArticles"
  SampleN = 60000         # real vectors pulled out of YottaDB
  Queries = 20           # ranking comparisons
  K = 10                 # top-k overlap
  IndexN = 2000          # vectors per scratch HNSW index
  GraphQueries = 20      # queries against those indexes
  ScratchGlobal = "^HNSWQT"
  TightGlobal = "^HNSWQTTIGHT"

proc unpackFloats(s: string): seq[float32] =
  let n = s.len div sizeof(float32)
  result = newSeq[float32](n)
  if n > 0:
    copyMem(result[0].addr, s[0].unsafeAddr, n * sizeof(float32))

proc dot(a, b: seq[float32]): float32 =
  for i in 0 ..< a.len:
    result += a[i] * b[i]

proc report(name: string, xs: seq[float32]) =
  var lo = Inf.float32
  var hi = -Inf.float32
  var sum = 0.0'f32
  for x in xs:
    lo = min(lo, x)
    hi = max(hi, x)
    sum += x
  echo &"  {name:<14} min {lo:.6f}  mean {sum / float32(xs.len):.6f}  max {hi:.6f}"

proc randomVecs(n: int, rng: var Rand): seq[seq[float32]] =
  for i in 0 ..< n:
    var v = newSeq[float32](Dim)
    for d in 0 ..< Dim:
      v[d] = float32(2.0 * rand(rng, 1.0) - 1.0)
    normalize(v)
    result.add v

proc storedVecs(n: int): seq[seq[float32]] =
  ## `n` vectors spread evenly over the id space, skipping id holes.
  let count = nodeCount(Global)
  if count == 0:
    return
  for k in 0 ..< n:
    let id = k * count div n
    let s = ydb_get(Global & "NODE", @[$id, "vec"])
    if s.len == Dim * sizeof(float32):
      result.add unpackFloats(s)

proc codeStats(tag: string, vecs: seq[seq[float32]], codes: seq[seq[int8]],
               scales: seq[float32]) =
  ## Genuine grid overflow (a component that had to be clamped, not merely one
  ## that lands on the extreme code - with a per-vector scale the maximum
  ## component reaches +-127 by construction) and how many of the 255 levels the
  ## data actually uses, which is the resolution that is really available.
  var saturated = 0
  var used: set[int8]
  for i, v in vecs:
    for x in v:
      if abs(int(round(x / scales[i]))) > 127:
        inc saturated
  for c in codes:
    for x in c:
      used.incl x
  let components = vecs.len * Dim
  echo &"  {tag}: clamped {saturated} / {components} components " &
       &"({100.0 * float(saturated) / float(components):.4f} %), " &
       &"{used.len} of 255 codes used"

proc roundTripCosine(tag: string, vecs: seq[seq[float32]],
                     codes: seq[seq[int8]], scale: float32) =
  var cosines: seq[float32]
  var lengths: seq[float32]
  for i, v in vecs:
    let back = dequantizeInt8(codes[i], scale)
    cosines.add dot(v, back)
    var sum = 0.0'f32
    for x in back:
      sum += x * x
    lengths.add sqrt(sum)      # > 1 explains the odd cos(v, v~) above 1.0
  echo &"\n[{tag}] scale {scale:.8f}"
  report("cos(v, v~)", cosines)
  report("|v~|", lengths)
  codeStats(tag, vecs, codes, repeat(scale, vecs.len))

proc rankOverlap(tag: string, vecs: seq[seq[float32]], codes: seq[seq[int8]],
                 scales: seq[float32]) =
  ## Use each of the first `Queries` vectors as a query, rank all other
  ## candidates by exact dot product and by `dotInt8` (each vector with its own
  ## scale, so a per-vector grid is compared correctly too), and look at how far
  ## the top-k sets and the top-1 pick have moved.
  var overlaps: seq[float32]
  var top1 = 0
  for q in 0 ..< Queries:
    var exact: seq[(float32, int)]
    var approx: seq[(float32, int)]
    for j in 0 ..< vecs.len:
      if j == q:
        continue
      exact.add (dot(vecs[q], vecs[j]), j)
      approx.add (dotInt8(codes[q], codes[j], scales[q], scales[j]), j)
    exact.sort(proc(a, b: (float32, int)): int = cmp(b[0], a[0]))
    approx.sort(proc(a, b: (float32, int)): int = cmp(b[0], a[0]))
    var hit = 0
    for i in 0 ..< K:
      for j in 0 ..< K:
        if exact[i][1] == approx[j][1]:
          inc hit
    overlaps.add float32(hit) / float32(K)
    if exact[0][1] == approx[0][1]:
      inc top1
  echo &"[{tag}] ranking over {vecs.len - 1} candidates"
  report(&"top-{K} overlap", overlaps)
  echo &"  identical top-1: {top1} / {Queries} queries"

proc resetIndex(global: string) =
  for g in [global.nodeGlobal, global.keyGlobal, global.metaGlobal]:
    ydb_delete(g, @[], YDB_DEL_TREE)

proc vecBytes(ix: HnswIndex): int =
  ## Bytes of vector payload actually on disk, summed over the nodes.
  for id in 0 ..< ix.count:
    result += ydb_get(ix.params.globalNode, @[$id, "vec"]).len

proc exactTop(vecs: seq[seq[float32]], qi, k: int): seq[int] =
  ## Brute-force nearest neighbours of `vecs[qi]`, the query's own node excluded.
  var scored: seq[(float32, int)]
  for j in 0 ..< vecs.len:
    if j != qi:
      scored.add (dot(vecs[qi], vecs[j]), j)
  scored.sort(proc(a, b: (float32, int)): int = cmp(b[0], a[0]))
  for i in 0 ..< min(k, scored.len):
    result.add scored[i][1]

proc graphTop(ix: HnswIndex, q: seq[float32], qi, k: int): seq[int] =
  ## The index's own answer, with the query's node filtered out so it is
  ## comparable with `exactTop`.
  for h in ix.search(q, k = k + 1, ef = 64):
    if h.id != qi:
      result.add h.id
    if result.len == k:
      break

proc sameTop(a, b: seq[int]): int =
  ## How many ids the two top-k lists have in common.
  for x in a:
    for y in b:
      if x == y:
        inc result

when isMainModule:
  var rng = initRand(7)

  echo "=== storage ==="
  echo &"  dim {Dim}: float32 blob {Dim * sizeof(float32)} B vs " &
       &"int8 blob {Dim} B (4.0x smaller)"
  let sample = storedVecs(SampleN)
  echo &"  sampled {sample.len} live vectors from {Global}NODE"

  # --- synthetic, per-vector grid ------------------------------------------
  echo "\n=== uniform random vectors (L2-normalised), per-vector scale ==="
  let synth = randomVecs(200, rng)
  var synthCodes: seq[seq[int8]]
  var synthScales: seq[float32]
  var synthCos: seq[float32]
  var synthErr: seq[float32]
  for v in synth:
    let (codes, scale) = quantizeInt8(v)
    synthCodes.add codes
    synthScales.add scale
    let back = dequantizeInt8(codes, scale)
    synthCos.add dot(v, back)
    for d in 0 ..< Dim:
      synthErr.add abs(v[d] - back[d])
  report("cos(v, v~)", synthCos)
  report("abs err", synthErr)

  echo "\n=== dotInt8 vs dot(dequantized), random vectors ==="
  var synthDiff: seq[float32]
  for i in 0 ..< 20:
    for j in 1 ..< 20:
      let a = dequantizeInt8(synthCodes[i], synthScales[i])
      let b = dequantizeInt8(synthCodes[j], synthScales[j])
      synthDiff.add abs(dot(a, b) - dotInt8(synthCodes[i], synthCodes[j],
                                            synthScales[i], synthScales[j]))
  report("abs diff", synthDiff)

  # --- real stored vectors, shared grid ------------------------------------
  if sample.len > 1:
    echo "\n=== stored SBERT vectors, one shared scale for the collection ==="
    let gscale = quantizeScale(sample)
    var codes: seq[seq[int8]]
    for v in sample:
      codes.add quantizeInt8(v, gscale)
    roundTripCosine("shared scale", sample, codes, gscale)
    rankOverlap("shared scale", sample, codes, repeat(gscale, sample.len))

    echo "\n=== stored SBERT vectors, per-vector scale ==="
    var ownCodes: seq[seq[int8]]
    var ownScales: seq[float32]
    var ownCos, shareCos: seq[float32]
    for i, v in sample:
      let (c, s) = quantizeInt8(v)
      ownCodes.add c
      ownScales.add s
      ownCos.add dot(v, dequantizeInt8(c, s))
      shareCos.add dot(v, dequantizeInt8(codes[i], gscale))
    report("cos own scale", ownCos)
    report("cos shared", shareCos)
    codeStats("own scale", sample, ownCodes, ownScales)
    rankOverlap("own scale", sample, ownCodes, ownScales)

    echo "\n=== blob round-trip ==="
    var ok = true
    for c in codes:
      let packed = packInt8(c)
      if packed.len != Dim or unpackInt8(packed) != c:
        ok = false
    echo &"  packInt8/unpackInt8: " &
         (if ok: &"all {codes.len} blobs byte-identical at {Dim} B"
          else: "MISMATCH")

    echo "\n=== dotInt8 vs dot(dequantized), stored vectors ==="
    var realDiff: seq[float32]
    for i in 0 ..< Queries:
      for j in 1 ..< 50:
        let a = dequantizeInt8(codes[i], gscale)
        let b = dequantizeInt8(codes[j], gscale)
        realDiff.add abs(dot(a, b) - dotInt8(codes[i], codes[j], gscale, gscale))
    report("abs diff", realDiff)

  # --- end to end: real HNSW indexes, one per storage mode ----------------
  if sample.len > IndexN:
    let indexVecs = sample[0 ..< IndexN]
    let grid = quantizeScale(indexVecs)
    echo "\n=== HNSW index built in each storage mode ==="
    echo &"  {IndexN} real vectors, dim {Dim}, M=16 efConstruction=100 efSearch=64, " &
         &"vqInt8Fixed grid {grid:.8f}"

    for mode in [vqNone, vqInt8, vqInt8Fixed]:
      resetIndex(ScratchGlobal)
      let buildStart = cpuTime()
      var ix = openHnsw(hnswParams(ScratchGlobal, M = 16, efConstruction = 100,
                                   efSearch = 64, dim = Dim, quant = mode,
                                   quantScale = grid))
      for v in indexVecs:
        discard ix.add(v)
      let buildMs = (cpuTime() - buildStart) * 1000.0

      var overlaps: seq[float32]
      var top1 = 0
      var searchSeconds = 0.0
      for qi in 0 ..< GraphQueries:
        let truth = exactTop(indexVecs, qi, K)      # brute force, not timed
        let started = cpuTime()
        let got = graphTop(ix, indexVecs[qi], qi, K)
        searchSeconds += cpuTime() - started
        overlaps.add float32(sameTop(truth, got)) / float32(K)
        if got.len > 0 and truth.len > 0 and got[0] == truth[0]:
          inc top1
      let perQueryUs = searchSeconds * 1e6 / float64(GraphQueries)

      # The mode must come back from META, not from the arguments - also when the
      # argument contradicts what is on disk.
      let reopened = openHnsw(HnswParams(global: ScratchGlobal, quant: vqNone))
      let contested = openHnsw(
        HnswParams(global: ScratchGlobal,
                   quant: if mode == vqNone: vqInt8 else: vqNone))

      echo &"\n[{mode}] count={ix.count} live={ix.liveCount} " &
           &"{float(vecBytes(ix)) / float(IndexN):.1f} B/vector stored " &
           &"(float32 {4 * Dim} B), {ix.clamped} clamped"
      echo &"  build {buildMs:.0f} ms ({buildMs * 1000.0 / float(IndexN):.0f} us/insert)"
      report(&"top-{K} overlap", overlaps)
      echo &"  identical top-1 {top1} / {GraphQueries}, search {perQueryUs:.0f} us/query"
      echo &"  reopen -> {reopened.params.quant}, contradicting arg -> {contested.params.quant}"

    # A grid that is too small has to be visible, not silently lossy.
    resetIndex(TightGlobal)
    var tight = openHnsw(hnswParams(TightGlobal, dim = Dim, quant = vqInt8Fixed,
                                    quantScale = grid / 10.0'f32))
    for v in indexVecs[0 ..< 20]:
      discard tight.add(v)
    echo &"\n=== too-small grid ({grid / 10.0'f32:.8f} instead of {grid:.8f}) ==="
    echo &"  clamped components after 20 vectors: {tight.clamped}"
    resetIndex(TightGlobal)

    # A stray argument must not reinterpret an existing index.
    let real = openHnsw(hnswParams(Global, quant = vqInt8, quantScale = grid))
    echo &"\n=== legacy guard ===\n  openHnsw({Global}, quant = vqInt8) -> " &
         &"{real.params.quant} (must stay vqNone)"

    resetIndex(ScratchGlobal)
    echo "\nscratch globals cleaned up"
