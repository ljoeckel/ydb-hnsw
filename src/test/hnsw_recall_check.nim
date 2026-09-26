## Recall check for `search`: does the candidate window (`ef`) find the
## neighbours an exhaustive walk of the same graph finds?
##
## Ground truth is `search` itself with `ef = live count`. That does not stop at a
## candidate window - it walks the whole layer-0 graph best-first - so it returns
## the exact top-k *of the reachable component*. It is the standard way to measure
## HNSW recall without building a second, brute-force index, and it needs no
## embedding: the probes are vectors already in the index, read back from YottaDB
## the way `hnsw_search_bench` does it.
##
## Two separate things are reported, because they fail for different reasons:
##
##   recall@k per ef   the share of the exhaustive top-k a windowed search
##                     returns. This is what `efSearch` trades against latency.
##   self-hit rank     where the queried article's own node comes back. It is in
##                     the index, so a healthy graph returns it first with
##                     similarity ~1.0. If recall looks fine but rank 1 does not,
##                     the *graph* is damaged - deleted nodes that were never
##                     re-linked, stranded nodes - rather than the window being
##                     too narrow; `repair` is the tool for that. If both are bad,
##                     suspect `M` / `efConstruction` instead.
##
## A probe's own vector is read back from storage, so on a quantized index it is
## dequantized rather than bit-identical to the stored codes; `search`
## re-normalises and re-quantizes it, which is exactly what a real query does.
##
## Run: nim c -r -d:release -p:src src/test/hnsw_recall_check.nim [global] [probes] [k] [cache]
##   The exhaustive pass reads every reachable node per probe, so pass `cache`
##   (and keep `probes` small) unless the index fits in RAM comfortably.

import std/[os, strformat, strutils, sets, times]

import hnsw, yottadb

const
  DefaultGlobal = "^HNSWArticles"
  DefaultProbes = 25
  DefaultK = 10
  EfLadder = [16, 32, 64, 128, 256]

proc unpackFloats(s: string): seq[float32] =
  let n = s.len div sizeof(float32)
  result = newSeq[float32](n)
  if n > 0:
    copyMem(result[0].addr, s[0].unsafeAddr, n * sizeof(float32))

proc queryVec(ix: HnswIndex, id: int): seq[float32] =
  ## `id`'s stored vector as float32 - the same read `hnsw_search_bench` uses.
  ## Returns an empty seq for an id without a vector.
  let s = ydb_get(ix.params.globalNode, @[$id, "vec"])
  case ix.params.quant
  of vqNone:
    unpackFloats(s)
  of vqInt8:
    if s.len <= sizeof(float32):
      return
    var scale: float32
    copyMem(scale.addr, s[0].unsafeAddr, sizeof(float32))
    dequantizeInt8(unpackInt8(s[sizeof(float32) .. ^1]), scale)
  of vqInt8Fixed:
    dequantizeInt8(unpackInt8(s), ix.params.quantScale)

proc main() =
  var global = DefaultGlobal
  var nProbes = DefaultProbes
  var k = DefaultK
  var useCache = false
  let args = commandLineParams()
  if args.len > 0: global = args[0]
  if args.len > 1: nProbes = parseInt(args[1])
  if args.len > 2: k = parseInt(args[2])
  if args.len > 3: useCache = args[3] == "cache"

  let openStart = epochTime()
  var ix = openHnsw(hnswParams(global, cacheVectors = useCache))
  if useCache:
    echo &"preloaded {ix.cachedVectors} vectors in {epochTime() - openStart:.1f} s"
  echo &"{global}: live={ix.liveCount} count={ix.count} dim={ix.dim} " &
       &"quant={ix.params.quant} M={ix.params.M} efSearch={ix.params.efSearch} " &
       &"entry={ix.entry} maxLevel={ix.maxLevel}"
  if ix.liveCount == 0:
    quit "empty index - nothing to measure", 1
  let live = ix.liveCount

  # Probes: live ids and their stored vectors, lowest ids first.
  var sample: seq[(int, seq[float32])]
  var id = 0
  while id < ix.count and sample.len < nProbes:
    if ix.hasId(id):
      let v = queryVec(ix, id)
      if v.len > 0:
        sample.add (id, v)
    inc id
  if sample.len == 0:
    quit "no live vectors found", 1

  echo &"{sample.len} probes, k={k}, exhaustive ef={live}, ladder {EfLadder}"

  var recall = newSeq[float](EfLadder.len)
  var exhaustiveFirst = 0     # exhaustive top-1 is the queried node itself
  var windowFirst = 0         # ... and the widest windowed search agrees
  var windowFound = 0

  for (qid, vec) in sample:
    # Ground truth: no candidate window, the whole of layer 0.
    let t0 = epochTime()
    let exact = ix.search(vec, k = k, ef = live)
    var exactIds = initHashSet[int]()
    for h in exact:
      exactIds.incl h.id
    if exact.len > 0 and exact[0].id == qid:
      inc exhaustiveFirst

    var line = &"{qid:>7}  exact {exact.len:>2} in {epochTime() - t0:5.2f} s"
    for j, ef in EfLadder:
      let got = ix.search(vec, k = k, ef = ef)
      var shared = 0
      for h in got:
        if h.id in exactIds:
          inc shared
      let r = shared.float / k.float
      recall[j] += r
      line.add &"  ef{ef}:{r:4.2f}"

      if ef == EfLadder[^1]:
        inc windowFound
        if got.len > 0 and got[0].id == qid:
          inc windowFirst
    echo line

  let n = sample.len.float
  echo ""
  echo &"recall@{k} over {sample.len} probes (ground truth: search at ef={live}):"
  for j, ef in EfLadder:
    echo &"  ef = {ef:>3}: {recall[j] / n:.3f}"
  echo &"self-hit rank 1, exhaustive:   {exhaustiveFirst}/{sample.len}"
  echo &"self-hit rank 1, ef = {EfLadder[^1]:>3}:     {windowFirst}/{windowFound}"

when isMainModule:
  main()
