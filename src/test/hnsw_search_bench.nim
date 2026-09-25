## End-to-end search benchmark against a real index in YottaDB.
##
## Times `search` (the whole graph walk, including every YottaDB read it does)
## so the gain from a faster distance kernel can be told apart from the time the
## process spends inside libyottadb. Queries are vectors that are already in the
## index, read back from YottaDB - no embedding needed.
##
## Run: nim c -r -d:release -p:src src/test/hnsw_search_bench.nim [global] [queries] [cache]

import std/[math, os, strformat, strutils, times]
import hnsw, yottadb

const
  DefaultGlobal = "^HNSWArticles"
  Reps = 3           # passes over the query set, best one reported

proc unpackFloats(s: string): seq[float32] =
  let n = s.len div sizeof(float32)
  result = newSeq[float32](n)
  if n > 0:
    copyMem(result[0].addr, s[0].unsafeAddr, n * sizeof(float32))

proc queryVec(ix: HnswIndex, id: int): seq[float32] =
  ## `id`'s stored vector as float32. For a quantized index this dequantizes,
  ## which is fine for a benchmark query - the point is a realistic vector, not
  ## bit-exactness with what `search` then re-quantizes.
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
  var nq = 200
  var useCache = false
  let args = commandLineParams()
  if args.len > 0: global = args[0]
  if args.len > 1: nq = parseInt(args[1])
  if args.len > 2: useCache = args[2] == "cache"

  # `openHnsw` is where a model would be loaded; the search itself never embeds,
  # so opening without a loader is enough.
  let preloadStart = epochTime()
  var ix = openHnsw(hnswParams(global, cacheVectors = useCache))
  if useCache:
    echo &"preloaded {ix.cachedVectors} vectors in {epochTime() - preloadStart:.1f} s"
  echo &"{global}: count={ix.count} live={ix.liveCount} dim={ix.dim} " &
       &"quant={ix.params.quant} M={ix.params.M} efSearch={ix.params.efSearch}"
  if ix.liveCount == 0:
    echo "empty index - nothing to measure"
    return

  # Live ids, and a real stored vector as the query for each.
  var vecs: seq[seq[float32]]
  var id = 0
  while id < ix.count and vecs.len < nq:
    if ix.hasId(id):
      let v = queryVec(ix, id)
      if v.len > 0:
        vecs.add v
    inc id
  let qn = vecs.len
  if qn == 0:
    echo "no live vectors found"
    return
  echo &"{qn} queries taken from the index, k=10, ef={ix.params.efSearch}"

  # Warm-up pass: also faults in whatever YottaDB has not cached yet.
  var hits = 0
  for i in 0 ..< qn:
    hits += ix.search(vecs[i], k = 10).len

  var best = Inf
  for _ in 0 ..< Reps:
    let t0 = epochTime()
    var found = 0
    for i in 0 ..< qn:
      found += ix.search(vecs[i], k = 10).len
    let dt = epochTime() - t0
    if dt < best:
      best = dt
    hits += found

  echo &"search {best * 1e6 / float64(qn):.1f} us/query  " &
       &"({qn} queries, best of {Reps}, {hits} hits total)"

main()
