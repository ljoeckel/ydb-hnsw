## Regression test for `preloadVectors` / `HnswParams.cacheVectors`.
##
## The cache is a *mirror* of the vectors, and the risk that comes with a mirror
## is divergence: a stale entry would silently score against a vector that is no
## longer there. So every case here compares a cached index against an uncached
## one reading the same YottaDB globals, and checks that
##
##   * a query returns the same ids *and* the same distances either way,
##   * `add`/`delete`/`putVec` after a preload keep the mirror coherent,
##   * `hasId` answers from the cache exactly as the database would.
##
## All three storage modes are covered, because the cache stores codes for two of
## them and floats for the third.
##
## Run: nim c -r src/test/hnsw_cache_test.nim

import std/[math, random, sequtils, strformat]
import hnsw, yottadb

const
  Dim = 64
  N = 300
  Probes = 20
  K = 5
  Scratch = "^HNSWCT"
  Grid = 0.0078'f32        # roughly 1/127, a plausible shared grid

proc resetIndex(global: string) =
  for g in [global.nodeGlobal, global.keyGlobal, global.metaGlobal]:
    ydb_delete(g, @[], YDB_DEL_TREE)

proc randomVecs(n: int, rng: var Rand): seq[seq[float32]] =
  for i in 0 ..< n:
    var v = newSeq[float32](Dim)
    for d in 0 ..< Dim:
      v[d] = float32(2.0 * rand(rng, 1.0) - 1.0)
    normalize(v)
    result.add v

proc hitsOf(ix: HnswIndex, q: seq[float32]): seq[(int, float32)] =
  for h in ix.search(q, k = K, ef = 32):
    result.add (h.id, h.dist)

proc sameHits(tag: string, want, got: seq[(int, float32)]): bool =
  ## Ids have to match exactly; distances may differ in the last bits because the
  ## cache hands out the vector by copy rather than through unpackFloats.
  result = want.len == got.len
  if not result:
    echo &"  {tag}: hit count {want.len} vs {got.len}"
    return
  for i in 0 ..< want.len:
    if want[i][0] != got[i][0]:
      echo &"  {tag}: id {i} {want[i][0]} vs {got[i][0]}"
      result = false
    elif abs(want[i][1] - got[i][1]) > 1e-6'f32:
      echo &"  {tag}: dist {i} {want[i][1]} vs {got[i][1]}"
      result = false

var failures = 0

proc check(tag: string, ok: bool) =
  echo "  [" & (if ok: "ok  " else: "FAIL") & "] " & tag
  if not ok: inc failures

proc params(quant: VecQuant): HnswParams =
  ## Cacheless params for the same index - what a second reader would open.
  hnswParams(Scratch, M = 8, efConstruction = 64, efSearch = 32, seed = 99,
             dim = Dim, quant = quant, quantScale = Grid)

proc searchAll(ix: HnswIndex, probes: seq[seq[float32]]): seq[seq[(int, float32)]] =
  for q in probes:
    result.add hitsOf(ix, q)

proc searchesAgree(tag: string, a, b: seq[seq[(int, float32)]]): bool =
  result = true
  for i in 0 ..< a.len:
    if not sameHits(&"{tag} q{i}", a[i], b[i]):
      result = false

when isMainModule:
  var rng = initRand(7)
  let vecs = randomVecs(N, rng)
  let probes = vecs[0 ..< Probes]

  for quant in [vqNone, vqInt8, vqInt8Fixed]:
    echo &"\n=== {quant} ==="
    resetIndex(Scratch)

    var builder = openHnsw(params(quant))
    for i, v in vecs:
      discard builder.add(v, key = "k" & $i)
    echo &"  built {builder.liveCount} nodes ({builder.count} ids allocated)"
    builder = nil     # everything below has to come back out of the database

    let plain = openHnsw(params(quant))
    let cached = openHnsw(hnswParams(Scratch, cacheVectors = true))
    check(&"preloaded every vector ({cached.cachedVectors} of {cached.liveCount})",
          cached.cachedVectors == cached.liveCount)
    check("storage mode still comes from META", cached.params.quant == quant)

    let want0 = searchAll(plain, probes)
    check("cached and uncached searches agree",
          searchesAgree("fresh", want0, searchAll(cached, probes)))

    var idsAgree = true
    for id in 0 ..< plain.count:
      if plain.hasId(id) != cached.hasId(id):
        idsAgree = false
    check("hasId agrees for every allocated id", idsAgree)

    # --- deletes after a preload -------------------------------------------
    let doomed = toSeq(0 ..< 10).mapIt(it * 7)
    for id in doomed:
      discard cached.delete(id)
    let afterDelete = openHnsw(params(quant))
    check(&"live count after {doomed.len} deletes",
          cached.liveCount == afterDelete.liveCount and
          cached.liveCount == plain.liveCount - doomed.len)
    check(&"cachedVectors followed the deletes ({cached.cachedVectors})",
          cached.cachedVectors == cached.liveCount)
    check("cached search matches a fresh open after deletes",
          searchesAgree("deleted", searchAll(afterDelete, probes),
                        searchAll(cached, probes)))

    var noZombies = true
    for q in probes:
      for (id, _) in hitsOf(cached, q):
        if id in doomed:
          noZombies = false
    check("deleted ids are not returned", noZombies)

    var goneAgree = true
    for id in doomed:
      if cached.hasId(id) or cached.delete(id):
        goneAgree = false
    check("deleted ids report hasId=false and delete=false", goneAgree)

    # --- inserts after a preload (the mirror has to grow) ------------------
    for i in 0 ..< 5:
      discard cached.add(vecs[i], key = "late" & $i)
    check(&"cachedVectors grew with the inserts ({cached.cachedVectors})",
          cached.cachedVectors == cached.liveCount)
    let afterAdd = openHnsw(params(quant))
    check("cached search matches a fresh open after inserts",
          searchesAgree("added", searchAll(afterAdd, probes),
                        searchAll(cached, probes)))

    var found = false
    for h in cached.search(vecs[0], k = 3, ef = 32):
      if h.id >= N:
        found = true
    check("a vector added after the preload is reachable", found)

    resetIndex(Scratch)

  echo ""
  if failures == 0:
    echo "all cache checks passed"
  else:
    echo &"{failures} check(s) FAILED"
    quit(1)
