## Scratch test for the deletion path: synthetic vectors, no Python, scratch
## globals ^HNSWTEST* so the real index is untouched.
import std/[random, strformat, strutils]
import yottadb
import hnsw

const Global = "^HNSWTEST"
const Dim = 8
const N = 50

proc selfRetrievalOk(ix: HnswIndex, vecs: seq[seq[float32]]): bool =
  ## Every live node must still find itself as the nearest neighbour.
  for i in 0 ..< vecs.len:
    if not ix.hasId(i):
      continue
    let hits = ix.search(vecs[i], k = 1)
    if hits.len == 0 or hits[0].id != i or 1.0'f32 - hits[0].dist < 0.99:
      echo &"  MISS id={i} -> {hits}"
      return false
  true

# Raw blob access to the link lists, so the test can strand a node by hand the
# way an un-repaired delete would leave it.
proc readLinks(ix: HnswIndex, id, layer: int): seq[int] =
  let s = ydb_get(ix.params.globalNode, @[$id, "links", $layer])
  let n = s.len div sizeof(int64)
  result = newSeq[int](n)
  if n > 0:
    let src = cast[ptr UncheckedArray[int64]](s[0].unsafeAddr)
    for i in 0 ..< n:
      result[i] = int(src[i])

proc writeLinks(ix: HnswIndex, id, layer: int, links: seq[int]) =
  if links.len == 0:
    ydb_delete(ix.params.globalNode, @[$id, "links", $layer], YDB_DEL_NODE)
  else:
    var blob = newString(links.len * sizeof(int64))
    let dst = cast[ptr UncheckedArray[int64]](blob[0].addr)
    for i, x in links:
      dst[i] = int64(x)
    ydb_set(ix.params.globalNode, @[$id, "links", $layer], blob)

proc levelOfRaw(ix: HnswIndex, id: int): int =
  parseInt(ydb_get(ix.params.globalNode, @[$id, "level"]))

when isMainModule:
  for g in [Global.nodeGlobal, Global.keyGlobal, Global.metaGlobal]:
    ydb_delete(g, @[], YDB_DEL_TREE)

  var rng = initRand(42)
  var vecs: seq[seq[float32]]
  for i in 0 ..< N:
    var v = newSeq[float32](Dim)
    for d in 0 ..< Dim:
      v[d] = rand(rng, 2.0'f32) - 1.0'f32
    vecs.add v

  var ix = openHnsw(Global, M = 8, efConstruction = 32, efSearch = 64)
  for i in 0 ..< N:
    discard ix.add(vecs[i], key = "k" & $i)

  echo &"built      : count={ix.count} live={ix.liveCount} entry={ix.entry} " &
       &"maxLevel={ix.maxLevel} nodeCount={nodeCount(Global)}"
  echo "self-retrieval before deletes : ", selfRetrievalOk(ix, vecs)

  # --- repair: strand a node by hand, then let repair re-link it -------------
  echo "repair() on a healthy index (expect 0) : ", ix.repair()
  let victim = 21
  for l in 0 .. levelOfRaw(ix, victim):
    writeLinks(ix, victim, l, @[])
  echo &"stripped node {victim}: L0 links now {readLinks(ix, victim, 0).len}"
  echo "self-retrieval while stranded  : ", selfRetrievalOk(ix, vecs)
  echo "repair() re-linked layers      : ", ix.repair()
  echo &"node {victim} L0 links after repair: ", readLinks(ix, victim, 0).len
  echo "self-retrieval after repair    : ", selfRetrievalOk(ix, vecs)

  # --- delete one middle node -------------------------------------------------
  echo "delete(10) : ", ix.delete(10), "| hasId=", ix.hasId(10),
       "| lookup(k10)=", ix.lookup("k10"), "| live=", ix.liveCount,
       "| count=", ix.count, "| nodeCount=", nodeCount(Global)
  var leaked = false
  for h in ix.search(vecs[10], k = 5):
    if h.id == 10:
      leaked = true
  echo "deleted node still returned by search : ", leaked
  echo "delete(10) again (idempotent)          : ", ix.delete(10)

  # --- delete every other node, then check the graph is still navigable -------
  for i in countup(0, N - 1, 2):
    discard ix.delete(i)
  echo &"after evens: live={ix.liveCount} count={ix.count} entry={ix.entry} " &
       &"maxLevel={ix.maxLevel}"
  echo "self-retrieval after deletes  : ", selfRetrievalOk(ix, vecs)

  # --- delete the entry point: a new one must be elected ----------------------
  let oldEntry = ix.entry
  discard ix.delete(oldEntry)
  echo &"deleted entry {oldEntry} -> new entry={ix.entry} (live node: {ix.hasId(ix.entry)})"
  echo "self-retrieval after entry delete : ", selfRetrievalOk(ix, vecs)
  echo "repair() after the deletes (expect 0) : ", ix.repair()

  # --- stale bookkeeping: point META at a deleted node and reopen ------------
  # This is the "search suddenly returns nothing" failure a crashed or
  # interrupted delete can leave behind.
  ydb_set(Global.metaGlobal, @["entry"], "2")        # 2 was deleted above
  var ix2 = openHnsw(Global, M = 8, efConstruction = 32, efSearch = 64)
  echo &"reopened with entry={ix2.entry} (deleted): search -> ",
       ix2.search(vecs[1], k = 1).len, " hits"
  echo "repair() : ", ix2.repair(), " layer(s) relinked -> entry=", ix2.entry
  echo &"search after repair           : ", ix2.search(vecs[1], k = 1).len, " hits"
  echo "self-retrieval after repair   : ", selfRetrievalOk(ix2, vecs)

  # --- re-add an article that was deleted ------------------------------------
  let newId = ix.add(vecs[1], key = "k1")
  echo &"re-added k1 -> new id {newId} (was 1), lookup(k1)=", ix.lookup("k1")

  # --- delete everything -----------------------------------------------------
  var remaining: seq[int]
  for i in 0 ..< ix.count:
    if ix.hasId(i):
      remaining.add i
  for i in remaining:
    discard ix.delete(i)
  echo &"emptied    : live={ix.liveCount} entry={ix.entry} maxLevel={ix.maxLevel} " &
       &"nodeCount={nodeCount(Global)} count={ix.count}"
  echo "search on empty index : ", ix.search(vecs[1], k = 3).len, " hits"

  # --- delete-time repair: neighbour whose *only* link is the deleted node ----
  # (a fresh index, so this does not depend on how the graph above happened to
  # come out; nodes 0 and 1 are wired to point at each other only)
  for g in [Global.nodeGlobal, Global.keyGlobal, Global.metaGlobal]:
    ydb_delete(g, @[], YDB_DEL_TREE)
  var ix3 = openHnsw(Global, M = 8, efConstruction = 32, efSearch = 64)
  for i in 0 ..< 6:
    discard ix3.add(vecs[i], key = "n" & $i)
  writeLinks(ix3, 0, 0, @[1])          # node 0's only link is node 1
  writeLinks(ix3, 1, 0, @[0])          # node 1's only link is node 0
  echo &"before delete: node 0 L0 links = {readLinks(ix3, 0, 0)}"
  discard ix3.delete(1)
  echo &"after deleting node 1: node 0 L0 links = {readLinks(ix3, 0, 0)} " &
       &"(re-linked by delete itself)"

  for g in [Global.nodeGlobal, Global.keyGlobal, Global.metaGlobal]:
    ydb_delete(g, @[], YDB_DEL_TREE)
  echo "scratch globals cleaned up"
