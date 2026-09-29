## Brute-force recall of an HNSW index: of the *true* k nearest neighbours of a
## query, how many does the graph return?
##
## `hnsw_recall_check` asks whether a narrow candidate window finds what a full
## walk of the *same graph* finds. This program asks the other question: does the
## graph find the right neighbours of the *index's own collection*? Ground truth
## is an exhaustive scan of the vector store - the index's own `^...NODE` unless
## `--dataset` names another one - so a link lost or a node stranded by a delete
## shows up as recall below 1.
##
## For one probe q, with q's own node excluded on both sides (leave-one-out):
##
##   exact(q)  the k most similar vectors in the store. That global is the whole
##             population `search` can return from, so nothing has to be filtered
##             out of it: a node in the store is a node the graph holds.
##   got(q)    `ix.search(q, k + 1, ef)` with q's own hit dropped and the rest
##             cut to k. The `k + 1` because a self-query would otherwise spend
##             one of its k slots on itself, which alone would cap recall at
##             (k - 1) / k and look like a graph problem.
##   recall@k  |exact ∩ got| / min(k, |exact|), averaged over the probes. This is
##             the id-overlap recall of the ANN literature, and on a collection
##             with duplicate articles it *under*-reports: two nodes carrying the
##             same vector are the same neighbour as far as a distance is
##             concerned, but only the exact id counts here.
##   +ties     the same count with duplicates forgiven: a returned node counts as
##             found when its similarity to a stored vector reaches the k-th exact
##             similarity. On a clean collection the two columns agree; where they
##             diverge, the gap is duplicate vectors, not a graph fault.
##
## Probes are stored vectors read back the way the scan reads them - for a
## quantized index that is the stored codes dequantized, i.e. the vectors the
## graph itself compares. Both sides of every distance come from the store, which
## is what keeps this a measurement of the *graph*: quantization error is not part
## of these numbers, and an index whose codes have lost the original distances
## still scores 1.0 here. To see that loss, measure against the original float32
## vectors from outside the index.
##
## Reported alongside recall, because they fail for different reasons:
##
##   top-1 match  how many probes' returned nearest *other* vector is (as near
##                as) the exact nearest. (Other, because the exact nearest of a
##                probe is the probe itself.) Losing the top spot while recall
##                stays high points at the window, or at a duplicate taking the
##                spot; losing both is the graph.
##   self rank 1  whether the probe's own node comes back first. It is in the
##                index and is its own nearest neighbour, so a healthy graph
##                returns it at similarity ~1.0. Recall can look fine while this
##                does not - stranded or unreachable nodes - and then `repair` is
##                the tool, not a wider `ef`. A duplicate of the probe sitting at
##                similarity 1.0 also takes this spot, which is not a fault; the
##                `+ties` column is the one that forgives that.
##
## Below the table, the exact ids the widest `ef` did not return are split by what
## can be done about them: those a returned neighbour duplicates at the same or a
## closer distance (no similarity lost - the same thing `+ties` forgives), those
## unreachable from their own vector (a stranded node - `repair`), and those
## reachable but not reached (a wider `ef`).
##
## One thing this cannot see:
##
##   * Ties at the boundary. `+ties` forgives a neighbour a returned vector
##     matches or beats in distance; it cannot forgive the k-th slot being filled
##     by a vector a hair *further* than the exact k-th neighbour, which is what a
##     large duplicate group produces. A few such slots in 500 move recall by less
##     than a percent, but they never quite go away.
##
## Run: nim c -r -d:release -p:src src/test/hnsw_recall_bruteforce.nim [options]
##   --index=GLOBAL    index to measure (default ^HNSWArticles)
##   --dataset=GLOBAL  vector store to scan: one shape, `^GLOBAL(id)` is the
##                     vector and its subscript the node id (default: the index's
##                     own store, its global name with `NODE` appended)
##   --probes=N        stored vectors used as queries (default 100)
##   --k=N             neighbours compared per probe (default 10)
##   --ef=A,B,...      candidate widths to report  (default 16,32,64,128,256)
##   --seed=N          probe sampling seed, for reproducible runs (default 1234)
##   --cache           mirror the index's vectors into memory before searching
##   --verbose         one line per probe, at the widest ef
##   --help            this text
##
## One run reads every stored vector once (plus one per probe), so it costs about
## a `preloadVectors` plus `probes * live` dot products - seconds to a couple of
## minutes on the article index, not a benchmark loop.
##
## No Python needed: this program never embeds, it only reads and searches.

import std/[algorithm, os, random, sequtils, sets, strformat, strutils, tables, times]

import yottadb
import ../hnsw

when defined(hnswSimd):
  # Same kernel family as `hnsw`; see the note there on why the parallelism is
  # written down instead of left to the compiler.
  import nimsimd/avx

const
  DefaultIndex = "^HNSWArticles"
  DefaultProbes = 100
  DefaultK = 10
  DefaultEf = "16,32,64,128,256"
  ProgressEvery = 50_000

  Usage = """
usage: hnsw_recall_bruteforce [option=value]...

  --index=GLOBAL    index to measure (default ^HNSWArticles)
  --dataset=GLOBAL  vector store to scan for ground truth. One shape only:
                    `^GLOBAL(id)` is the vector, its subscript the node id.
                    Default: the index's own store (its name + "NODE")
  --probes=N        stored vectors used as queries (default 100)
  --k=N             neighbours to compare per probe (default 10)
  --ef=A,B,...      candidate widths to report (default 16,32,64,128,256)
  --seed=N          probe sampling seed (default 1234)
  --cache           mirror the index's vectors into memory before searching
  --verbose         one line per probe, at the widest ef
  --help            this text

The store is scanned exhaustively, so a run reads every vector in it. Every id
it holds is a node `search` may return, by subscript, but only when the store is
the one this graph was built in - or an id-aligned copy of it.
"""

type
  Options = object
    index, dataset: string
    probes, k, seed: int
    efs: seq[int]
    cache, verbose, help: bool

  TopK = object
    ## The best `k` of a stream, ascending by similarity. `sims[0]` is the
    ## weakest entry kept, i.e. the one a new candidate has to beat.
    k: int
    sims: seq[float32]
    ids: seq[int]

# ---------------------------------------------------------------------------
# argument parsing
# ---------------------------------------------------------------------------

proc parseArgs(): Options =
  result.index = DefaultIndex
  result.dataset = ""          # empty: the index's own store, resolved on open
  result.probes = DefaultProbes
  result.k = DefaultK
  result.seed = 1234
  for part in DefaultEf.split(','):
    result.efs.add parseInt(part)

  for i in 1 .. paramCount():
    let a = paramStr(i)
    let eq = a.find('=')
    let key = if eq < 0: a else: a[0 ..< eq]
    let val = if eq < 0: "" else: a[eq + 1 .. ^1]
    case key
    of "--index":
      if val.len == 0: quit "--index needs a global name", 2
      result.index = val
    of "--dataset": result.dataset = val
    of "--probes": result.probes = parseInt(val)
    of "--k": result.k = parseInt(val)
    of "--seed": result.seed = parseInt(val)
    of "--ef":
      if val.len == 0: quit "--ef needs at least one width", 2
      result.efs.setLen(0)
      for part in val.split(','):
        result.efs.add parseInt(part)
    of "--cache": result.cache = true
    of "--verbose", "-v": result.verbose = true
    of "--help", "-h": result.help = true
    else: quit &"unknown option {a} - try --help", 2

  if result.probes <= 0: quit "--probes must be positive", 2
  if result.k <= 0: quit "--k must be positive", 2

  result.efs.sort()
  var deduped: seq[int]
  for ef in result.efs:
    if ef <= 0: quit "--ef values must be positive", 2
    if deduped.len == 0 or deduped[^1] != ef:
      deduped.add ef
  result.efs = deduped

# ---------------------------------------------------------------------------
# vectors
# ---------------------------------------------------------------------------

proc unpackFloats(s: string): seq[float32] =
  let n = s.len div sizeof(float32)
  result = newSeq[float32](n)
  if n > 0:
    copyMem(result[0].addr, s[0].unsafeAddr, n * sizeof(float32))

proc dotF32(a, b: seq[float32]): float32 =
  ## Dot product - which is the cosine similarity, the vectors being L2
  ## normalized. Mirrors `hnsw`'s kernel, scalar fallback included.
  doAssert a.len == b.len, &"dot: {a.len} vs {b.len} dims"
  let n = a.len
  if n == 0: return 0.0'f32
  when defined(hnswSimd):
    var i = 0
    var acc0 = mm256_setzero_ps()
    var acc1 = mm256_setzero_ps()
    var acc2 = mm256_setzero_ps()
    var acc3 = mm256_setzero_ps()
    while i + 32 <= n:
      acc0 = mm256_add_ps(acc0, mm256_mul_ps(mm256_loadu_ps(a[i].unsafeAddr),
                                             mm256_loadu_ps(b[i].unsafeAddr)))
      acc1 = mm256_add_ps(acc1, mm256_mul_ps(mm256_loadu_ps(a[i + 8].unsafeAddr),
                                             mm256_loadu_ps(b[i + 8].unsafeAddr)))
      acc2 = mm256_add_ps(acc2, mm256_mul_ps(mm256_loadu_ps(a[i + 16].unsafeAddr),
                                             mm256_loadu_ps(b[i + 16].unsafeAddr)))
      acc3 = mm256_add_ps(acc3, mm256_mul_ps(mm256_loadu_ps(a[i + 24].unsafeAddr),
                                             mm256_loadu_ps(b[i + 24].unsafeAddr)))
      i += 32
    while i + 8 <= n:
      acc0 = mm256_add_ps(acc0, mm256_mul_ps(mm256_loadu_ps(a[i].unsafeAddr),
                                             mm256_loadu_ps(b[i].unsafeAddr)))
      i += 8
    let acc = mm256_add_ps(mm256_add_ps(acc0, acc1), mm256_add_ps(acc2, acc3))
    var s = mm_add_ps(mm256_castps256_ps128(acc), mm256_extractf128_ps(acc, 1))
    s = mm_hadd_ps(s, s)
    s = mm_hadd_ps(s, s)
    var buf: array[4, float32]
    mm_storeu_ps(buf[0].addr, s)
    result = buf[0]
    while i < n:
      result += a[i] * b[i]
      inc i
  else:
    for i in 0 ..< n:
      result += a[i] * b[i]

proc offer(t: var TopK, sim: float32, id: int) =
  ## One candidate for one probe. The kept list stays ascending, so the worst of
  ## the k is at index 0 and is one compare away.
  if t.k <= 0: return
  if t.sims.len < t.k:
    t.sims.add sim
    t.ids.add id
    var i = t.sims.len - 1
    while i > 0 and t.sims[i - 1] > t.sims[i]:
      swap(t.sims[i - 1], t.sims[i])   # index access is an lvalue, `swap` is fine
      swap(t.ids[i - 1], t.ids[i])
      dec i
  elif sim > t.sims[0]:
    t.sims[0] = sim
    t.ids[0] = id
    var i = 0
    while i + 1 < t.sims.len and t.sims[i] > t.sims[i + 1]:
      swap(t.sims[i], t.sims[i + 1])
      swap(t.ids[i], t.ids[i + 1])
      inc i

proc storedVec(ix: HnswIndex, store: string, id: int): seq[float32] =
  ## `id`'s vector as the store holds it, materialized as float32 - the same read
  ## and the same dequantization `hnsw.loadVec` does. Empty when there is no such
  ## node (the empty string is 0 dimensions, not a zero vector) or when what is
  ## there does not decode to the index's dimension.
  let s = ydb_get(store, @[$id])
  if s.len == 0: return
  case ix.params.quant
  of vqNone: result = unpackFloats(s)
  of vqInt8:
    if s.len <= sizeof(float32): return
    var scale: float32
    copyMem(scale.addr, s[0].unsafeAddr, sizeof(float32))
    result = dequantizeInt8(unpackInt8(s[sizeof(float32) .. ^1]), scale)
  of vqInt8Fixed:
    result = dequantizeInt8(unpackInt8(s), ix.params.quantScale)
  if result.len != ix.dim:
    result.setLen(0)              # not a vector of this index - not a candidate

proc loadedVec(ix: HnswIndex, store: string, id: int): seq[float32] =
  ## `storedVec`, L2 normalized - the one form the scan, the probes and the
  ## thresholds all compare in, so both sides of a distance are the same kind of
  ## vector.
  result = storedVec(ix, store, id)
  normalize(result)

proc keyOf(ix: HnswIndex, id: int): string =
  ## The key the index filed `id` under, for reading a report line - the idxref of
  ## the article, on the article index. One small read, and only the verbose
  ## report asks for it.
  ydb_get(ix.params.globalKey, @[$id])

const SimNone = -2.0'f32
  ## Sentinel for "no similarity available": an id without a usable vector. Never
  ## a legal similarity, which is in [-1, 1].

const SimEps = 1e-5'f32
  ## Slack when comparing two similarities. Both sides are computed by `dotF32` on
  ## the same stored vector, so they agree to the bit; the slack is only there so
  ## a rebuild with different kernels cannot turn a tie into a miss.

proc simTo(ix: HnswIndex, store: string, id: int, q: seq[float32]): float32 =
  ## Similarity of `q` to `id`'s stored vector, computed the way the ground-truth
  ## scan computed its distances, so a threshold taken from the exact set can be
  ## compared against it. `SimNone` when the id has no usable vector.
  let v = loadedVec(ix, store, id)
  if v.len != q.len:
    return SimNone
  dotF32(q, v)

# ---------------------------------------------------------------------------
# reading the vector store
# ---------------------------------------------------------------------------

proc storeIds(store: string): tuple[ids: seq[int], nodes, ignored: int] =
  ## Every id the vector store holds, oldest first. The store has exactly one
  ## shape - `^store(id)` is the vector and its subscript the id - so a node is
  ## taken when it has a single subscript that parses as one. Anything else is
  ## counted into `ignored` and skipped rather than guessed at; the store is not
  ## supposed to have another shape, so a count there is worth printing.
  ##
  ## No value is read here: whether a node really carries a vector of this index's
  ## dimension is decided by the scan (and by `search`), which reads every one of
  ## them anyway.
  var subs: Subscripts = @[]
  var rc: int
  (rc, subs) = ydb_node_next(store, subs)
  while rc == YDB_OK:
    inc result.nodes
    var id = -1
    if subs.len == 1 and subs[0].allCharsInSet({'0' .. '9'}):
      try:
        id = parseInt(subs[0])
      except ValueError:
        id = -1                   # absurdly long - not an id this index has
    if id < 0:
      inc result.ignored
    else:
      result.ids.add id
    (rc, subs) = ydb_node_next(store, subs)
  result.ids.sort()

proc sampleIndices(n, count, seed: int): seq[int] =
  ## `n` distinct indices in `0 ..< count`, drawn deterministically from `seed`
  ## and returned sorted. Random rather than strided because ids and keys are not
  ## order-free: a stride can land on one feed's block of the collection.
  if n >= count:
    for i in 0 ..< count: result.add i
    return
  var rng = initRand(seed)
  var picked = initHashSet[int]()
  while picked.len < n:
    picked.incl rand(rng, count - 1)
  for i in picked:
    result.add i
  result.sort()

# ---------------------------------------------------------------------------
# ground truth and measurement
# ---------------------------------------------------------------------------

proc scanGroundTruth(ix: HnswIndex, store: string, ids: seq[int],
                     probeIds: seq[int], probeVecs: seq[seq[float32]], k: int
                    ): tuple[tops: seq[TopK], noVec: int] =
  ## One pass over every stored vector, offering it to every probe, so each
  ## probe's heap ends up holding its exact k nearest. One pass and not one pass
  ## per probe, because the YottaDB read - not the dot products - is what the scan
  ## costs.
  result.tops = newSeq[TopK](probeIds.len)
  for t in result.tops.mitems:
    t.k = k
  let started = epochTime()
  for n, id in ids:
    var v = loadedVec(ix, store, id)
    if v.len == 0:
      inc result.noVec             # not a vector of this index - search skips it too
      continue
    for pi, pid in probeIds:
      if pid == id:
        continue                  # leave-one-out: the probe is not its own neighbour
      result.tops[pi].offer(dotF32(probeVecs[pi], v), id)
    if (n + 1) mod ProgressEvery == 0:
      echo &"  {n + 1}/{ids.len} scanned, {epochTime() - started:.0f} s"

proc evaluate(ix: HnswIndex, store: string, probeIds: seq[int],
              probeVecs: seq[seq[float32]], tops: seq[TopK], k: int,
              efs: seq[int], verbose: bool) =
  ## Search every probe at every ef and compare with the exact neighbourhoods.
  var exact = newSeq[HashSet[int]](tops.len)
  for i, t in tops:
    exact[i] = initHashSet[int]()
    for id in t.ids:
      exact[i].incl id

  # The exact similarity of a returned node, computed from the store the way the
  # ground truth was and not from `Hit.sim()`, which on a quantized index is a
  # distance between codes: the two are not comparable. Cached because the same
  # nodes come back at every ef.
  var simCache = initTable[(int, int), float32]()

  echo &"     ef   recall@{k}   +ties   top-1    self#1"
  var missedAll, coveredMissed, strandedAll, notReachedAll, unreadableAll, slotTotal = 0
  for fi, ef in efs:
    var recall = 0.0
    var tolerant = 0.0
    var counted = 0
    var selfFirst, top1Agree = 0
    var lines: seq[string]
    for pi in 0 ..< probeIds.len:
      let probeId = probeIds[pi]
      let denom = min(k, tops[pi].ids.len)
      if denom == 0:
        continue
      inc counted
      # k + 1: the probe itself is in the graph and will come back first.
      let hits = ix.search(probeVecs[pi], k = k + 1, ef = ef)
      # The weakest exact neighbour's similarity is the bar a returned node has
      # to reach to count in the tie-tolerant column; the strongest one is what
      # the returned nearest is compared with.
      let bar = if tops[pi].sims.len > 0: tops[pi].sims[0] else: SimNone
      let best = if tops[pi].sims.len > 0: tops[pi].sims[^1] else: SimNone
      let selfAtRank1 = hits.len > 0 and hits[0].id == probeId
      var shared, taken, tied = 0
      var firstOther = -1            # nearest returned that is not the probe
      var firstOtherSim = SimNone
      var got = initHashSet[int]()
      var gotSims: seq[float32]
      for h in hits:
        if h.id == probeId:
          continue
        var sim = SimNone
        let ck = (pi, h.id)
        if simCache.hasKey(ck):
          sim = simCache[ck]
        else:
          sim = simTo(ix, store, h.id, probeVecs[pi])
          simCache[ck] = sim
        if firstOther < 0:
          firstOther = h.id
          firstOtherSim = sim
        if taken < denom:
          got.incl h.id
          gotSims.add sim
          if h.id in exact[pi]:
            inc shared
          if sim >= bar - SimEps:
            inc tied
          inc taken
        if taken >= denom and firstOther >= 0:
          break
      recall += shared.float / denom.float
      tolerant += tied.float / denom.float
      if selfAtRank1:
        inc selfFirst
      if firstOtherSim >= best - SimEps:
        inc top1Agree
      if verbose and fi == efs.high:
        lines.add &"  {probeId:>7} {keyOf(ix, probeId):<18} " &
                  &"exact {tops[pi].ids[^1]:>7} ({best:.3f})  " &
                  &"got {firstOther:>7} ({firstOtherSim:.3f})  " &
                  &"{shared:>2}/{denom} ({max(tied - shared, 0)} tie)  kth {bar:.3f}" &
                  (if best >= 0.999'f32: "  duplicate" else: "")
      # Only for the headline ef, and only for the missing exact ids, splitting
      # them the way the user has to act on them:
      #
      #   covered      the returned list already holds as many vectors at least as
      #                close as this one, so nothing a distance can measure was
      #                lost - duplicates, counted from the distance side by `+ties`.
      #   stranded     not covered, and not found by its own vector either: the
      #                node has no usable links and no window will find it.
      #   not reached  not covered, but reachable - what a larger `ef` buys.
      if fi == efs.high:
        slotTotal += denom
        for id in exact[pi]:
          if id in got:
            continue
          inc missedAll
          let mv = loadedVec(ix, store, id)
          if mv.len == 0:
            inc unreadableAll
            continue
          let ms = dotF32(probeVecs[pi], mv)
          # Rank-aware, not just "something as close exists": count how many of
          # the exact neighbours sit at least as close as this one, and how many
          # of the returned do. Enough returned ones and this id would not have
          # changed the answer.
          var exactAtLeast, gotAtLeast = 0
          for s in tops[pi].sims:
            if s >= ms - SimEps:
              inc exactAtLeast
          for s in gotSims:
            if s != SimNone and s >= ms - SimEps:
              inc gotAtLeast
          if gotAtLeast >= exactAtLeast:
            inc coveredMissed
            continue
          var reachable = false
          for h in ix.search(mv, k = ef, ef = ef):
            if h.id == id:
              reachable = true
              break
          if reachable:
            inc notReachedAll
          else:
            inc strandedAll
    if counted > 0:
      echo &"{ef:>7}   {recall / counted.float:.3f}   " &
           &"{tolerant / counted.float:.3f}   " &
           &"{top1Agree:>2}/{counted}   {selfFirst:>3}/{counted}"
    for l in lines:
      echo l
  if missedAll > 0:
    echo &"exact ids not returned at ef = {efs[^1]}: {missedAll} of {slotTotal} probe slots"
    if coveredMissed > 0:
      echo &"  {coveredMissed} have a returned vector at the same or a closer distance " &
           &"- duplicates, no similarity lost"
    let cost = missedAll - coveredMissed - unreadableAll
    if cost > 0:
      echo &"  {cost} have no such stand-in:"
      if strandedAll > 0:
        echo &"    {strandedAll} not found by their own vector either - unreachable " &
             &"nodes, not a narrow window: run repair"
      if notReachedAll > 0:
        echo &"    {notReachedAll} reachable but not reached - a larger ef may help"
    if unreadableAll > 0:
      echo &"  {unreadableAll} had no readable stored vector to check"

# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

when isMainModule:
  let opts = parseArgs()
  if opts.help:
    echo Usage
    quit 0

  let openStart = epochTime()
  var ix = openHnsw(hnswParams(opts.index, cacheVectors = opts.cache))
  if opts.cache:
    echo &"preloaded {ix.cachedVectors} vectors in {epochTime() - openStart:.1f} s"
  echo &"{opts.index}: live={ix.liveCount} count={ix.count} dim={ix.dim} " &
       &"quant={ix.params.quant} M={ix.params.M} efSearch={ix.params.efSearch}"
  if ix.liveCount == 0 or ix.dim == 0:
    quit "empty index - nothing to measure", 1
  if opts.k > ix.liveCount:
    quit &"k={opts.k} is larger than the {ix.liveCount} live vectors", 1

  # The vector store, one deep-first pass over `^store(id) = vector`. Its ids are
  # the graph's node ids, which is what makes a scan of it a ground truth here.
  let store = if opts.dataset.len > 0: opts.dataset else: ix.params.globalNode
  if store != ix.params.globalNode:
    echo &"store {store}, not this index's own {ix.params.globalNode}"
  let (ids, nodes, ignored) = storeIds(store)
  echo &"{store}: {nodes} node(s), {ids.len} id(s)"
  if ignored > 0:
    echo &"  {ignored} node(s) ignored: not `(id) = vector`"
  if ids.len == 0:
    quit &"no vectors in {store} - is the index built?", 1
  if ids.len != ix.liveCount:
    echo &"  note: the store holds {ids.len} ids, META says live={ix.liveCount}"

  # Probes: stored vectors, so the query is the same kind of vector the scan and
  # `search` both work with.
  var probeIds: seq[int]
  var probeVecs: seq[seq[float32]]
  var badProbes = 0
  for i in sampleIndices(opts.probes, ids.len, opts.seed):
    let v = loadedVec(ix, store, ids[i])
    if v.len == 0:
      inc badProbes
      continue
    probeIds.add ids[i]
    probeVecs.add v
  if probeIds.len == 0:
    quit &"no probe vector of {ix.dim} dimensions in {store}", 1
  if badProbes > 0:
    echo &"  {badProbes} probe(s) dropped: no usable vector"

  let efs = opts.efs.filterIt(it >= opts.k)
  if efs.len == 0:
    quit &"--ef has no width >= k={opts.k}", 2
  if efs.len < opts.efs.len:
    echo &"  ef below k={opts.k} left out: search widens a window to k anyway"

  echo &"{probeIds.len} probes, k={opts.k}, exhaustive ground truth over {ids.len} stored vectors"
  let (tops, noVec) = scanGroundTruth(ix, store, ids, probeIds, probeVecs, opts.k)
  if noVec > 0:
    echo &"  {noVec} id(s) without a usable vector - left out of the exact sets"
  echo ""
  echo &"recall@{opts.k} over {probeIds.len} probes " &
       &"(ground truth: exhaustive scan of {store}):"
  evaluate(ix, store, probeIds, probeVecs, tops, opts.k, efs, opts.verbose)
