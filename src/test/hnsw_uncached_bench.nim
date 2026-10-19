## A/B harness for the *uncached* candidate decode - `distanceAt`'s YottaDB
## fallback, the path that used to call `loadVec` (`newSeqUninit` + `copyMem`,
## plus a whole-blob slice for `vqInt8`) and now reads the blob once and *views*
## it (`storedBlob` + `storedView`).
##
## The benchmark builds its own index in a scratch global, from a fixed seed, so
## two builds of the library measure the same graph and the same vectors. With
## `cacheVectors` off (the default here) **every** candidate takes the fallback,
## which is what makes this a direct measurement of the change rather than a
## measurement of how rarely it is hit. Query blobs are read before the timed
## loop: the query's own read is the same on both sides and would only add noise.
##
## How to A/B it:
##
##   git stash push -- src/hnsw.nim
##   nim c -d:release -o:/tmp/bench_old src/test/hnsw_uncached_bench.nim
##   git stash pop
##   nim c -d:release -o:/tmp/bench_new src/test/hnsw_uncached_bench.nim
##   for i in 1 2 3; do taskset -c 2 /tmp/bench_old; taskset -c 2 /tmp/bench_new; done
##
## Interleave the two binaries (this box drifts by tens of percent between
## minutes) and compare runs from the same session, never against a number
## recorded on another day. Pin the CPU: both binaries are single-threaded, but
## the machine has other work on it.
##
## Answers, not just times: `--dump` prints `q=<id>: <id>:<dist> ...` for every
## query in all three modes plus a reader whose vector cache was opened while the
## index was empty (so it walks through the fallback with a cache present). That
## output is byte-identical across the change - float32 distances printed by `$`
## round-trip exactly - so it can be diffed:
##
##   /tmp/bench_old --dump > /tmp/dump_old.txt
##   /tmp/bench_new --dump > /tmp/dump_new.txt
##   diff /tmp/dump_old.txt /tmp/dump_new.txt && echo identical
##
## Measured with this program on the 2026-10 change (`--reps=2`, otherwise the
## defaults: dim 384, 5k nodes, 1000 queries, k=10, ef=64, cache off, pinned to
## core 2, two interleaved passes of old/new binaries): vqNone 1936.9/1925.7 ->
## 1881.1/1867.1 us/query (~1.03x), vqInt8 2211.8/2182.0 -> 1484.5/1484.1
## us/query (~1.48x). The int8 win is the larger one because `unpackQVec`
## decoded through `s[4..^1]`, a second whole-blob copy on top of the seq. The
## printed checksums (sum of every returned distance) were identical between the
## two builds, and `--dim=64 --nodes=300 --dump` produced 900 identical lines.
##
## Run:
##   nim c -r -d:release src/test/hnsw_uncached_bench.nim
##   nim c -r -d:release src/test/hnsw_uncached_bench.nim --nodes=1000 --dim=128
##   nim c -r -d:release src/test/hnsw_uncached_bench.nim --dump
##   nim c -r -d:release src/test/hnsw_uncached_bench.nim --mode=all --cache=on
##
## Nothing outside `^HNSWUNCBENCH*` / `^HNSWUNCDUMP*` is touched.

import std/[math, monotimes, parseopt, random, strformat, strutils, times]
import yottadb
import ../hnsw
import ../hnsw_options

const
  Seed = 7
    ## Fixed so every run builds the same graph from the same vectors.
  K = 10
    ## Results per query. Only affects how much the caller copies out, not the
    ## walk, as long as `ef >= K`.
  QuantScale = 1.0'f32 / 127.0'f32
    ## A grid that never clamps an L2-normalised vector (`|x_i| <= 1`).

type
  Cfg = object
    dim: int
    nodes: int
    queries: int
    ef: int
    reps: int
    modes: seq[VecQuant]
    cacheVectors: bool
    dump: bool

proc parseMode(s: string): VecQuant =
  case s.toLowerAscii
  of "none", "float", "float32": vqNone
  of "int8", "sq": vqInt8
  of "int8fixed", "fixed": vqInt8Fixed
  else: quit &"unknown --mode={s} (none | int8 | int8fixed | all)", 2

proc parseCfg(): Cfg =
  result = Cfg(dim: 384, nodes: 5000, queries: 1000, ef: 64, reps: 3,
               modes: @[vqNone, vqInt8])
  var p = initOptParser()
  for kind, key, val in p.getopt():
    case kind
    of cmdEnd: discard
    of cmdShortOption, cmdLongOption:
      case key.toLowerAscii
      of "mode":
        result.modes =
          if val.toLowerAscii == "all": @[vqNone, vqInt8, vqInt8Fixed]
          else: @[parseMode(val)]
      of "dim": result.dim = parseInt(val)
      of "nodes": result.nodes = parseInt(val)
      of "queries": result.queries = parseInt(val)
      of "ef": result.ef = parseInt(val)
      of "reps": result.reps = parseInt(val)
      of "cache", "cachevectors": result.cacheVectors = val.toLowerAscii in ["on", "true", "t"]
      of "dump": result.dump = true
      of "h", "help":
        echo "options: --dim=N --nodes=N --queries=N --ef=N --reps=N " &
             "--mode=none|int8|int8fixed|all --cache=on|off --dump"
        quit 0
      else: quit &"unknown option --{key}", 2
    of cmdArgument: quit &"unexpected argument {key}", 2

proc info(msg: string) =
  ## Everything that is not the dump goes to stderr, so `--dump > file` stays
  ## diffable.
  stderr.writeLine msg

proc resetIndex(p: HnswParams) =
  ## Every tree the index owns, so a rerun does not inherit levels or links from
  ## the previous one.
  for g in [p.globalNode, p.globalKey, p.globalMeta, p.globalLinks,
            p.globalLevel, p.globalQueue]:
    ydb_delete(g, @[], YDB_DEL_TREE)

proc randomVec(rng: var Rand, dim: int): seq[float32] =
  result = newSeq[float32](dim)
  for i in 0 ..< dim:
    result[i] = rng.rand(-1.0'f32 .. 1.0'f32)

proc params(cfg: Cfg, global: string, quant: VecQuant): HnswParams =
  ## `cacheLinks` off as well: an uncached process reads its links from YottaDB
  ## too, which is the realistic shape of the fallback path.
  hnswParams(global, M = 16, efConstruction = 200, efSearch = cfg.ef,
             seed = Seed, dim = cfg.dim, quant = quant,
             quantScale = QuantScale, linkFmt = lfUInt32,
             cacheVectors = cfg.cacheVectors, cacheLinks = false)

proc fill(ix: HnswIndex, cfg: Cfg) =
  var rng = initRand(Seed)
  for i in 0 ..< cfg.nodes:
    discard ix.add(randomVec(rng, cfg.dim), "k" & $i)

proc build(cfg: Cfg, global: string, quant: VecQuant): HnswIndex =
  let p = params(cfg, global, quant)
  resetIndex(p)
  result = openHnsw(p)
  result.fill(cfg)

proc queryBlobs(ix: HnswIndex, want: int): seq[string] =
  ## Stored vectors of the first `want` live ids, as query blobs. Read up front
  ## so the timed loop does not include the query's own YottaDB read.
  var id = 0
  while id < ix.count and result.len < want:
    if ix.hasId(id):
      let b = ydb_get(ix.params.globalNode, @[$id])
      if b.len > 0:
        result.add b
    inc id

proc describe(ix: HnswIndex, cfg: Cfg): string =
  &"{ix.params.global}: count={ix.count} live={ix.liveCount} dim={ix.dim} " &
  &"quant={ix.params.quant} cacheVectors={cfg.cacheVectors} " &
  &"nodes={cfg.nodes} queries={cfg.queries} ef={cfg.ef}"

proc bench(cfg: Cfg, quant: VecQuant) =
  let global = "^HNSWUNCBENCH" & $quant
  let buildStart = getMonoTime()
  var ix = build(cfg, global, quant)
  let built = (getMonoTime() - buildStart).inMilliseconds.float / 1000.0
  let blobs = ix.queryBlobs(cfg.queries)
  info describe(ix, cfg) & &" (built in {built:.1f} s)"
  if blobs.len == 0:
    info "  no live vectors - nothing to measure"
    return

  # One pass that both warms the walk (YottaDB buffers, page cache) and yields
  # the checksum: the sum of every returned distance. Two builds that walk the
  # same graph produce the same sum, and a changed decode would move it.
  var checksum = 0.0'f32
  for b in blobs:
    for h in ix.searchBlob(b, k = K, ef = cfg.ef):
      checksum += h.dist

  var best = Inf
  for _ in 0 ..< cfg.reps:
    let t0 = getMonoTime()
    for b in blobs:
      for h in ix.searchBlob(b, k = K, ef = cfg.ef):
        discard h
    let us = (getMonoTime() - t0).inMicroseconds.float / blobs.len.float
    if us < best:
      best = us

  info &"  {best:7.1f} us/query  ({blobs.len} queries, best of {cfg.reps})" &
       &"  checksum {checksum:.6f}"

proc dump(cfg: Cfg, quant: VecQuant) =
  ## One line per query, byte-comparable between builds.
  let ix = build(cfg, "^HNSWUNCDUMP" & $quant, quant)
  info describe(ix, cfg)
  for id in 0 ..< min(cfg.nodes, ix.count):
    if not ix.hasId(id):
      continue
    let b = ydb_get(ix.params.globalNode, @[$id])
    if b.len == 0:
      continue
    var line = $quant & " q=" & $id & ":"
    for h in ix.searchBlob(b, k = 5, ef = cfg.ef):
      line.add " " & $h.id & ":" & $h.dist
    echo line

proc dumpStale(cfg: Cfg) =
  ## A reader that opened the index while it was empty: its `count`/`entry` are
  ## stale and its vector cache has no slot for any id, so every candidate takes
  ## the fallback even though a cache exists - the second-writer case
  ## `distanceAt` documents. Adopting the writer's shape avoids reopening, which
  ## would preload the cache and hide the path.
  const Global = "^HNSWUNCDUMPS"
  resetIndex(params(cfg, Global, vqNone))
  var reader = openHnsw(params(cfg, Global, vqNone))
  var writer = openHnsw(params(cfg, Global, vqNone))
  writer.fill(cfg)
  reader.count = writer.count
  reader.entry = writer.entry
  reader.maxLevel = writer.maxLevel
  info "staleCache: count=" & $reader.count & " cachedVectors=" & $reader.cachedVectors
  for id in 0 ..< min(cfg.nodes, writer.count):
    if not reader.hasId(id):
      continue
    let b = ydb_get(reader.params.globalNode, @[$id])
    if b.len == 0:
      continue
    var line = "staleCache q=" & $id & ":"
    for h in reader.searchBlob(b, k = 5, ef = cfg.ef):
      line.add " " & $h.id & ":" & $h.dist
    echo line

when isMainModule:
  let cfg = parseCfg()
  if cfg.dump:
    for quant in cfg.modes:
      dump(cfg, quant)
    dumpStale(cfg)
  else:
    if cfg.cacheVectors:
      info "note: cacheVectors is on - only ids the cache does not hold take " &
           "the fallback, so expect a smaller difference than the cache-off numbers"
    for quant in cfg.modes:
      bench(cfg, quant)
