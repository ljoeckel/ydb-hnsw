## Where does a query's 4.6 ms actually go?
##
## `search` on the article index (`^HNSWArticles`, 730k live vectors, dim 384,
## vqNone, ef=64) takes milliseconds, while the float32 dot product for one
## candidate is well under a microsecond. This measures the per-access costs the
## walk pays instead - one `ydb_get` plus one `unpackFloats` per candidate - so
## the arithmetic and the I/O can be compared directly.
##
## Run: nim c -r -d:release -p:src src/test/hnsw_ydb_bench.nim

import std/[math, random, strformat, strutils, times]
import yottadb

const
  Global = "^HNSWArticles"
  Ops = 200_000

proc unpackFloats(s: string): seq[float32] =
  let n = s.len div sizeof(float32)
  result = newSeq[float32](n)
  if n > 0:
    copyMem(result[0].addr, s[0].unsafeAddr, n * sizeof(float32))

proc unpackLinks(s: string): seq[int] =
  let n = s.len div sizeof(int64)
  result = newSeq[int](n)
  if n > 0:
    let src = cast[ptr UncheckedArray[int64]](s[0].unsafeAddr)
    for i in 0 ..< n:
      result[i] = int(src[i])

proc scalarDot(a, b: openArray[float32]): float32 =
  for i in 0 ..< a.len:
    result += a[i] * b[i]

var sink = 0.0'f64

proc timeIt(tag: string, body: proc()) =
  let t0 = epochTime()
  body()
  let dt = (epochTime() - t0) * 1e9 / float64(Ops)
  echo &"  {tag:<34} {dt:9.1f} ns/op"

proc main() =
  let meta = Global & "META"
  let node = Global & "NODE"
  let countS = ydb_get(meta, @["count"])
  if countS.len == 0:
    echo "no index in " & Global
    return
  let count = parseInt(countS)
  echo &"{Global}: count={count}"

  # A set of live ids, plus one vector to dot against.
  var ids: seq[int]
  var rng = initRand(7)
  while ids.len < 5000:
    let id = rand(rng, count - 1)
    if ydb_data(node, @[$id, "vec"]) != 0:
      ids.add id
  let q = unpackFloats(ydb_get(node, @[$ids[0], "vec"]))
  echo &"  query dim {q.len}, blob {q.len * 4} bytes"

  var i = 0
  timeIt("ydb_get(vec) + unpack", proc() =
    for _ in 0 ..< Ops:
      let v = unpackFloats(ydb_get(node, @[$ids[i mod ids.len], "vec"]))
      sink += v[0]
      inc i)
  timeIt("ydb_get(links) + unpackInts", proc() =
    for _ in 0 ..< Ops:
      let l = unpackLinks(ydb_get(node, @[$ids[i mod ids.len], "links", "0"]))
      sink += float64(l.len)
      inc i)
  timeIt("ydb_data(vec)", proc() =
    for _ in 0 ..< Ops:
      sink += float64(ydb_data(node, @[$ids[i mod ids.len], "vec"]))
      inc i)
  timeIt("scalar dot (384d)", proc() =
    for _ in 0 ..< Ops:
      sink += scalarDot(q, q))
  timeIt("seq alloc 384 float32 + copy", proc() =
    for _ in 0 ..< Ops:
      var v = newSeq[float32](384)
      sink += v[0])

  echo "\n=> a level-0 walk that touches N candidates pays N * (ydb_get+unpack)"

main()
