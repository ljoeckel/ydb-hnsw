## HNSW (Hierarchical Navigable Small World) approximate nearest-neighbour
## search, with the graph persisted in YottaDB.
##
## Based on Malkov & Yashunin, "Efficient and robust approximate nearest
## neighbor search using Hierarchical Navigable Small World graphs"
## (arXiv:1603.09320): a multi-layer proximity graph, greedy descent through the
## upper layers, then a best-first search with a candidate width of `ef` on
## layer 0. Neighbour lists are picked with the paper's heuristic (Algorithm 4)
## rather than plain k-nearest, which is what keeps the graph navigable.
##
## YottaDB layout, all under one global (default `^HNSW`). Vectors and
## neighbour lists are packed binary blobs; YottaDB values are length-delimited,
## so embedded NUL bytes are fine:
##
##   ^HNSWxxxMETA("dim")             embedding dimension
##   ^HNSWxxxMETA("M")               max links per node above level 0
##   ^HNSWxxxMETA("efConstruction")  candidate width while inserting
##   ^HNSWxxxMETA("count")           next free id (high-water mark, see below)
##   ^HNSWxxxMETA("live")            number of nodes actually present
##   ^HNSWxxxMETA("entry")           entry-point node id
##   ^HNSWxxxMETA("maxLevel")        current top layer
##   ^HNSWxxxMETA("quant")           vector storage mode (see below)
##   ^HNSWxxxMETA("qscale")          shared int8 grid, only for vqInt8Fixed
##   ^HNSWxxxNODE(id,"vec")          the L2-normalised vector, see below
##   ^HNSWxxxNODE(id,"level")        top layer of this node
##   ^HNSWxxxNODE(id,"links",layer)  neighbour ids as int64 (only if non-empty)
##   ^HNSWxxxKEY(key)                node id for an external key
##
## The key index is what makes incremental loading safe: `lookup` / `contains`
## answer "is this article already in the graph?" without scanning, and
## `addOnce` uses it to insert only genuinely new articles. Keys are arbitrary
## strings, but keep them short and stable (an id or a slug) - YottaDB
## subscripts are limited to about 1 KB, so a full article body is not a good
## key.
##
## Node ids are dense integers 0 ..< count, so iteration never depends on
## YottaDB's lexicographic subscript order.
##
## Deletion (`delete` / `deleteKey`) unlinks the node from every neighbour
## first, then drops its data and both directions of the key mapping, so a
## plain `Data ^...KEY(key)` test correctly reports it as gone. Ids are not
## recycled: `count` only ever grows and is the next-id allocator, while `live`
## tracks how many nodes actually exist - so a deleted id stays a permanent hole
## and `add` always hands out a fresh one.
##
## Unlinking can strand a neighbour whose only link was the node that just went
## away, so `delete` immediately re-links the neighbours it strands and `repair`
## sweeps the index for nodes with no links left (plus stale `entry`/`maxLevel`
## bookkeeping). Both work the same way: search with the stranded node's *own*
## vector - it does not have to be reachable for that - and write fresh links
## from the result, through the same `connect` the insert path uses.
##
## `global` must be a legal YottaDB name: letters, digits and `%` only - no
## underscores (YottaDB rejects "^HNSW_ARTICLES" with %YDB-E-INVVARNAME).
##
## Vectors are L2-normalised on insert, which makes cosine similarity a plain
## dot product and cosine distance `1 - dot`. How they are then stored is
## configurable, per index, through `openHnsw(..., quant = ...)`:
##
##   vqNone       4 * dim bytes - the float32 vector as-is (default)
##   vqInt8       4 + dim bytes - float32 scale, then one int8 code per dimension
##   vqInt8Fixed  dim bytes     - codes on one shared grid (`quantScale`)
##
## Scalar quantization costs at most half a step per component, so the index
## trades a little accuracy for a quarter of the space: measured on the 846k
## vector article index, cos(v, v~) >= 0.9983, 0 components clamped and the top-1
## hit unchanged. The mode is written to META with the first vector, so a
## non-empty index keeps the mode it was built with (a legacy index without a
## "quant" key is vqNone by definition); changing modes means rebuilding into a
## new global, because the blobs would not decode.
##
## Everything that compares two vectors - graph construction, search, the
## neighbour heuristic - works on the *stored* form, so a quantized index builds
## its graph from the very codes it will search later. There is no second,
## inconsistent notion of distance.

import std/[algorithm, heapqueue, math, random, sets, strutils, unicode]
import yottadb

type
  VecQuant* = enum
    ## How an index stores (and compares) its vectors.
    vqNone      ## float32, 4 bytes per dimension - default, lossless
    vqInt8      ## scalar-quantized, grid stored with each vector (4 + dim bytes)
    vqInt8Fixed ## scalar-quantized on one grid for the whole index (dim bytes)

  HnswParams* = object
    global*: string                 ## YottaDB global basename ^HNSWxxx
    globalNode*: string             ## YottaDB global ^HNSWxxxNODE
    globalKey*: string              ## YottaDB global ^HNSWxxxKEY
    globalMeta*: string             ## YottaDB global ^HNSWxxxMETA
    M*: int = 16                    ## links per node above layer 0
    efConstruction*: int = 200      ## candidate width while inserting
    efSearch*: int = 64             ## default candidate width while searching
    seed*: int = 1234               ## RNG seed for the random level draw
    dim*: int = 0                   ## Dimension
    quant*: VecQuant = vqNone       ## vector storage / quantization mode
    quantScale*: float32 = 0.0'f32  ## the int8 grid for vqInt8Fixed, ignored otherwise

  Neighbor* = tuple[dist: float32, id: int]

  Hit* = object
    id*: int
    dist*: float32          ## cosine distance (1 - cosine similarity)

  StoredVec = object
    ## A vector the way the index holds it: `codes` + `scale` for the quantized
    ## modes, `floats` for vqNone. Both fields empty means there is no vector at
    ## that id - deleted, or never written.
    floats: seq[float32]
    codes: seq[int8]
    scale: float32

  HnswIndex* = ref object
    params*: HnswParams
    dim*: int
    count*: int             ## next free id (high-water mark; deletes leave holes)
    live*: int              ## nodes actually present
    entry*: int             ## -1 while the index is empty
    maxLevel*: int
    clamped*: int           ## components clamped to the int8 grid since open. Only
                            ## vqInt8Fixed can clamp (a per-vector grid reaches
                            ## +-127 by construction); a non-zero count there means
                            ## quantScale is too small and accuracy is leaking away
    rng: Rand

  # Two orderings of the same pair, so `heapqueue` can serve as the
  # min-heap (candidates) and the max-heap (current best results).
  AscItem = object
    dist: float32
    id: int
  DescItem = object
    dist: float32
    id: int

proc `<`(a, b: AscItem): bool = a.dist < b.dist
proc `<`(a, b: DescItem): bool = a.dist > b.dist
proc cmpNeighbor(a, b: Neighbor): int = cmp(a.dist, b.dist)

proc sim*(hit: Hit): float32 =
    1.0'f32 - hit.dist

# ---------------------------------------------------------------------------
# text normalization (dedup keys / embedding input)
# ---------------------------------------------------------------------------

const Latin1Fold: array[0xC0 .. 0xFF, string] = [
  # U+00C0 .. U+00CF
  "a", "a", "a", "a", "ae", "aa", "ae", "c",
  "e", "e", "e", "e", "i", "i", "i", "i",
  # U+00D0 .. U+00DF
  "d", "n", "o", "o", "o", "o", "oe", "",
  "oe", "u", "u", "u", "ue", "y", "th", "ss",
  # U+00E0 .. U+00EF
  "a", "a", "a", "a", "ae", "aa", "ae", "c",
  "e", "e", "e", "e", "i", "i", "i", "i",
  # U+00F0 .. U+00FF
  "d", "n", "o", "o", "o", "o", "oe", "",
  "oe", "u", "u", "u", "ue", "y", "th", "y",
]

proc foldLetter*(r: Rune): string =
  ## Transliterate one letter to plain ASCII.
  ##
  ## Latin-1 accents are folded away (é -> e, ñ -> n) and the letters that stand
  ## for a distinct sound are expanded the conventional way, which is what makes
  ## "Türkei" and "Tuerkei" meet: ä -> ae, ö -> oe, ü -> ue, ß -> ss, å -> aa,
  ## æ -> ae, ø -> oe, þ -> th. Characters that fold to nothing (× , ÷) return
  ## an empty string and act as a separator.
  ##
  ## Anything outside Latin-1 - Cyrillic, Greek, CJK, Latin Extended-A - is
  ## returned unchanged, so it stays valid UTF-8 instead of being mangled.
  if r.int in 0xC0 .. 0xFF:
    Latin1Fold[r.int]
  else:
    r.toUTF8

proc hnswNormalize*(title: string): string =
    ## Normalised form of a title: lower-cased, transliterated to plain letters,
    ## and reduced to words of `[a-z0-9]` separated by single spaces.
    ##
    ## Iterates over *runes*, not bytes, so a multi-byte character is one unit
    ## and the result is always valid UTF-8 - a byte-wise filter cuts such
    ## characters in half ("verrückt" loses the second byte of its "ü").
    ##
    ## Every run of anything that is not a letter or digit collapses to a single
    ## space, rather than disappearing. That keeps word boundaries intact:
    ## "M5 chip" stays "m5 chip", where deleting the separator would give
    ## "m5chip" - see the note below on why that matters for the embedding.
    result = newStringOfCap(title.len)
    var pendingSeparator = false
    for r in fasttrim(title).runes:
        let lc = r.toLower
        if lc.isAlpha:
            let folded = foldLetter(lc)
            if folded.len == 0:
                pendingSeparator = true
                continue
            if pendingSeparator and result.len > 0:
                result.add ' '
            pendingSeparator = false
            result.add folded
        elif lc.int in ord('0') .. ord('9'):
            # std/unicode has no isDigit(Rune), hence the explicit range.
            if pendingSeparator and result.len > 0:
                result.add ' '
            pendingSeparator = false
            result.add lc
        else:
            pendingSeparator = true



# ---------------------------------------------------------------------------
# binary encoding
# ---------------------------------------------------------------------------

proc packFloats(xs: openArray[float32]): string =
  result = newString(xs.len * sizeof(float32))
  if xs.len > 0:
    copyMem(result[0].addr, xs[0].unsafeAddr, xs.len * sizeof(float32))

proc unpackFloats(s: string): seq[float32] =
  let n = s.len div sizeof(float32)
  result = newSeq[float32](n)
  if n > 0:
    copyMem(result[0].addr, s[0].unsafeAddr, n * sizeof(float32))

proc packInts(xs: openArray[int]): string =
  result = newString(xs.len * sizeof(int64))
  if xs.len > 0:
    let dst = cast[ptr UncheckedArray[int64]](result[0].addr)
    for i, x in xs:
      dst[i] = int64(x)

proc unpackInts(s: string): seq[int] =
  let n = s.len div sizeof(int64)
  result = newSeq[int](n)
  if n > 0:
    let src = cast[ptr UncheckedArray[int64]](s[0].unsafeAddr)
    for i in 0 ..< n:
      result[i] = int(src[i])

proc packLevel(level: int): string =
  ## Levels are tiny; store them as text so the data is readable in `mupip
  ## extract` / `ydb` output.
  $level


# ---------------------------------------------------------------------------
# vector helpers
# ---------------------------------------------------------------------------

proc normalize*(v: var seq[float32]) =
  var sum = 0.0'f32
  for x in v:
    sum += x * x
  let norm = sqrt(sum)
  if norm > 0:
    for i in 0 ..< v.len:
      v[i] = v[i] / norm

proc dot(a, b: seq[float32]): float32 =
  for i in 0 ..< a.len:
    result += a[i] * b[i]


# ---------------------------------------------------------------------------
# scalar quantization (float32 -> int8)
# ---------------------------------------------------------------------------
#
# One float32 component becomes one signed byte plus one shared step size:
# `x ~= code * scale`, `code = round(x / scale)` clipped to -127 .. 127. That is
# 4 bytes -> 1 byte per dimension, and because the codes are summed in int32 the
# dot product of a stored and a quantized query vector stays exact given the
# codes - the only error is the per-component rounding of at most `scale / 2`.
#
# The quantizer is *symmetric* (no zero point): code 0 is exactly 0.0, which
# keeps the sign of every component and makes `dotInt8` a plain sum of products.
# It also needs no training pass, unlike the per-dimension ranges FAISS learns
# for `QT_8bit`. The price is that a vector whose components are all one-sided
# spends half its codes on values that never occur, and that a *shared* scale is
# pulled down by whichever component is largest anywhere in the collection.
#
# Which scale to use:
#   * `quantizeScale(allVectors)` - one grid for the whole collection. Use this
#     for an index: query and stored vector must land on the same grid or
#     `dotInt8` compares codes that mean different magnitudes.
#   * `quantizeInt8(v)` - a fresh grid per vector (`max|x| / 127`). Best
#     resolution for a single vector, at the cost of a float per vector and of
#     erasing magnitude differences between vectors.

const
  QuantMax = 127'i32
    ## Largest int8 code. The range is -127 .. 127 rather than -128 .. 127 so it
    ## stays symmetric around zero and the round-trip error is bounded by
    ## `scale / 2` on both sides instead of one side getting a half step extra.

proc quantizeScale*(vectors: openArray[seq[float32]]): float32 =
  ## Shared step size for a whole collection: `max|x| / 127` over every
  ## component of every vector.
  ##
  ## Returns 1.0 for an all-zero input, so the codes come out as zeros and
  ## dequantize back to zeros instead of hitting a division by zero.
  var maxAbs = 0.0'f32
  for v in vectors:
    for x in v:
      let a = abs(x)
      if a > maxAbs:
        maxAbs = a
  if maxAbs > 0: maxAbs / float32(QuantMax) else: 1.0'f32

proc quantizeInt8*(v: openArray[float32], scale: float32,
                   clamped: var int): seq[int8] =
  ## As above, but counts the components that had to be clamped to the grid in
  ## `clamped` (incremented, not reset). A non-zero count means `scale` is too
  ## small for this vector - which is the normal case for single-vector
  ## quantization but a real loss for `vqInt8Fixed`, so pick that grid with
  ## `quantizeScale` over the whole collection (or a deliberately generous upper
  ## bound) rather than from one vector.
  result = newSeq[int8](v.len)
  if scale <= 0:
    return
  let inv = 1.0'f32 / scale
  for i, x in v:
    var q = int(round(x * inv))
    if q > QuantMax:
      q = QuantMax
      inc clamped
    elif q < -QuantMax:
      q = -QuantMax
      inc clamped
    result[i] = int8(q)

proc quantizeInt8*(v: openArray[float32], scale: float32): seq[int8] =
  ## Scalar quantization of one vector on a given grid: one signed byte per
  ## component, `x ~= result[i] * scale`.
  ##
  ## Components above the grid (possible when the scale comes from a different
  ## vector) saturate at +-127 instead of wrapping. `scale <= 0` yields all-zero
  ## codes.
  var clamped = 0
  quantizeInt8(v, scale, clamped)

proc quantizeInt8*(v: openArray[float32]): tuple[codes: seq[int8], scale: float32] =
  ## Scalar quantization of one vector on its own grid, returning the codes and
  ## the scale they belong to. Round-trip with
  ## `dequantizeInt8(codes, scale)`, compare with `dotInt8(a, b, sa, sb)`.
  var maxAbs = 0.0'f32
  for x in v:
    let a = abs(x)
    if a > maxAbs:
      maxAbs = a
  result.scale = if maxAbs > 0: maxAbs / float32(QuantMax) else: 1.0'f32
  result.codes = quantizeInt8(v, result.scale)

proc dequantizeInt8*(codes: openArray[int8], scale: float32): seq[float32] =
  ## Back to float32: `code * scale`. This is the lossy step; every component is
  ## off by at most `scale / 2`.
  result = newSeq[float32](codes.len)
  for i, c in codes:
    result[i] = float32(c) * scale

proc dotInt8*(a, b: openArray[int8], scaleA = 1.0'f32,
              scaleB = 1.0'f32): float32 =
  ## Dot product of two quantized vectors: the codes are multiplied and summed
  ## in int32, so that part is exact (127 * 127 * 4096 stays far below 2^31) and
  ## only the final scaling touches the float unit.
  ##
  ## For two L2-normalised vectors that were quantized on the same grid this is
  ## their cosine similarity, with no extra error beyond the quantization.
  doAssert a.len == b.len, "dimension mismatch: " & $a.len & " vs " & $b.len
  var acc = 0'i32
  for i in 0 ..< a.len:
    acc += int32(a[i]) * int32(b[i])
  float32(acc) * scaleA * scaleB

proc packInt8*(xs: openArray[int8]): string =
  ## A dim-byte blob for YottaDB - a quarter of what `packFloats` writes for the
  ## same vector. The scale cannot be recovered from the bytes, so store it
  ## alongside (a second YottaDB node, or once in META if the whole collection
  ## shares one).
  result = newString(xs.len)
  if xs.len > 0:
    copyMem(result[0].addr, xs[0].unsafeAddr, xs.len)

proc unpackInt8*(s: string): seq[int8] =
  ## Inverse of `packInt8`: one byte per dimension, `s.len` == dim.
  result = newSeq[int8](s.len)
  if s.len > 0:
    copyMem(result[0].addr, s[0].unsafeAddr, s.len)


# ---------------------------------------------------------------------------
# vector storage
# ---------------------------------------------------------------------------

proc packQVec(scale: float32, codes: openArray[int8]): string =
  ## A quantized vector as stored: the float32 scale first, then one byte per
  ## dimension, so a single YottaDB read gets both halves (4 + dim bytes).
  result = newString(sizeof(float32) + codes.len)
  copyMem(result[0].addr, scale.unsafeAddr, sizeof(float32))
  if codes.len > 0:
    copyMem(result[sizeof(float32)].addr, codes[0].unsafeAddr, codes.len)

proc unpackQVec(s: string): StoredVec =
  ## Inverse of `packQVec`. An empty string - or anything without room for the
  ## scale - means there is no vector at that id.
  if s.len <= sizeof(float32):
    return
  var scale: float32
  copyMem(scale.addr, s[0].unsafeAddr, sizeof(float32))
  result.scale = scale
  result.codes = unpackInt8(s[sizeof(float32) .. ^1])

proc isMissing(v: StoredVec): bool =
  ## True when the id has no vector: deleted, or never written.
  v.floats.len == 0 and v.codes.len == 0

proc dimOf(v: StoredVec): int =
  if v.codes.len > 0: v.codes.len else: v.floats.len

proc dotStored(a, b: StoredVec): float32 =
  ## Dot product of two stored vectors, in whichever form they are held: int8
  ## multiply-adds for the quantized modes, float32 for vqNone.
  ##
  ## A missing vector, or a length mismatch, scores 0 (= distance 1, the farthest
  ## a pair of L2-normalised vectors can be). `connect` runs over link lists that
  ## can still name a freed id, and that case must not raise.
  if a.isMissing or b.isMissing:
    return 0.0'f32
  if dimOf(a) != dimOf(b):
    return 0.0'f32
  if a.codes.len > 0:
    dotInt8(a.codes, b.codes, a.scale, b.scale)
  else:
    dot(a.floats, b.floats)

proc distanceStored(a, b: StoredVec): float32 =
  ## Cosine distance between two L2-normalised stored vectors.
  1.0'f32 - dotStored(a, b)

proc loadVec(ix: HnswIndex, id: int): StoredVec =
  ## The vector of `id` as stored - one YottaDB read. In the quantized modes this
  ## never materializes float32; the codes come back as they are on disk.
  let s = ydb_get(ix.params.globalNode, @[$id, "vec"])
  if s.len == 0:
    return
  case ix.params.quant
  of vqNone: result.floats = unpackFloats(s)
  of vqInt8: result = unpackQVec(s)
  of vqInt8Fixed:
    result.scale = ix.params.quantScale
    result.codes = unpackInt8(s)

proc putVec(ix: HnswIndex, id: int, v: StoredVec) =
  case ix.params.quant
  of vqNone:
    ydb_set(ix.params.globalNode, @[$id, "vec"], packFloats(v.floats))
  of vqInt8:
    ydb_set(ix.params.globalNode, @[$id, "vec"], packQVec(v.scale, v.codes))
  of vqInt8Fixed:
    ydb_set(ix.params.globalNode, @[$id, "vec"], packInt8(v.codes))

proc toStored(ix: HnswIndex, v: seq[float32]): StoredVec =
  ## Encode an L2-normalised vector the way this index stores it. This is the
  ## only place a vector is quantized, so insertion, repair and search cannot
  ## disagree about the grid.
  case ix.params.quant
  of vqNone:
    result.floats = v
  of vqInt8:
    (result.codes, result.scale) = quantizeInt8(v)
  of vqInt8Fixed:
    result.codes = quantizeInt8(v, ix.params.quantScale, ix.clamped)
    result.scale = ix.params.quantScale


# ---------------------------------------------------------------------------
# YottaDB accessors
# ---------------------------------------------------------------------------

proc putLevel(ix: HnswIndex, id, level: int) =
  ydb_set(ix.params.globalNode, @[$id, "level"], packLevel(level))

proc getLevel(ix: HnswIndex, id: int): int =
  let s = ydb_get(ix.params.globalNode, @[$id, "level"])
  if s.len == 0: 0 else: parseInt(s)


proc putKey(ix: HnswIndex, key: string, id: int) =
  if key.len > 0:
    ydb_set(ix.params.globalKey, @[key], $id)
    ydb_set(ix.params.globalKey, @[$id], key) # inverted

proc lookup*(ix: HnswIndex, key: string): int =
  ## Node id stored for `key`, or -1 if that key was never indexed.
  let s = ydb_get(ix.params.globalKey, @[key])
  if s.len == 0: -1 else: parseInt(s)

proc contains*(ix: HnswIndex, key: string): bool =
  ## Whether `key` is already indexed in *this* index. One YottaDB `Data` call
  ## instead of a value read, so it is the cheap way to skip work for an article
  ## that is already in the graph - and it asks the index that is being built,
  ## not a hard-coded global.
  ydb_data(ix.params.globalKey, @[key]) != 0

proc putLinks(ix: HnswIndex, id, level: int, links: openArray[int]) =
  if links.len > 0:
    ydb_set(ix.params.globalNode, @[$id, "links", $level], packInts(links))

proc getLinks(ix: HnswIndex, id, level: int): seq[int] =
  unpackInts(ydb_get(ix.params.globalNode, @[$id, "links", $level]))

proc metaSet(ix: HnswIndex, key, value: string) =
  ydb_set(ix.params.globalMeta, @[key], value)

proc metaGet(ix: HnswIndex, key: string, default: int): int =
  let s = ydb_get(ix.params.globalMeta, @[key])
  if s.len == 0: default else: parseInt(s)

proc metaGetFloat(ix: HnswIndex, key: string, default: float32): float32 =
  let s = ydb_get(ix.params.globalMeta, @[key])
  if s.len == 0: default else: float32(parseFloat(s))

proc metaGetName(ix: HnswIndex, key, default: string): string =
  ## Text META value (`quant`), with a default for keys an older index lacks.
  let s = ydb_get(ix.params.globalMeta, @[key])
  if s.len == 0: default else: s

proc maxLinks*(ix: HnswIndex, level: int): int =
  ## Level 0 keeps twice as many links - that is where the whole graph gets
  ## walked, so it pays for itself there.
  if level == 0: 2 * ix.params.M else: ix.params.M


# ---------------------------------------------------------------------------
# global names
# ---------------------------------------------------------------------------

proc nodeGlobal*(global: string): string = global & "NODE"
proc keyGlobal*(global: string): string = global & "KEY"
proc metaGlobal*(global: string): string = global & "META"


# ---------------------------------------------------------------------------
# lifecycle
# ---------------------------------------------------------------------------
proc openHnsw*(global: string, M = 16, efConstruction = 200,
               efSearch = 64, seed = 1234, dim = 0, quant = vqNone,
               quantScale = 0.0'f32): HnswIndex =
  ## Open (or create) an index stored in `global`.
  ##
  ## `dim`, `quant` and `quantScale` only have to be passed the first time;
  ## afterwards they are read back from YottaDB. Params already in the database
  ## win over the arguments, so reopening an index cannot silently change its
  ## geometry or reinterpret its stored vectors.
  ##
  ## The mode of a non-empty index is whatever META says: an index with vectors
  ## but no "quant" key predates quantization and is `vqNone` no matter what is
  ## passed here, because its blobs are float32 and reading them as codes would
  ## be silent nonsense. Changing modes means rebuilding into a new global.
  ##
  ## `quantScale` is the int8 grid for `vqInt8Fixed` - take it from
  ## `quantizeScale(sample)` over a representative sample of the collection - and
  ## is required (> 0) in that mode, ignored in the others.
  new(result)
  result.params = HnswParams(globalNode: global.nodeGlobal,
                             globalKey: global.keyGlobal,
                             globalMeta: global.metaGlobal,
                             M: M,
                             efConstruction: efConstruction,
                             efSearch: efSearch, seed: seed)
  result.rng = initRand(seed)
  result.entry = -1

  result.params.M = result.metaGet("M", M)
  result.params.efConstruction = result.metaGet("efConstruction", efConstruction)
  result.dim = result.metaGet("dim", dim)
  result.count = result.metaGet("count", 0)
  # Indexes written before deletion existed have no "live" key; for those every
  # allocated id is still present.
  result.live = result.metaGet("live", result.count)
  result.maxLevel = result.metaGet("maxLevel", 0)
  result.entry = result.metaGet("entry", -1)

  let storedQuant = result.metaGetName("quant", "")
  if storedQuant.len > 0:
    # The index has a storage mode; it wins, like M and dim.
    result.params.quant = parseEnum[VecQuant](storedQuant, vqNone)
  elif result.count > 0:
    # Non-empty and no "quant" key: written before quantization existed.
    result.params.quant = vqNone
  else:
    result.params.quant = quant
  result.params.quantScale = result.metaGetFloat("qscale", quantScale)

  if result.params.quant == vqInt8Fixed and result.params.quantScale <= 0:
    raise newException(ValueError,
      "vqInt8Fixed needs a positive quantScale, e.g. quantizeScale(sample)")


proc openHnsw*(p: HnswParams): HnswIndex =
    openHnsw(p.global, p.M, p.efConstruction, p.efSearch, p.seed, p.dim, p.quant, p.quantScale)

proc hasId*(ix: HnswIndex, id: int): bool =
  ## Whether a live node exists at `id`. One YottaDB `data` call, no value read.
  ydb_data(ix.params.globalNode, @[$id, "vec"]) != 0

proc liveCount*(ix: HnswIndex): int =
  ## Number of nodes actually in the graph (`ix.count` is the id high-water
  ## mark instead, which is >= this once anything has been deleted).
  ix.live

proc nodeCount*(global: string): int =
  ## Number of nodes straight from YottaDB, without building an index object.
  ## Reads the META global, so it works for any index name.
  let meta = global.metaGlobal
  let live = ydb_get(meta, @["live"])
  if live.len > 0:
    return parseInt(live)
  let allocated = ydb_get(meta, @["count"])
  if allocated.len == 0: 0 else: parseInt(allocated)


# ---------------------------------------------------------------------------
# graph construction
# ---------------------------------------------------------------------------

proc randomLevel(ix: HnswIndex): int =
  ## Exponentially decaying level draw: most nodes land on layer 0, a few get
  ## promoted, which is what gives the structure its log-time descent.
  let mL = 1.0 / ln(ix.params.M.float)
  # rand(r, 1.0) is uniform in [0, 1); guard the log against an exact zero.
  let u = max(rand(ix.rng, 1.0), 1e-12)
  int(floor(-ln(u) * mL))

proc searchLayer(ix: HnswIndex, q: StoredVec, entryPoints: openArray[int],
                 ef: int, level: int): seq[Neighbor] =
  ## Greedy best-first search on one layer, against the *stored* vectors (codes
  ## for a quantized index). Returns the `ef` closest nodes found, sorted by
  ## increasing distance.
  var visited = initHashSet[int]()
  var candidates = initHeapQueue[AscItem]()
  var results = initHeapQueue[DescItem]()

  for ep in entryPoints:
    if ep < 0 or ep >= ix.count or visited.containsOrIncl(ep):
      continue
    let ev = ix.loadVec(ep)
    if ev.isMissing:
      continue                        # deleted id (e.g. a stale entry point)
    let d = distanceStored(q, ev)
    candidates.push(AscItem(dist: d, id: ep))
    results.push(DescItem(dist: d, id: ep))

  while candidates.len > 0:
    let c = candidates.pop()
    # Everything left is farther than the worst result we keep -> done.
    if results.len >= ef and c.dist > results[0].dist:
      break
    for e in ix.getLinks(c.id, level):
      if visited.containsOrIncl(e):
        continue
      let ev = ix.loadVec(e)
      if ev.isMissing:
        # Freed id, reached through a link that neighbour pruning left one-way.
        # Treating it as "skip" keeps deleted nodes out of the result set.
        continue
      let d = distanceStored(q, ev)
      if results.len < ef or d < results[0].dist:
        candidates.push(AscItem(dist: d, id: e))
        results.push(DescItem(dist: d, id: e))
        if results.len > ef:
          discard results.pop()   # max-heap: drops the farthest

  # Pop the max-heap to get descending order, then reverse to ascending.
  while results.len > 0:
    let r = results.pop()
    result.add (dist: r.dist, id: r.id)
  result.reverse()

proc selectNeighbors(ix: HnswIndex, candidates: seq[Neighbor], m: int,
                     keepPruned = false): seq[int] =
  ## Malkov & Yashunin Algorithm 4, run on stored vectors. Candidates are
  ## accepted only when they are closer to the base vector than to every
  ## neighbour already accepted; this spreads the links in different directions
  ## instead of clustering them all in one spot. The distances to the base are
  ## already in `candidates`, so the heuristic only needs the vectors to compare
  ## candidates with each other.
  ##
  ## Those vectors are loaded once up front: the naive version re-read the same
  ## blobs from YottaDB once per pairwise comparison, i.e. O(m^2) reads per layer
  ## per insert instead of O(m) - and with the codes in hand the comparisons
  ## themselves are int8 multiply-adds.
  var sorted = candidates
  sorted.sort(cmpNeighbor)

  var vecs = newSeq[StoredVec](sorted.len)
  for i in 0 ..< sorted.len:
    vecs[i] = ix.loadVec(sorted[i].id)

  var chosen: seq[int]
  var pruned: seq[int]
  for i in 0 ..< sorted.len:
    if chosen.len >= m:
      break
    var keep = true
    for j in chosen:
      if distanceStored(vecs[i], vecs[j]) < sorted[i].dist:
        keep = false
        break
    if keep:
      chosen.add i
    else:
      pruned.add i

  for i in chosen:
    result.add sorted[i].id
  if keepPruned:
    # Fill up with the discarded ones rather than returning fewer than `m`.
    for i in pruned:
      if result.len >= m:
        break
      result.add sorted[i].id

proc connect(ix: HnswIndex, id: int, vec: StoredVec, level: int, ep: var int) =
  ## Give `id` neighbours on one layer: search for candidates, pick them with the
  ## heuristic, write both link directions (pruning a neighbour whose list
  ## overflows) and move `ep` to the closest candidate - the entry point for the
  ## next layer down.
  ##
  ## Shared by `add` and by the repair step, so a re-linked node ends up
  ## connected by exactly the same rules as a freshly inserted one.
  var w: seq[Neighbor]
  for cand in ix.searchLayer(vec, [ep], ix.params.efConstruction, level):
    # A node must never link to itself. During `add` this cannot happen (the new
    # node is in nobody's list yet), but a node being re-linked is still in its
    # neighbours' lists, so the search can hand it back as a candidate.
    if cand.id != id:
      w.add cand
  if w.len == 0:
    return

  let neighbors = ix.selectNeighbors(w, ix.params.M)
  ix.putLinks(id, level, neighbors)

  for n in neighbors:
    var nLinks = ix.getLinks(n, level)
    if id in nLinks:
      # Already points back at us - possible while repairing, when only the
      # outgoing half of the link was lost. Do not add a second copy.
      continue
    nLinks.add id
    let limit = ix.maxLinks(level)
    if nLinks.len > limit:
      # Re-run the heuristic from n's point of view.
      let nVec = ix.loadVec(n)
      var cands: seq[Neighbor]
      for x in nLinks:
        cands.add (dist: distanceStored(nVec, ix.loadVec(x)), id: x)
      nLinks = ix.selectNeighbors(cands, limit)
    ix.putLinks(n, level, nLinks)

  ep = w[0].id

proc add*(ix: HnswIndex, vec: openArray[float32], key = ""): int =
  ## Insert one vector as a new node at the next free id (`ix.count`) and link it
  ## into the graph, returning that id.
  ##
  ## Passing `key` records the key -> node id mapping used by `lookup`/`contains`.
  ## `vec` is copied before it is normalised, so the caller's seq is untouched.
  let id = ix.count

  var v = @vec
  if ix.dim == 0:
    ix.dim = v.len
    ix.metaSet("dim", $ix.dim)
  doAssert v.len == ix.dim, "vector has " & $v.len & " dims, index expects " & $ix.dim
  normalize(v)

  let level = ix.randomLevel()

  # Encode once, store that, and build the graph from it - the neighbours are
  # then the ones the stored codes actually have. `v` is L2-normalised by now,
  # which is what makes the quantization grid meaningful.
  let sv = ix.toStored(v)

  # Persist the geometry with the first node: the graph is only valid for the M
  # and efConstruction it was built with, so a later open() must not be able to
  # change them (openHnsw prefers whatever the database already holds). Same for
  # the storage mode, which decides how every vector blob in the index decodes.
  if ix.count == 0:
    ix.metaSet("M", $ix.params.M)
    ix.metaSet("efConstruction", $ix.params.efConstruction)
    ix.metaSet("quant", $ix.params.quant)
    if ix.params.quant == vqInt8Fixed:
      ix.metaSet("qscale", $ix.params.quantScale)

  # The node's own data first: later steps look their neighbours up by id.
  ix.putVec(id, sv)
  ix.putLevel(id, level)
  ix.putKey(key, id)

  # `count` is the id allocator: it only ever grows, so ids freed by delete()
  # stay holes and are never handed out again.
  ix.count += 1
  ix.metaSet("count", $ix.count)
  inc ix.live
  ix.metaSet("live", $ix.live)

  if ix.entry < 0:
    # First node: it becomes the entry point at its own (bootstrap) level.
    ix.entry = id
    ix.maxLevel = level
    ix.metaSet("entry", $ix.entry)
    ix.metaSet("maxLevel", $ix.maxLevel)
    return id

  var ep = ix.entry

  # Phase 1: greedy descent through the layers above ours (ef = 1).
  for l in countdown(ix.maxLevel, level + 1):
    let w = ix.searchLayer(sv, [ep], 1, l)
    if w.len > 0:
      ep = w[0].id

  # Phase 2: connect on every layer we exist on, top down.
  for l in countdown(min(level, ix.maxLevel), 0):
    ix.connect(id, sv, l, ep)

  if level > ix.maxLevel:
    ix.maxLevel = level
    ix.entry = id
    ix.metaSet("entry", $ix.entry)
    ix.metaSet("maxLevel", $ix.maxLevel)

  return id


# ---------------------------------------------------------------------------
# deletion
# ---------------------------------------------------------------------------

proc firstLiveAtLevel(ix: HnswIndex, exceptId, level: int): int =
  ## First live node other than `exceptId` that exists on `level`, or -1.
  ##
  ## Only a node whose own top layer reaches `level` can hold links there, so
  ## this is the only kind of node safe to start a layer-`level` search from. -1
  ## means nobody else is on that layer.
  for cand in 0 ..< ix.count:
    if cand != exceptId and ix.hasId(cand) and ix.getLevel(cand) >= level:
      return cand
  -1

proc relink(ix: HnswIndex, id: int, level: int, vec: StoredVec): bool =
  ## Re-connect `id` on one layer, returning whether it now has links there.
  ##
  ## The node does not have to be reachable for this to work: we search the graph
  ## with the node's *own vector*, so the stranded node is just another query and
  ## the result is a fresh set of neighbours.
  ##
  ## Returns false without writing anything when there is nothing to link to -
  ## a node that is alone on its top layer is *supposed* to have an empty list
  ## there, and inventing a link to a node from a lower layer would corrupt that
  ## layer's invariant.
  if ix.live < 2:
    return false

  var ep = ix.entry
  if ep == id or not ix.hasId(ep) or ix.getLevel(ep) < level:
    ep = ix.firstLiveAtLevel(id, level)
  if ep < 0:
    return false

  # the same greedy descent the insert path uses to reach this layer
  for l in countdown(min(ix.maxLevel, ix.getLevel(ep)), level + 1):
    let w = ix.searchLayer(vec, [ep], 1, l)
    if w.len > 0:
      ep = w[0].id

  ix.connect(id, vec, level, ep)
  ix.getLinks(id, level).len > 0

proc unlink(ix: HnswIndex, id: int): seq[(int, int)] =
  ## Drop `id` from every neighbour's link list, on each layer the node exists
  ## on. This has to run *before* the node's own data disappears, because the
  ## node's own link lists are the only list of who points back at it.
  ##
  ## Returns the (neighbour, layer) pairs that were left with no links at all -
  ## those are the nodes a delete would strand, and the caller re-links them.
  let level = ix.getLevel(id)
  for l in 0 .. level:
    for n in ix.getLinks(id, l):
      if not ix.hasId(n):
        continue                      # neighbour already deleted as well
      let nLinks = ix.getLinks(n, l)
      var kept: seq[int]
      for x in nLinks:
        if x != id:
          kept.add x
      if kept.len == nLinks.len:
        continue                      # link was already pruned one-way
      if kept.len == 0:
        ydb_delete(ix.params.globalNode, @[$n, "links", $l], YDB_DEL_NODE)
        result.add (n, l)
      else:
        ix.putLinks(n, l, kept)

proc electEntry(ix: HnswIndex) =
  ## Re-elect the entry point after the old one was deleted: the live node on
  ## the highest layer, with `maxLevel` following it.
  ##
  ## This is the O(count) path, and it only runs when the entry point itself is
  ## removed. Entry points live on the top layer, so that is rare; ordinary
  ## deletes stay O(neighbours).
  var bestId = -1
  var bestLevel = -1
  for id in 0 ..< ix.count:
    if not ix.hasId(id):
      continue
    let lv = ix.getLevel(id)
    if lv > bestLevel:
      bestLevel = lv
      bestId = id
  if bestId < 0:
    ix.entry = -1
    ix.maxLevel = 0
  else:
    ix.entry = bestId
    ix.maxLevel = bestLevel
  ix.metaSet("entry", $ix.entry)
  ix.metaSet("maxLevel", $ix.maxLevel)

proc delete*(ix: HnswIndex, id: int): bool =
  ## Delete one node, returning whether there was one at `id`.
  ##
  ## Unlink first, then drop data and key mappings - so `Data ^...KEY(key)`
  ## correctly reports the article as gone; re-adding the same article after that
  ## simply allocates a new id. If the deleted node was the entry point, a new
  ## one is elected from the remaining nodes.
  if not ix.hasId(id):
    return false

  let stranded = ix.unlink(id)

  # Both directions of the key index. The `lookup(key) == id` guard matters
  # because forward and inverted mappings share one global: a key that happens
  # to look like an id must not delete somebody else's entry.
  let key = ydb_get(ix.params.globalKey, @[$id])
  ydb_delete(ix.params.globalKey, @[$id], YDB_DEL_NODE)
  if key.len > 0 and ix.lookup(key) == id:
    ydb_delete(ix.params.globalKey, @[key], YDB_DEL_NODE)

  # YDB_DEL_TREE (= 1) removes the node and everything under it: vec, level and
  # every links/<layer>. YDB_DEL_NODE (= 2) would only clear this node's value.
  ydb_delete(ix.params.globalNode, @[$id], YDB_DEL_TREE)

  dec ix.live
  ix.metaSet("live", $ix.live)

  if ix.entry == id:
    ix.electEntry()

  # A neighbour whose only link was this node has just become unreachable. Give
  # it new ones now, rather than leaving it to rot until someone runs `repair`.
  # Deliberately after the node's own data is gone, so these searches cannot run
  # into the node being deleted.
  for (n, l) in stranded:
    if ix.hasId(n) and ix.getLinks(n, l).len == 0:
      discard ix.relink(n, l, ix.loadVec(n))

  true

proc deleteKey*(ix: HnswIndex, key: string): bool =
  ## Delete the node stored under `key`, returning whether there was one.
  let id = ix.lookup(key)
  if id < 0:
    return false
  ix.delete(id)

proc repair*(ix: HnswIndex): int =
  ## Sweep the index and re-connect live nodes that have no links left, returning
  ## how many layers were re-linked.
  ##
  ## `delete` already re-links the neighbours it strands, so this is for damage
  ## from earlier (deletions that predate the repair) or from an interrupted
  ## run. It also repairs stale `entry` / `maxLevel` bookkeeping, which a broken
  ## entry point can otherwise turn into "search returns nothing".
  ##
  ## Costs one `links` read per layer per live node, i.e. O(count) YottaDB calls -
  ## maintenance work, not something to call in a loop.
  result = 0
  if ix.count == 0:
    return

  # Fix the entry point first: every relink below searches from it, and a dead
  # entry point makes all of them come back empty.
  if not ix.hasId(ix.entry):
    ix.electEntry()

  var bestId = -1
  var bestLevel = -1
  for id in 0 ..< ix.count:
    if not ix.hasId(id):
      continue
    let lv = ix.getLevel(id)
    if lv > bestLevel:
      bestLevel = lv
      bestId = id

    # every layer this node exists on that has lost all its links
    var vec: StoredVec
    var haveVec = false
    for l in 0 .. lv:
      if ix.getLinks(id, l).len > 0:
        continue
      if not haveVec:
        vec = ix.loadVec(id)
        haveVec = true
      if ix.relink(id, l, vec):
        inc result

  # the entry point must be a live node on the top layer, or searches start from
  # a corpse and come back empty
  if bestId < 0:
    if ix.entry != -1:
      ix.entry = -1
      ix.maxLevel = 0
      ix.metaSet("entry", $ix.entry)
      ix.metaSet("maxLevel", $ix.maxLevel)
  elif ix.entry < 0 or not ix.hasId(ix.entry) or ix.getLevel(ix.entry) < bestLevel:
    ix.entry = bestId
    ix.maxLevel = bestLevel
    ix.metaSet("entry", $ix.entry)
    ix.metaSet("maxLevel", $ix.maxLevel)


proc search*(ix: HnswIndex, vec: openArray[float32], k = 5, ef = 0): seq[Hit] =
  ## The `k` closest nodes to `vec`, nearest first.
  ##
  ## `ef` is the search candidate width (>= k); it defaults to the index's
  ## efSearch. Larger values trade speed for recall.
  if ix.entry < 0 or ix.count == 0:
    return @[]
  var v = @vec
  doAssert v.len == ix.dim, "vector has " & $v.len & " dims, index expects " & $ix.dim
  normalize(v)
  # The query goes onto the same grid as the stored vectors - otherwise the codes
  # compared in `searchLayer` would not mean the same magnitudes.
  let q = ix.toStored(v)

  let width = max(if ef > 0: ef else: ix.params.efSearch, k)

  var ep = ix.entry
  for l in countdown(ix.maxLevel, 1):
    let w = ix.searchLayer(q, [ep], 1, l)
    if w.len > 0:
      ep = w[0].id

  let w = ix.searchLayer(q, [ep], width, 0)
  for i in 0 ..< min(k, w.len):
    result.add Hit(id: w[i].id, dist: w[i].dist)


# ---------------------------------------------------------------------------
# introspection
# ---------------------------------------------------------------------------

#proc vectorOf*(ix: HnswIndex, id: int): seq[float32] =
#  ix.getVec(id)

#proc payloadOf*(ix: HnswIndex, id: int): string =
#  ix.getPayload(id)

#proc levelOf*(ix: HnswIndex, id: int): int =
#  ix.getLevel(id)

#proc linksOf*(ix: HnswIndex, id, level: int): seq[int] =
#  ix.getLinks(id, level)

proc levelsSummary*(ix: HnswIndex): string =
  ## "node id -> top layer" and the link counts on each layer, for eyeballing
  ## what actually got written to the database. Deleted ids are skipped: they
  ## are holes in the id space, not level-0 nodes.
  var perLevel: seq[int]
  for id in 0 ..< ix.count:
    let s = ydb_get(ix.params.globalNode, @[$id, "level"])
    if s.len == 0:
      continue
    let lv = parseInt(s)
    if lv >= perLevel.len:
      perLevel.setLen(lv + 1)
    for l in 0 .. lv:
      perLevel[l] += 1
  for l, n in perLevel:
    if l > 0:
      result.add "  "
    result.add "L" & $l & ": " & $n
  result.add " node(s), entry=" & $ix.entry & " maxLevel=" & $ix.maxLevel
