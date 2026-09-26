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
##   ^HNSWxxxMETA("model")           embedding model the vectors come from
##   ^HNSWxxxNODE(id)                the L2-normalised vector, see below
##   ^HNSWxxxLINKS(id, layer)        neighbour ids as int64 (only if non-empty)
##   ^HNSWxxxLEVEL(id)               top layer of the node
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
## The embedding model is named in `HnswParams.model` and pinned in META with the
## first vector, next to the geometry: vectors of two different models cannot be
## compared, so a reopen uses the model recorded in the index, not the one the
## caller happens to pass. `openHnsw` hands that name to the loader the host
## installs through `setModelLoader` (`bert_nim.loadModel` does it when it is
## imported), which keeps this module free of Python - a program that never
## embeds, like the tests, opens indexes without loading a model.
##
## Vectors are L2-normalised on insert, which makes cosine similarity a plain
## dot product and cosine distance `1 - dot`. How they are then stored is
## configurable, per index, through `HnswParams.quant`:
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
##
## How fast a search is depends on two things, and they are not the same size:
##
##   * the distance kernel - `dot`, `dotInt8` and `normalize` run once per
##     candidate. With `-d:hnswSimd` they use nimsimd intrinsics: measured on the
##     384-dim article vectors, 4.1x on the float32 dot (321 -> 78 ns), 6.2x on
##     the int8 dot (353 -> 57 ns) and 7.3x on `normalize` (1054 -> 144 ns).
##     `--passC:"-march=native"` on its own buys nothing here: these are
##     reductions, and gcc will not reassociate them without `-ffast-math`, so
##     the vectorization has to be written out.
##   * the per-candidate database round trip, which costs far more than the
##     arithmetic it feeds: one `ydb_get` + `unpackFloats` on the article index is
##     ~2.1 us against ~0.3 us for the scalar 384-dim dot. One query scores on the
##     order of a thousand candidates, so ~90 % of its 4.6 ms is YottaDB reads.
##     `HnswParams.cacheVectors` (or `preloadVectors`) mirrors the vectors into
##     process memory - 4 * dim * live bytes, 1.1 GB for that index's 732k live
##     vectors - and takes the query to ~0.9 ms. The store is dense, so it is the
##     number of *live* vectors that sizes it, not the id high-water mark `count`
##     (896k on that index, the difference being deletions). That is the bigger
##     lever, and it is what lets the kernel speed matter at all.

import std/[algorithm, heapqueue, math, random, sets, strutils, unicode, sets]
import yottadb

when defined(hnswSimd):
  # 256-bit float / 128-bit integer distance kernels. The C compiler has to
  # target a CPU with AVX (this project passes `-mavx`), which is why it is a
  # define and not the default: without it the scalar loops are used, and those
  # build anywhere. Enable with `-d:hnswSimd`.
  import nimsimd/avx

const DefaultModel* = "paraphrase-multilingual-MiniLM-L12-v2" # 384 dims
#const DefaultModel* = "intfloat/multilingual-e5-base"  # 768 dims
  ## The sentence-transformers model an index uses unless `HnswParams.model` -
  ## or the "model" key of an index that already exists - names another one.

type
  VecQuant* = enum
    ## How an index stores (and compares) its vectors.
    vqNone      ## float32, 4 bytes per dimension - default, lossless
    vqInt8      ## scalar-quantized, grid stored with each vector (4 + dim bytes)
    vqInt8Fixed ## scalar-quantized on one grid for the whole index (dim bytes)

  ModelLoader* = proc(model: string): string
    ## How `openHnsw` reaches the embedding model named in `HnswParams.model`:
    ## load `model` and return the name it ends up loaded under. The host program
    ## installs it with `setModelLoader` - `bert_nim.nim` does that at import
    ## time - which is what keeps this module free of Python. A program that
    ## never embeds (the pure-Nim tests) leaves it nil.

  HnswParams* = object
    model*: string = DefaultModel   ## The hugging face model name
    global*: string                 ## YottaDB global basename ^HNSWxxx
    globalNode*: string             ## YottaDB global ^HNSWxxxNODE
    globalLinks*: string            ## YottaDB global ^HNSWxxxLINKS
    globalLevel*: string            ## YottaDB global ^HNSWxxxLEVEL
    globalKey*: string              ## YottaDB global ^HNSWxxxKEY
    globalMeta*: string             ## YottaDB global ^HNSWxxxMETA
    M*: int = 16                    ## links per node above layer 0
    efConstruction*: int = 200      ## candidate width while inserting
    efSearch*: int = 64             ## default candidate width while searching
    seed*: int = 1234               ## RNG seed for the random level draw
    dim*: int = 0                   ## Dimension
    quant*: VecQuant = vqNone       ## vector storage / quantization mode
    quantScale*: float32 = 0.0'f32  ## the int8 grid for vqInt8Fixed, ignored otherwise
    cacheVectors*: bool = false     ## read every vector into memory on open, see
                                    ## `preloadVectors`
    batchSize*: int = 128           ## Default batch size for g.e. batchedIterator, etc.

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

  VecStore = object
    ## Optional in-process copy of every vector in the index, see
    ## `preloadVectors`.
    ##
    ## Dense: the arena holds one entry per vector actually *present*, and the two
    ## maps below translate between node ids and slots. Keying the arena by id
    ## would size it `count` instead - a high-water mark that deletion pushes far
    ## above the number of live vectors (896349 ids against 732151 vectors on the
    ## article index, and the cleanup pass only widens that gap). A lookup is
    ## still arithmetic plus one copy; the id -> slot indirection is the only
    ## addition.
    ##
    ## Both maps are 8 bytes per entry against `4 * dim` for a float32 vector, so
    ## the arena is what has to stay tight, not them.
    dim: int
    nSlots: int            ## slots filled; the arena may be reserved for more
    floats: seq[float32]   ## vqNone: `dim` floats per slot
    codes: seq[int8]       ## quantized modes: `dim` codes per slot
    scales: seq[float32]   ## vqInt8: one scale per slot
    idToSlot: seq[int]     ## id -> slot, -1 where `id` has no vector here
    slotToId: seq[int]     ## slot -> owning id, `nSlots` entries
    complete: bool         ## the store covers the whole index - only then may
                           ## `hasId` answer from it instead of from YottaDB

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
    cache: VecStore         ## optional in-process copy of the vectors, see
                            ## `preloadVectors`; empty means every read goes to
                            ## YottaDB

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

var zerosCache: HashSet[int]

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

proc bytesPerVector*(ix: HnswIndex): int =
    ## What one stored vector costs, from the mode and the dimension alone - the
    ## blobs do not have to be read back for this.
    case ix.params.quant
    of vqNone: 4 * ix.dim
    of vqInt8: 4 + ix.dim
    of vqInt8Fixed: ix.dim


proc packFloats(xs: openArray[float32]): string =
  result = newString(xs.len * sizeof(float32))
  if xs.len > 0:
    copyMem(result[0].addr, xs[0].unsafeAddr, xs.len * sizeof(float32))

proc unpackFloats(s: string): seq[float32] =
  let n = s.len div sizeof(float32)
  result = newSeqUninit[float32](n)
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
  result = newSeqUninit[int](n)
  if n > 0:
    let src = cast[ptr UncheckedArray[int64]](s[0].unsafeAddr)
    for i in 0 ..< n:
      result[i] = int(src[i])


# ---------------------------------------------------------------------------
# vector helpers
# ---------------------------------------------------------------------------
#
# The two reductions the whole index is built on have hand-written SIMD bodies
# under `-d:hnswSimd`. Two things are worth knowing about them:
#
#   * They are *not* what the compiler would have produced. These are floating
#     point reductions, and without `-ffast-math` no compiler may reassociate
#     them, so `-march=native` alone leaves the scalar loop untouched (measured:
#     321 ns before and after). The parallelism has to be written down.
#   * Doing so changes the summation order, so the last bits of a dot product
#     move. That is fine here - the distance only decides an ordering, and the
#     graph is built with the same kernel it is searched with - but it does mean
#     the SIMD and scalar builds are not bit-identical.

when defined(hnswSimd):
  proc simdDotF32(pa, pb: ptr UncheckedArray[float32], n: int): float32 =
    ## AVX dot product: 8 floats per multiply, 32 per iteration over four
    ## independent accumulators. Four, not one, because a single `acc = acc + x*y`
    ## chain is bound by the add's 3-cycle latency and would run at a third of the
    ## multiply throughput; with four chains the two units stay busy.
    var i = 0
    var acc0 = mm256_setzero_ps()
    var acc1 = mm256_setzero_ps()
    var acc2 = mm256_setzero_ps()
    var acc3 = mm256_setzero_ps()
    while i + 32 <= n:
      acc0 = mm256_add_ps(acc0, mm256_mul_ps(mm256_loadu_ps(pa[i].unsafeAddr),
                                             mm256_loadu_ps(pb[i].unsafeAddr)))
      acc1 = mm256_add_ps(acc1, mm256_mul_ps(mm256_loadu_ps(pa[i + 8].unsafeAddr),
                                             mm256_loadu_ps(pb[i + 8].unsafeAddr)))
      acc2 = mm256_add_ps(acc2, mm256_mul_ps(mm256_loadu_ps(pa[i + 16].unsafeAddr),
                                             mm256_loadu_ps(pb[i + 16].unsafeAddr)))
      acc3 = mm256_add_ps(acc3, mm256_mul_ps(mm256_loadu_ps(pa[i + 24].unsafeAddr),
                                             mm256_loadu_ps(pb[i + 24].unsafeAddr)))
      i += 32
    while i + 8 <= n:
      acc0 = mm256_add_ps(acc0, mm256_mul_ps(mm256_loadu_ps(pa[i].unsafeAddr),
                                             mm256_loadu_ps(pb[i].unsafeAddr)))
      i += 8
    # 256 -> 128 -> 4 -> 1. Two `hadd` pairs reduce the 4 lanes; the store is the
    # cheapest way back into a Nim float32 (nimsimd has no scalar extractor for
    # `M128` without pulling in another header).
    let acc = mm256_add_ps(mm256_add_ps(acc0, acc1), mm256_add_ps(acc2, acc3))
    var s = mm_add_ps(mm256_castps256_ps128(acc), mm256_extractf128_ps(acc, 1))
    s = mm_hadd_ps(s, s)
    s = mm_hadd_ps(s, s)
    var buf: array[4, float32]
    mm_storeu_ps(buf[0].addr, s)
    result = buf[0]
    while i < n:
      result += pa[i] * pb[i]
      inc i

  proc simdDotI8(pa, pb: ptr UncheckedArray[int8], n: int): int32 =
    ## int8 dot product in 128-bit integer SIMD. This machine has AVX but not
    ## AVX2, so there is no 256-bit integer path to take - but 128 bits still
    ## beats scalar by a wide margin: `pmovsxbw` widens 8 bytes to int16 and
    ## `pmaddwd` folds 8 products into 4 int32, so 16 components go through per
    ## iteration. The accumulators stay exact int32.
    var i = 0
    var acc0 = mm_setzero_si128()
    var acc1 = mm_setzero_si128()
    while i + 16 <= n:
      let a0 = mm_loadu_si128(pa[i].unsafeAddr)
      let b0 = mm_loadu_si128(pb[i].unsafeAddr)
      acc0 = mm_add_epi32(acc0, mm_madd_epi16(mm_cvtepi8_epi16(a0),
                                              mm_cvtepi8_epi16(b0)))
      # `cvtepi8_epi16` sign-extends the *low* 8 bytes, so the upper half has to
      # be shifted down first. Both operands are shifted the same way, which is
      # what keeps the pair alignment `pmaddwd` needs.
      acc1 = mm_add_epi32(acc1, mm_madd_epi16(mm_cvtepi8_epi16(mm_srli_si128(a0, 8)),
                                              mm_cvtepi8_epi16(mm_srli_si128(b0, 8))))
      i += 16
    let acc = mm_add_epi32(acc0, acc1)
    var t: array[4, int32]
    mm_storeu_si128(t[0].addr, acc)
    result = t[0] + t[1] + t[2] + t[3]
    while i < n:
      result += int32(pa[i]) * int32(pb[i])
      inc i

proc normalize*(v: var seq[float32]) =
  when defined(hnswSimd):
    ## Two vectorized passes - sum the squares, then scale by 1/norm. Scaling by
    ## the reciprocal instead of dividing per component is a small accuracy trade
    ## that the multiply saves over the divide.
    let n = v.len
    var i = 0
    var acc0 = mm256_setzero_ps()
    var acc1 = mm256_setzero_ps()
    while i + 16 <= n:
      let x0 = mm256_loadu_ps(v[i].addr)
      acc0 = mm256_add_ps(acc0, mm256_mul_ps(x0, x0))
      let x1 = mm256_loadu_ps(v[i + 8].addr)
      acc1 = mm256_add_ps(acc1, mm256_mul_ps(x1, x1))
      i += 16
    let acc = mm256_add_ps(acc0, acc1)
    var s = mm_add_ps(mm256_castps256_ps128(acc), mm256_extractf128_ps(acc, 1))
    s = mm_hadd_ps(s, s)
    s = mm_hadd_ps(s, s)
    var buf: array[4, float32]
    mm_storeu_ps(buf[0].addr, s)
    var sum = buf[0]
    while i < n:
      sum += v[i] * v[i]
      inc i
    let norm = sqrt(sum)
    if norm > 0:
      let inv = mm256_set1_ps(1.0'f32 / norm)
      i = 0
      while i + 8 <= n:
        mm256_storeu_ps(v[i].addr, mm256_mul_ps(mm256_loadu_ps(v[i].addr), inv))
        i += 8
      while i < n:
        v[i] = v[i] * (1.0'f32 / norm)
        inc i
  else:
    var sum = 0.0'f32
    for x in v:
      sum += x * x
    let norm = sqrt(sum)
    if norm > 0:
      for i in 0 ..< v.len:
        v[i] = v[i] / norm

proc dot(a, b: seq[float32]): float32 =
  when defined(hnswSimd):
    if a.len > 0:
      return simdDotF32(cast[ptr UncheckedArray[float32]](a[0].unsafeAddr),
                        cast[ptr UncheckedArray[float32]](b[0].unsafeAddr), a.len)
  else:
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
  result = newSeqUninit[int8](v.len)
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
  result = newSeqUninit[float32](codes.len)
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
  when defined(hnswSimd):
    if a.len > 0:
      return float32(simdDotI8(cast[ptr UncheckedArray[int8]](a[0].unsafeAddr),
                               cast[ptr UncheckedArray[int8]](b[0].unsafeAddr),
                               a.len)) * scaleA * scaleB
    return 0.0'f32
  else:
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
  result = newSeqUninit[int8](s.len)
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

proc cacheSlot(ix: HnswIndex, id: int): int =
  ## The slot holding `id`'s vector, or -1 when the cache has none for it: an id
  ## outside the index, one whose vector was never mirrored, or a deleted one.
  ##
  ## It sits here rather than with the rest of the cache because `loadVec` - the
  ## hot path - is the first thing that needs it.
  if id >= 0 and id < ix.cache.idToSlot.len:
    ix.cache.idToSlot[id]
  else:
    -1


proc loadVec(ix: HnswIndex, id: int): StoredVec =
  ## The vector of `id` as stored - from the in-process cache when there is one,
  ## otherwise one YottaDB read. In the quantized modes this never materializes
  ## float32; the codes come back as they are on disk.
  ##
  ## With a cache the read pattern is what makes this cheap: `searchLayer` asks
  ## for a thousand different ids per query, so a lazy cache (fill on first touch)
  ## would hit almost nothing and only add a lookup - the win comes from
  ## `preloadVectors` having the whole index in memory already.
  ##
  ## An id the cache does not hold - never mirrored, or freed by a delete - falls
  ## through to YottaDB, which is also how a deleted id is recognised: the read
  ## comes back empty.
  let slot = ix.cacheSlot(id)
  if slot >= 0:
    let base = slot * ix.cache.dim
    case ix.params.quant
    of vqNone:
      result.floats = newSeqUninit[float32](ix.cache.dim)
      copyMem(result.floats[0].addr, ix.cache.floats[base].unsafeAddr, ix.cache.dim * sizeof(float32))
    of vqInt8:
      result.codes = newSeqUninit[int8](ix.cache.dim)
      copyMem(result.codes[0].addr, ix.cache.codes[base].unsafeAddr, ix.cache.dim)
      result.scale = ix.cache.scales[slot]
    of vqInt8Fixed:
      result.scale = ix.params.quantScale
      result.codes = newSeqUninit[int8](ix.cache.dim)
      copyMem(result.codes[0].addr, ix.cache.codes[base].unsafeAddr, ix.cache.dim)
    return

  # Check if id is already zeros-cache (old deleted vector)
  if id in zerosCache: 
    return
  let s = ydb_get(ix.params.globalNode, @[$id])
  if s.len == 0:
    zerosCache.incl(id) # add to zeros-cache
    return
  case ix.params.quant
  of vqNone: result.floats = unpackFloats(s)
  of vqInt8: result = unpackQVec(s)
  of vqInt8Fixed:
    result.scale = ix.params.quantScale
    result.codes = unpackInt8(s)

# ---------------------------------------------------------------------------
# in-process vector cache
# ---------------------------------------------------------------------------

proc cacheOn(ix: HnswIndex): bool =
  ## Whether a cache exists at all.
  ix.cache.idToSlot.len > 0

proc cacheIds(ix: HnswIndex, count: int) =
  ## Make the id -> slot map cover every id below `count`. This is the one part
  ## still keyed by id and so the one part that grows with `count` - which is why
  ## it is a plain int seq and not a second arena.
  if count <= ix.cache.idToSlot.len:
    return
  let have = ix.cache.idToSlot.len
  ix.cache.idToSlot.setLen(count)
  for i in have ..< count:
    ix.cache.idToSlot[i] = -1

proc cacheReserve(ix: HnswIndex, slots: int) =
  ## Make room for at least `slots` vectors, and never for fewer - a slot a delete
  ## frees is handed out again by the next `cachePut`, so the reservation does not
  ## have to follow `nSlots` back down.
  ##
  ## `preloadVectors` reserves the whole index in one call, which is what keeps
  ## the arena free of the geometric over-allocation an add-at-a-time build would
  ## accumulate: one `setLen` to the target size is exact, whereas growing one
  ## slot at a time leaves up to a third of the arena unused.
  if slots <= ix.cache.slotToId.len:
    return
  ix.cache.slotToId.setLen(slots)
  case ix.params.quant
  of vqNone:
    ix.cache.floats.setLen(slots * ix.cache.dim)
  of vqInt8:
    ix.cache.codes.setLen(slots * ix.cache.dim)
    ix.cache.scales.setLen(slots)
  of vqInt8Fixed:
    ix.cache.codes.setLen(slots * ix.cache.dim)

proc cachePut(ix: HnswIndex, id: int, v: StoredVec) =
  ## Mirror `v` into the cache. Called from `putVec`, so every write this index
  ## makes keeps the cache coherent and an insert never invalidates it.
  if not ix.cacheOn:
    return
  ix.cacheIds(id + 1)
  if ix.cache.idToSlot[id] < 0:
    # A fresh id - or one whose slot a delete gave back. Reusing the freed slots
    # before reserving more is what keeps a delete/insert cycle flat.
    if ix.cache.nSlots >= ix.cache.slotToId.len:
      ix.cacheReserve(ix.cache.nSlots + 1)
    ix.cache.idToSlot[id] = ix.cache.nSlots
    ix.cache.slotToId[ix.cache.nSlots] = id
    inc ix.cache.nSlots
  let slot = ix.cache.idToSlot[id]
  let base = slot * ix.cache.dim
  case ix.params.quant
  of vqNone:
    if v.floats.len == ix.cache.dim:
      copyMem(ix.cache.floats[base].addr, v.floats[0].unsafeAddr,
              ix.cache.dim * sizeof(float32))
  of vqInt8:
    if v.codes.len == ix.cache.dim:
      copyMem(ix.cache.codes[base].addr, v.codes[0].unsafeAddr, ix.cache.dim)
      ix.cache.scales[slot] = v.scale
  of vqInt8Fixed:
    if v.codes.len == ix.cache.dim:
      copyMem(ix.cache.codes[base].addr, v.codes[0].unsafeAddr, ix.cache.dim)

proc cacheFree(ix: HnswIndex, id: int) =
  ## Drop `id`'s vector from the cache, handing the slot back by swapping the last
  ## one into it. That keeps slots `0 ..< nSlots` exactly the vectors that are
  ## there - no hole per deleted id, which is the whole point of the dense store -
  ## at the cost of one vector copy per delete instead of a free list to walk.
  ##
  ## `slotToId` is what makes the swap possible: without it the moved vector could
  ## not be traced back to the id that has to be told about its new slot.
  let slot = ix.cacheSlot(id)
  if slot < 0:
    return
  let last = ix.cache.nSlots - 1
  if slot != last:
    let moved = ix.cache.slotToId[last]
    let src = last * ix.cache.dim
    let dst = slot * ix.cache.dim
    # `dst + dim <= src`, so the two ranges cannot overlap and a plain copy is
    # enough.
    case ix.params.quant
    of vqNone:
      copyMem(ix.cache.floats[dst].addr, ix.cache.floats[src].unsafeAddr,
              ix.cache.dim * sizeof(float32))
    of vqInt8:
      copyMem(ix.cache.codes[dst].addr, ix.cache.codes[src].unsafeAddr, ix.cache.dim)
      ix.cache.scales[slot] = ix.cache.scales[last]
    of vqInt8Fixed:
      copyMem(ix.cache.codes[dst].addr, ix.cache.codes[src].unsafeAddr, ix.cache.dim)
    ix.cache.slotToId[slot] = moved
    ix.cache.idToSlot[moved] = slot
  ix.cache.idToSlot[id] = -1
  ix.cache.nSlots = last

proc preloadVectors*(ix: HnswIndex): int =
  ## Read every vector into process memory, returning how many were loaded.
  ##
  ## One query scores on the order of a thousand candidates, each of which is a
  ## separate `ydb_get` costing about 2 us - roughly seven times what the
  ## distance it feeds costs to compute. With the vectors in memory a query only
  ## goes to YottaDB for the neighbour lists.
  ##
  ## Memory is `dim * live` bytes in the quantized modes and `4 * dim * live` for
  ## `vqNone` (1.1 GB for the article index's 732k live vectors), plus the
  ## `count`-entry id -> slot map, all owned by the index object. The store is
  ## dense, so the deletions behind `count` no longer inflate it the way an
  ## id-keyed arena would. The pass is O(count) reads, so it belongs in an open
  ## path, not in a query - `HnswParams.cacheVectors` does exactly that.
  ##
  ## The cache is a mirror, never the source of truth: writes go to YottaDB
  ## first and are mirrored here, so nothing is lost if the process dies. The one
  ## thing it cannot see is *another* process writing to the same global; call
  ## this again (or reopen) to pick such changes up.
  if ix.dim <= 0 or ix.count == 0:
    return 0
  ix.cache = VecStore(dim: ix.dim)
  ix.cacheIds(ix.count)
  # `live` comes from META, so the arena is reserved for the vectors that are
  # actually there rather than for the id high-water mark: 4 * dim * live instead
  # of 4 * dim * count. A stale `live` - or another writer - only costs one growth
  # step in the loop below.
  ix.cacheReserve(max(ix.live, 1))
  echo "Preloading up to ", ix.live, " vector(s) of ", ix.count, " id(s)"

  for id in 0 ..< ix.count:
    let s = ydb_get(ix.params.globalNode, @[$id])
    if s.len == 0:
      continue                      # deleted id, or never written
    let slot = ix.cache.nSlots
    case ix.params.quant
    of vqNone:
      if s.len != ix.dim * sizeof(float32):
        continue
      ix.cacheReserve(slot + 1)
      copyMem(ix.cache.floats[slot * ix.dim].addr, s[0].unsafeAddr, s.len)
    of vqInt8:
      if s.len != sizeof(float32) + ix.dim:
        continue
      ix.cacheReserve(slot + 1)
      copyMem(ix.cache.scales[slot].addr, s[0].unsafeAddr, sizeof(float32))
      copyMem(ix.cache.codes[slot * ix.dim].addr, s[sizeof(float32)].unsafeAddr, ix.dim)
    of vqInt8Fixed:
      if s.len != ix.dim:
        continue
      ix.cacheReserve(slot + 1)
      copyMem(ix.cache.codes[slot * ix.dim].addr, s[0].unsafeAddr, ix.dim)
    ix.cache.idToSlot[id] = slot
    ix.cache.slotToId[slot] = id
    ix.cache.nSlots = slot + 1
    inc result
  ix.cache.complete = true
  echo "Cache loaded with ", result, " entries."

proc cachedVectors*(ix: HnswIndex): int =
  ## How many vectors the in-process cache holds (0 when there is none). O(1) now
  ## that the store is dense, instead of a scan over the whole id range.
  ix.cache.nSlots

proc putVec(ix: HnswIndex, id: int, v: StoredVec) =
  ## YottaDB first, then the mirror - never the other way round, so an
  ## interrupted write can only leave the cache behind, not ahead of the
  ## database.
  case ix.params.quant
  of vqNone:
    ydb_set(ix.params.globalNode, @[$id], packFloats(v.floats))
  of vqInt8:
    ydb_set(ix.params.globalNode, @[$id], packQVec(v.scale, v.codes))
  of vqInt8Fixed:
    ydb_set(ix.params.globalNode, @[$id], packInt8(v.codes))
  ix.cachePut(id, v)

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
  ydb_set(ix.params.globalLevel, @[$id], $level)

proc getLevel(ix: HnswIndex, id: int): int =
  let s = ydb_get(ix.params.globalLevel, @[$id])
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
    ydb_set(ix.params.globalLinks, @[$id, $level], packInts(links))

proc getLinks(ix: HnswIndex, id, level: int): seq[int] =
  unpackInts(ydb_get(ix.params.globalLinks, @[$id, $level]))

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

# proc nodeGlobal*(global: string): string = global & "NODE"
# proc keyGlobal*(global: string): string = global & "KEY"
# proc metaGlobal*(global: string): string = global & "META"


# ---------------------------------------------------------------------------
# lifecycle
# ---------------------------------------------------------------------------
var modelLoader: ModelLoader = nil

proc setModelLoader*(loader: ModelLoader) =
  ## Install the way `openHnsw` loads the model named in `HnswParams.model` - see
  ## `ModelLoader`. `bert_nim.loadModel` is what the embedding side registers.
  modelLoader = loader

proc deriveGlobalNames(p: var HnswParams) =
  ## Fill in the `^...NODE` / `^...KEY` / `^...META` names that belong to
  ## `p.global`, unless a caller named them itself.

  if p.globalNode.len == 0: p.globalNode = p.global & "NODE"
  if p.globalLinks.len == 0: p.globalLinks = p.global & "LINKS"
  if p.globalLevel.len == 0: p.globalLevel = p.global & "LEVEL"
  if p.globalKey.len == 0: p.globalKey = p.global & "KEY"
  if p.globalMeta.len == 0: p.globalMeta = p.global & "META"

proc hnswParams*(global: string, model = DefaultModel,
                 M = 16, efConstruction = 200, efSearch = 64, seed = 1234,
                 dim = 0, quant = vqNone, quantScale = 0.0'f32,
                 cacheVectors = false, batchSize = 128): HnswParams =
  ## Configure one index: `global` is the YottaDB global (`^HNSWxxx`) and the
  ## `NODE` / `KEY` / `META` names are derived from it, so no caller spells the
  ## layout out a second time.
  ##
  ## `model`, `dim`, `quant` and `quantScale` only have to be right the first
  ## time; an index that already exists keeps what META says (see `openHnsw`).
  ## `cacheVectors` is not stored in META and is honoured on every open, so it is
  ## the one thing here that always takes effect.
  result = HnswParams(global: global, model: model, M: M,
                      efConstruction: efConstruction, efSearch: efSearch,
                      seed: seed, dim: dim, quant: quant, quantScale: quantScale,
                      cacheVectors: cacheVectors, batchSize: batchSize)
  deriveGlobalNames(result)

proc openHnsw*(p: HnswParams): HnswIndex =
  ## Open (or create) the index described by `p` - the single way in, whether the
  ## params come from `hnswParams(...)` or are written out as an `HnswParams`.
  ##
  ## Params already in the database win over `p`, so reopening an index cannot
  ## silently change its geometry or reinterpret its stored vectors: `M`,
  ## `efConstruction`, `dim`, `quant` and `quantScale` are read back from META,
  ## and `p` only supplies them for an index that does not have them yet.
  ##
  ## The mode of a non-empty index is whatever META says: an index with vectors
  ## but no "quant" key predates quantization and is `vqNone` no matter what `p`
  ## says, because its blobs are float32 and reading them as codes would be
  ## silent nonsense. Changing modes means rebuilding into a new global.
  ##
  ## `p.quantScale` is the int8 grid for `vqInt8Fixed` - take it from
  ## `quantizeScale(sample)` over a representative sample of the collection - and
  ## is required (> 0) in that mode, ignored in the others.
  ##
  ## `p.model` names the sentence-transformers model the index works with, and is
  ## the reason opening an index loads one: once META has the model (written with
  ## the first vector), that recorded name wins and is what gets loaded, so a
  ## reopen always embeds with the model its vectors came from. Loading goes
  ## through `setModelLoader`; with no loader installed the name is only carried.
  new(result)
  result.params = p
  # Fill in `^...NODE` / `^...KEY` / `^...META` from `p.global` if the caller wrote
  # the params out as an `HnswParams` rather than going through `hnswParams` -
  # otherwise an empty name reaches the C API, and YottaDB answers that either
  # from the *previous* call's global (reading something unrelated) or with
  # %YDB-E-INVVARNAME, depending on what ran before. Neither is what the contract
  # above promises.
  deriveGlobalNames(result.params)
  result.rng = initRand(result.params.seed)
  result.entry = -1

  result.params.M = result.metaGet("M", p.M)
  result.params.efConstruction = result.metaGet("efConstruction", p.efConstruction)
  result.dim = result.metaGet("dim", p.dim)
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
    result.params.quant = p.quant
  result.params.quantScale = result.metaGetFloat("qscale", p.quantScale)

  if result.params.quant == vqInt8Fixed and result.params.quantScale <= 0:
    raise newException(ValueError,
      "vqInt8Fixed needs a positive quantScale, e.g. quantizeScale(sample)")

  # The embedding model belongs to the index's identity, like its dimension and
  # its storage mode: a vector is only comparable to one from the same model, so
  # META wins over `p` and a reopen loads the model the vectors were built with.
  # The load happens here, at the end, so a config error above fails before the
  # weights are pulled in. Without a loader (see `setModelLoader`) the name is
  # carried along but nothing is loaded - the tests never import Python.
  result.params.model = result.metaGetName("model", p.model)
  if result.params.model.len > 0 and not modelLoader.isNil:
    # The loader reports back the name it resolved to (a short id, a local path,
    # the default) - that resolved name is what the index is described by.
    result.params.model = modelLoader(result.params.model)

  # The vector cache is not part of an index's identity the way M, dim and the
  # storage mode are - it mirrors what is already there and can be thrown away -
  # so it is the one thing that comes from `p` and not from META. Reading the
  # vectors back is O(count) round trips, hence the explicit opt-in.
  if result.params.cacheVectors:
    discard result.preloadVectors()


proc hasId*(ix: HnswIndex, id: int): bool =
  ## Whether a live node exists at `id`. One YottaDB `data` call, no value read -
  ## and no call at all once `preloadVectors` has made the cache complete.
  if ix.cache.complete and id >= 0 and id < ix.cache.idToSlot.len:
    return ix.cache.idToSlot[id] >= 0
  ydb_data(ix.params.globalNode, @[$id]) != 0

proc liveCount*(ix: HnswIndex): int =
  ## Number of nodes actually in the graph (`ix.count` is the id high-water
  ## mark instead, which is >= this once anything has been deleted).
  ix.live

proc nodeCount*(global: string): int =
  ## Number of nodes straight from YottaDB, without building an index object.
  ## Reads the META global, so it works for any index name.
  let meta = global & "META"
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
  # the storage mode, which decides how every vector blob in the index decodes,
  # and for the embedding model, which decides what the vectors mean. The
  # dimension goes here too - `ix.dim` is settled by now, whether it came from
  # the caller or from the first vector - because everything that reads the index
  # back (including `preloadVectors`) needs it, and an open that only has the
  # default 0 would otherwise find a dimension-less index.
  if ix.count == 0:
    ix.metaSet("M", $ix.params.M)
    ix.metaSet("efConstruction", $ix.params.efConstruction)
    ix.metaSet("dim", $ix.dim)
    ix.metaSet("quant", $ix.params.quant)
    if ix.params.quant == vqInt8Fixed:
      ix.metaSet("qscale", $ix.params.quantScale)
    if ix.params.model.len > 0:
      ix.metaSet("model", ix.params.model)

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
        ydb_delete(ix.params.globalLinks, @[$n, $l], YDB_DEL_NODE)
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

  ydb_delete(ix.params.globalLevel, @[$id], YDB_DEL_NODE)
  ydb_delete(ix.params.globalLinks, @[$id], YDB_DEL_TREE)
  ydb_delete(ix.params.globalNode, @[$id], YDB_DEL_NODE)

  # The id is gone from the cache too. The store is dense, so this leaves no hole
  # behind: the slot goes back and the next `cachePut` reuses it.
  ix.cacheFree(id)

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

proc levelsSummary*(ix: HnswIndex): string =
  ## "node id -> top layer" and the link counts on each layer, for eyeballing
  ## what actually got written to the database. Deleted ids are skipped: they
  ## are holes in the id space, not level-0 nodes.
  var perLevel: seq[int]
  for id in 0 ..< ix.count:
    let s = ydb_get(ix.params.globalLevel, @[$id])
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
