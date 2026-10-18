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
##   ^HNSWxxxMETA("linkfmt")         how LINKS stores one neighbour id, see below
##   ^HNSWxxxNODE(id)                the L2-normalised vector, see below
##   ^HNSWxxxLINKS(id, layer)        neighbour ids, 8 or 4 bytes each
##                                   (only if non-empty)
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
## A neighbour id on disk is either `int64` (`lfInt`, 8 bytes, and
## what every index built before the option existed holds) or `uint32`
## (`lfUint32`, 4 bytes) as default. The narrow form halves the LINKS tree, which is where
## most of a node's bytes go - a level-0 list is `2M` ids - at the cost of
## capping the id space at 2^32 - 1, because ids come from a counter that never
## reuses one. `add` refuses to allocate past that, so the cap is a hard error at
## the insert that would cross it, not silent truncation.
##
## Like `quant` and `model`, the width is a property of the *index*, not of the
## build: it is pinned in META with the first node, so a reopen decodes the blobs
## the way they were written and two indexes of different widths can be open at
## once. An index that has neighbour lists but no "linkfmt" key predates the
## option and is `lfInt` by definition. Nothing else depends on the width -
## `HnswIndex` keeps ids as `int` throughout and only `packLinks` / `unpackLinks`
## see the stored form.
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
##
## `HnswParams.cacheLinks` covers the rest of the round trip. The vectors are its
## bulk, but every node the walk expands also costs one neighbour-list read - a
## `ydb_get` plus the `$id` / `$level` subscript strings plus an `unpackLinks`
## copy, roughly a hundred per query. That mirror is filled on demand rather than
## preloaded (a list is read the first time it is asked for and then kept: `2M`
## ids at layer 0, `M` per layer above it, 8 bytes each, ~13 MB for the article
## index's 98k nodes), which is what keeps a node another process added from
## being mistaken for a node with no links.

import std/[algorithm, math, random, sets, times, strutils, unicode]
import yottadb
import hnsw_options

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
  ModelLoader* = proc(model: string): string
    ## How `openHnsw` reaches the embedding model named in `HnswParams.model`:
    ## load `model` and return the name it ends up loaded under. The host program
    ## installs it with `setModelLoader` - `bert_nim.nim` does that at import
    ## time - which is what keeps this module free of Python. A program that
    ## never embeds (the pure-Nim tests) leaves it nil.

  Neighbor* = tuple[dist: float32, id: int]

  Hit* = object
    id*: int
    dist*: float32          ## cosine distance (1 - cosine similarity)

  NeighborView = object
    ## Borrowed neighbour results from one `searchLayer` call: `n` entries at
    ## `p`, **nearest first**.
    ##
    ## The entries live in `SearchScratch.found`, so a view is valid only until the
    ## next `searchLayer` on the same index. That is the price for removing an
    ## allocation from every layer of every query and every insert; the rule is
    ## the same one `VecView` carries, and every current caller already keeps it
    ## (the descent loops read `[0]` immediately, `connect` copies the entries
    ## into its own list, and `searchStored` copies its `k` hits out).
    p: ptr UncheckedArray[Neighbor]
    n: int

  StoredVec = object
    ## A vector the way the index holds it: `codes` + `scale` for the quantized
    ## modes, `floats` for vqNone. Both fields empty means there is no vector at
    ## that id - deleted, or never written.
    floats: seq[float32]
    codes: seq[int8]
    scale: float32

  VecView* = object
    ## A borrowed, non-owning view of one stored vector: the address of its codes
    ## (quantized modes) or floats (vqNone), plus the scale that belongs to the
    ## codes.
    ##
    ## It exists so the hot path can compare against the bytes the in-process
    ## cache already holds. `loadVec` returns an *owned* `StoredVec` - it
    ## allocates a seq and copies the vector into it - and `searchLayer` does one
    ## distance with that and throws it away: on the article index a couple of
    ## thousand 1536-byte allocations and copies per query, every one of them
    ## discarded. A view costs a pointer and a length instead, and the distance
    ## runs on the arena's own bytes (`dotView`, `distanceAt`).
    ##
    ## What a view costs *instead* is a lifetime rule: it points into a seq it
    ## does not own, so it may only be used while that seq is alive - and, for
    ## the cache arena, while the arena is not resized. `searchLayer` satisfies
    ## both: the query outlives the call, and a search never writes a vector, so
    ## nothing reaches `cacheReserve`/`setLen` between the first candidate and
    ## the last.
    floats: ptr UncheckedArray[float32]
    codes: ptr UncheckedArray[int8]
    scale: float32
    dim: int

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

  LinkStore = object
    ## Optional in-process mirror of `^...LINKS`, see `HnswParams.cacheLinks`.
    ##
    ## This is the link half of `VecStore`, and the two are separate because
    ## their shapes differ: a vector is one blob per node, a link list is one blob
    ## per (node, layer) and almost every node has only layer 0. The table is
    ## dense - `id * stride + level` selects a slot - because ids are dense
    ## integers, and hashing a value that is already an index is exactly what
    ## `seenOrMark` replaced a `HashSet` for.
    ##
    ## Unlike `VecStore` there is no preload and no `complete`: an entry is
    ## mirrored the first time something asks for it, and a miss always falls back
    ## to YottaDB. That is deliberate. `^...LINKS` has no node where the list is
    ## empty, so in the database's eyes "not mirrored" and "no links" are the
    ## same thing; filling lazily is what keeps a node a second writer added -
    ## whose subscripts this process has never seen - from being read as empty.
    ## The price is one read per distinct (node, layer) ever touched, around a
    ## hundred per query and shared by every later one, against the `count` reads
    ## a preload would spend on open.
    ##
    ## `lens` is the one field that says whether an entry is known: -1 not
    ## mirrored (ask YottaDB), 0 mirrored and empty, n > 0 mirrored with n ids at
    ## `offset`. `caps` is the room reserved for the entry - `2M`, the layer-0
    ## limit and so enough for every higher layer too - so a list that grows while
    ## the node is linked in does not have to move.
    stride: int              ## layers per id in the table (top layer + 1)
    ids: int                 ## ids covered: 0 ..< ids
    lens: seq[int32]         ## -1 unknown, 0 empty, else the link count
    caps: seq[int32]         ## arena slots reserved for the entry, 0 when none
    offset: seq[int32]       ## arena offset of the entry, -1 when it has none
    arena: seq[int]          ## the neighbour ids themselves

  HnswHeap = object
    ## The binary heap the walk runs on: `dists` holds the keys, `ids` the node
    ## each key belongs to, and `n` is how many of them are live.
    ##
    ## `heapqueue` was costing the hot loop more than the heap is worth. It keeps
    ## a `seq[T]` of 12-byte tuples, so consecutive comparisons land 16 bytes
    ## apart, and every use of `.len` in the loop is a call - with
    ## `results.len >= ef` sitting in the middle of the inner loop. Two arrays
    ## keep the keys adjacent, `n` is a plain `int` the loop holds in a register,
    ## and the sift moves parents into the hole instead of swapping on every
    ## level. Nothing is ever taken out of the arrays, only `n` moves, so a push
    ## after a pop reuses the space.
    ##
    ## There is one implementation and it is a min-heap. Both orderings the
    ## search needs come from the key: `candidates` pushes `dist`, so the nearest
    ## is on top and popped next, and `results` pushes `-dist`, so the *farthest*
    ## is on top and evicted first. Negating a float is exact, so no ordering
    ## changes - `worst` is the one place that convention is visible.
    dists: seq[float32]
    ids: seq[int]
    n: int

  SearchScratch = object
    ## The buffers one `searchLayer` call works in, kept across calls so a walk
    ## stops allocating after the first few.
    ##
    ## Per index, like `visitedStamp`, and for the same reason `searchLayer` is
    ## not reentrant today: a query-level parallel driver will give each thread
    ## its own searcher (or index) rather than share these. `found` is the one
    ## field with a lifetime rule - `NeighborView` points into it, so it may not
    ## be resized until the caller is done with the view.
    over: seq[int]          ## visited ids at or past `count` - another writer's
    spare: seq[int]         ## one link-list read the mirror did not have
    candidates: HnswHeap    ## frontier, keyed on distance (nearest on top)
    results: HnswHeap       ## the `ef` kept, keyed on `-distance` (farthest on top)
    found: seq[Neighbor]    ## result buffer, filled farthest-first

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
    visitedStamp: seq[int32] ## `searchLayer`'s visited set: one epoch stamp per
                             ## node id (see `seenOrMark`). Ids are dense, so
                             ## "seen this node?" is one array read; the HashSet
                             ## this replaces was ~25% of a build profile -
                             ## hashing a value that is already an index.
    visitEpoch: int32        ## the stamp the current `searchLayer` uses
    absentIds: HashSet[int] ## ids this index has already found to have no vector:
                            ## deleted, or named by a one-way link that pruning
                            ## left behind. A memo, not state - `count` never
                            ## hands an id out twice, so an id that is empty here
                            ## stays empty, and what it saves is a YottaDB read
                            ## per stale reference in the hot path.
                            ##
                            ## Deliberately per index, never process-wide: an id
                            ## going missing in one index says nothing about
                            ## another, and a global memo leaks exactly that way
                            ## - the entry from one index's delete makes a plain
                            ## `loadVec` skip a live node in the next one, while
                            ## an index with a preloaded cache still answers from
                            ## the cache and gets it right. Two readers of the
                            ## same globals then disagree, which is the one thing
                            ## a mirror must not do.
    cache: VecStore         ## optional in-process copy of the vectors, see
                            ## `preloadVectors`; empty means every read goes to
                            ## YottaDB
    links: LinkStore        ## optional in-process mirror of the neighbour lists,
                            ## see `HnswParams.cacheLinks`; empty means every read
                            ## goes to YottaDB
    scratch: SearchScratch  ## buffers `searchLayer` reuses; see `SearchScratch`

func heapPush(h: var HnswHeap, dist: float32, id: int) {.inline.} =
  ## Add `(dist, id)` to the min-heap `h`.
  ##
  ## The arena doubles, so re-heaping a whole layer's frontier does not
  ## reallocate once per candidate.
  if h.n == h.dists.len:
    let cap = if h.n == 0: 64 else: h.n * 2
    h.dists.setLen(cap)
    h.ids.setLen(cap)
  var i = h.n
  inc h.n
  # Sift up: move each parent down into the hole, then drop the new key in. A
  # swap per level would write every entry twice.
  while i > 0:
    let parent = (i - 1) shr 1
    if dist >= h.dists[parent]:
      break
    h.dists[i] = h.dists[parent]
    h.ids[i] = h.ids[parent]
    i = parent
  h.dists[i] = dist
  h.ids[i] = id

func heapPop(h: var HnswHeap): tuple[dist: float32, id: int] {.inline.} =
  ## Remove and return the smallest key. `h.n` has to be > 0 - the walk checks
  ## it, because a heap it can empty is how a layer ends.
  result = (h.dists[0], h.ids[0])
  dec h.n
  if h.n > 0:
    # The last entry takes the root's place and sifts back down into the hole.
    let dist = h.dists[h.n]
    let id = h.ids[h.n]
    var i = 0
    while true:
      var child = 2 * i + 1
      if child >= h.n:
        break
      if child + 1 < h.n and h.dists[child + 1] < h.dists[child]:
        inc child                  # the smaller of the two children
      if h.dists[child] >= dist:
        break
      h.dists[i] = h.dists[child]
      h.ids[i] = h.ids[child]
      i = child
    h.dists[i] = dist
    h.ids[i] = id

func worst(h: HnswHeap): float32 {.inline.} =
  ## The distance the `results` heap would evict next: the farthest the walk is
  ## still keeping. That heap is the min-heap keyed on `-dist`, so its top is the
  ## largest distance, and this negation is the only place the convention shows.
  ## `h.n` must be > 0, which every caller's `ef` guarantees.
  -h.dists[0]

func len(v: NeighborView): int {.inline.} =
  v.n

func `[]`(v: NeighborView, i: int): Neighbor {.inline.} =
  doAssert i >= 0 and i < v.n
  v.p[i]

iterator items(v: NeighborView): Neighbor =
  for i in 0 ..< v.n:
    yield v.p[i]

func sim*(hit: Hit): float32 =
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

func foldLetter*(r: Rune): string =
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

func hnswNormalize*(title: string): string =
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

func bytesPerVector*(ix: HnswIndex): int =
    ## What one stored vector costs, from the mode and the dimension alone - the
    ## blobs do not have to be read back for this.
    case ix.params.quant
    of vqNone: 4 * ix.dim
    of vqInt8: 4 + ix.dim
    of vqInt8Fixed: ix.dim

func linkBytes*(fmt: LinkFmt): int =
    ## What one stored neighbour id costs, so a level-0 list of `2M` links is
    ## `2M * linkBytes` bytes. The width is a property of the index, not of the
    ## process, so it has to be asked for rather than looked up.
    case fmt
    of lfInt: sizeof(int64)
    of lfUint32: sizeof(uint32)


func packFloats*(xs: openArray[float32]): string =
  result = newString(xs.len * sizeof(float32))
  if xs.len > 0:
    copyMem(result[0].addr, xs[0].unsafeAddr, xs.len * sizeof(float32))

func unpackFloats*(s: string): seq[float32] =
  let n = s.len div sizeof(float32)
  result = newSeqUninit[float32](n)
  if n > 0:
    copyMem(result[0].addr, s[0].unsafeAddr, n * sizeof(float32))

func packLinks(fmt: LinkFmt, xs: openArray[int]): string =
  ## One neighbour list as it goes into YottaDB, in the index's link width.
  ##
  ## `lfUint32` is the only place an id could be lost, and the range it can lose
  ## it in starts at 4 billion: `add` refuses to allocate an id past `high(uint32)`
  ## while an index is in that mode, so what is left here is the debug tripwire
  ## for that invariant rather than the release-path guard.
  case fmt
  of lfInt:
    result = newString(xs.len * sizeof(int64))
    if xs.len > 0:
      let dst = cast[ptr UncheckedArray[int64]](result[0].addr)
      for i, x in xs:
        dst[i] = int64(x)
  of lfUint32:
    result = newString(xs.len * sizeof(uint32))
    if xs.len > 0:
      let dst = cast[ptr UncheckedArray[uint32]](result[0].addr)
      for i, x in xs:
        doAssert x >= 0 and uint64(x) <= uint64(high(uint32))
        dst[i] = uint32(x)

func unpackLinks(fmt: LinkFmt, s: string): seq[int] =
  ## Inverse of `packLinks`, always back into `int` - the width is the on-disk
  ## detail, the graph works in ids.
  ##
  ## Which width to use comes from the index and never from `s.len`: an `lfInt`
  ## blob read as `lfUint32` would yield twice as many "ids", each half of a
  ## real one, and nothing would raise.
  case fmt
  of lfInt:
    let n = s.len div sizeof(int64)
    result = newSeqUninit[int](n)
    if n > 0:
      let src = cast[ptr UncheckedArray[int64]](s[0].unsafeAddr)
      for i in 0 ..< n:
        result[i] = int(src[i])
  of lfUint32:
    let n = s.len div sizeof(uint32)
    result = newSeqUninit[int](n)
    if n > 0:
      let src = cast[ptr UncheckedArray[uint32]](s[0].unsafeAddr)
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
  func simdDotF32(pa, pb: ptr UncheckedArray[float32], n: int): float32 =
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

  func simdDotI8(pa, pb: ptr UncheckedArray[int8], n: int): int32 =
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

func normalize*(v: var seq[float32]) =
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

func dot(a, b: openArray[float32]): float32 =
  ## The float32 distance kernel. `openArray`, not `seq`, so it can run on a
  ## `VecView` - a slice of the cache arena or of a `StoredVec` - without either
  ## side being copied: the SIMD body only ever needs the address and the length,
  ## and both are already there.
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

func quantizeScale*(vectors: openArray[seq[float32]]): float32 =
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

func quantizeInt8*(v: openArray[float32], scale: float32,
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

func quantizeInt8*(v: openArray[float32], scale: float32): seq[int8] =
  ## Scalar quantization of one vector on a given grid: one signed byte per
  ## component, `x ~= result[i] * scale`.
  ##
  ## Components above the grid (possible when the scale comes from a different
  ## vector) saturate at +-127 instead of wrapping. `scale <= 0` yields all-zero
  ## codes.
  var clamped = 0
  quantizeInt8(v, scale, clamped)

func quantizeInt8*(v: openArray[float32]): tuple[codes: seq[int8], scale: float32] =
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

func dequantizeInt8*(codes: openArray[int8], scale: float32): seq[float32] =
  ## Back to float32: `code * scale`. This is the lossy step; every component is
  ## off by at most `scale / 2`.
  result = newSeqUninit[float32](codes.len)
  for i, c in codes:
    result[i] = float32(c) * scale

func dotInt8*(a, b: openArray[int8], scaleA = 1.0'f32,
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

func packInt8*(xs: openArray[int8]): string =
  ## A dim-byte blob for YottaDB - a quarter of what `packFloats` writes for the
  ## same vector. The scale cannot be recovered from the bytes, so store it
  ## alongside (a second YottaDB node, or once in META if the whole collection
  ## shares one).
  result = newString(xs.len)
  if xs.len > 0:
    copyMem(result[0].addr, xs[0].unsafeAddr, xs.len)

func unpackInt8*(s: string): seq[int8] =
  ## Inverse of `packInt8`: one byte per dimension, `s.len` == dim.
  result = newSeqUninit[int8](s.len)
  if s.len > 0:
    copyMem(result[0].addr, s[0].unsafeAddr, s.len)


# ---------------------------------------------------------------------------
# vector storage
# ---------------------------------------------------------------------------

func packQVec(scale: float32, codes: openArray[int8]): string =
  ## A quantized vector as stored: the float32 scale first, then one byte per
  ## dimension, so a single YottaDB read gets both halves (4 + dim bytes).
  result = newString(sizeof(float32) + codes.len)
  copyMem(result[0].addr, scale.unsafeAddr, sizeof(float32))
  if codes.len > 0:
    copyMem(result[sizeof(float32)].addr, codes[0].unsafeAddr, codes.len)

func unpackQVec(s: string): StoredVec =
  ## Inverse of `packQVec`. An empty string - or anything without room for the
  ## scale - means there is no vector at that id.
  if s.len <= sizeof(float32):
    return
  var scale: float32
  copyMem(scale.addr, s[0].unsafeAddr, sizeof(float32))
  result.scale = scale
  result.codes = unpackInt8(s[sizeof(float32) .. ^1])

func isMissing(v: StoredVec): bool =
  ## True when the id has no vector: deleted, or never written.
  v.floats.len == 0 and v.codes.len == 0

func viewOf(v: StoredVec): VecView =
  ## Borrowed view of an owning `StoredVec`: the address of its codes or floats,
  ## and the scale that goes with the codes. Nothing is copied and nothing is
  ## owned - the seq has to outlive the view.
  if v.codes.len > 0:
    result.codes = cast[ptr UncheckedArray[int8]](v.codes[0].unsafeAddr)
    result.scale = v.scale
    result.dim = v.codes.len
  elif v.floats.len > 0:
    result.floats = cast[ptr UncheckedArray[float32]](v.floats[0].unsafeAddr)
    result.dim = v.floats.len

func dotCacheF32(q, v: ptr UncheckedArray[float32], n: int): float32 {.inline.} =
  ## The float32 kernel on two raw buffers: `dot` is still the only
  ## implementation, this is just the shape the walk wants, with the length
  ## already in hand.
  dot(toOpenArray(q, 0, n - 1), toOpenArray(v, 0, n - 1))

func dotCacheI8(q: ptr UncheckedArray[int8], qs: float32,
                v: ptr UncheckedArray[int8], vs: float32, n: int): float32 {.inline.} =
  ## The int8 kernel on two raw buffers, with both scales in hand.
  dotInt8(toOpenArray(q, 0, n - 1), toOpenArray(v, 0, n - 1), qs, vs)

func dotView(a, b: VecView): float32 {.inline.} =
  ## Dot product of two vectors held as views: the same `dot` / `dotInt8` kernels,
  ## run on bytes that are already in memory (the cache arena, or a `StoredVec`)
  ## instead of on a fresh copy of them.
  ##
  ## A missing vector, a length mismatch, or one side quantized where the other is
  ## not, scores 0 (= distance 1, the farthest a pair of L2-normalised vectors can
  ## be). `connect` runs over link lists that can still name a freed id, and that
  ## case must not raise.
  if a.dim == 0 or b.dim == 0 or a.dim != b.dim:
    return 0.0'f32
  if a.codes != nil and b.codes != nil:
    dotCacheI8(a.codes, a.scale, b.codes, b.scale, a.dim)
  elif a.floats != nil and b.floats != nil:
    dotCacheF32(a.floats, b.floats, a.dim)
  else:
    0.0'f32

func distanceView(a, b: VecView): float32 {.inline.} =
  ## Cosine distance between two L2-normalised stored vectors, both borrowed.
  1.0'f32 - dotView(a, b)

func cacheSlot(ix: HnswIndex, id: int): int =
  ## The slot holding `id`'s vector, or -1 when the cache has none for it: an id
  ## outside the index, one whose vector was never mirrored, or a deleted one.
  ##
  ## It sits here rather than with the rest of the cache because `loadVec` - the
  ## hot path - is the first thing that needs it.
  if id >= 0 and id < ix.cache.idToSlot.len:
    ix.cache.idToSlot[id]
  else:
    -1


proc loadVec*(ix: HnswIndex, id: int): StoredVec =
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

  # An id this index has already found empty - a deleted node reached through a
  # one-way link. Memoised per index, so it cannot answer for another index's id
  # space; see `absentIds`.
  if id in ix.absentIds:
    return
  let s = ydb_get(ix.params.globalNode, @[$id])
  if s.len == 0:
    ix.absentIds.incl(id)
    return
  case ix.params.quant
  of vqNone: result.floats = unpackFloats(s)
  of vqInt8: result = unpackQVec(s)
  of vqInt8Fixed:
    result.scale = ix.params.quantScale
    result.codes = unpackInt8(s)

proc vecViewAt(ix: HnswIndex, id: int, holder: var StoredVec): VecView =
  ## Borrowed view of `id`'s vector, for the paths that compare many vectors
  ## with each other instead of scoring one query against each of them.
  ##
  ## A cached vector is viewed where it lies. An id the cache does not hold -
  ## deleted, not mirrored, or no cache at all - is loaded into `holder`, so the
  ## caller has to keep `holder` alive for as long as it uses the view, and must
  ## not write a vector in between (a write can grow the arena). That is the same
  ## lifetime rule `VecView` carries everywhere else, and `selectNeighbors` meets
  ## it: it only reads.
  ##
  ## The decode is `loadVec`'s - same fields for the same mode, same scale - and a
  ## missing id yields a `dim == 0` view, which every distance treats as "no
  ## vector", exactly as `isMissing` did.
  let slot = ix.cacheSlot(id)
  if slot >= 0 and ix.cache.dim > 0:
    let base = slot * ix.cache.dim
    if ix.params.quant == vqNone:
      result.floats = cast[ptr UncheckedArray[float32]](
        ix.cache.floats[base].unsafeAddr)
    else:
      result.codes = cast[ptr UncheckedArray[int8]](
        ix.cache.codes[base].unsafeAddr)
      # vqInt8 carries a scale per vector, vqInt8Fixed one for the whole index.
      result.scale = if ix.params.quant == vqInt8: ix.cache.scales[slot]
                     else: ix.params.quantScale
    result.dim = ix.cache.dim
    return
  holder = ix.loadVec(id)
  result = viewOf(holder)

proc distanceAt(ix: HnswIndex, q: VecView, id: int, dist: var float32): bool {.inline.} =
  ## Distance from the query view `q` to node `id`, and whether `id` has a vector
  ## at all.
  ##
  ## The in-process cache is asked first, and a slot hit is the whole point of
  ## `VecView`: no `newSeqUninit`, no `copyMem`, no YottaDB call - the distance
  ## runs on the arena's own bytes. Only an id the cache does not hold falls
  ## through to `loadVec`, which is the three cases that need it: no cache at all,
  ## a vector another process wrote after we preloaded, and a deleted node (whose
  ## read comes back empty, which is how "missing" is recognised).
  ##
  ## Returns false when the id has no vector, so a caller skips it exactly as it
  ## did with `loadVec`'s `.isMissing`.
  ##
  ## This is the walk's per-candidate path, so the storage mode is not re-decided
  ## here. `q`'s form *is* the index's mode - `searchStored` builds it through
  ## `toStored` / `blobVector`, and `searchInt8` refuses an index that disagrees -
  ## and the arena a slot points into holds that same form. Reading it straight
  ## through the one branch below is what keeps the `case` on `quant`, and the
  ## `VecView` `cacheViewAt` used to build and throw away, out of the loop;
  ## `dotView` keeps its full checks for the callers that hand it two arbitrary
  ## vectors.
  ##
  ## The arena must not be resized while a returned distance is being used, which
  ## is exactly the window a walk runs in: a search writes no vector, so nothing
  ## reaches `cacheReserve` / `setLen` between the first candidate and the last.
  let slot = ix.cacheSlot(id)
  if slot >= 0:
    let base = slot * ix.cache.dim
    if q.codes != nil:
      # vqInt8 carries a scale per vector, vqInt8Fixed one for the whole index -
      # and only the former has `cache.scales` filled in.
      let vs = if ix.params.quant == vqInt8: ix.cache.scales[slot]
               else: ix.params.quantScale
      dist = 1.0'f32 - dotCacheI8(q.codes, q.scale,
               cast[ptr UncheckedArray[int8]](ix.cache.codes[base].unsafeAddr),
               vs, ix.cache.dim)
    else:
      dist = 1.0'f32 - dotCacheF32(q.floats,
               cast[ptr UncheckedArray[float32]](ix.cache.floats[base].unsafeAddr),
               ix.cache.dim)
    return true
  let ev = ix.loadVec(id)
  if ev.isMissing:
    return false
  dist = distanceView(q, viewOf(ev))
  true


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

# ---------------------------------------------------------------------------
# in-process link mirror
# ---------------------------------------------------------------------------
#
# `HnswParams.cacheLinks` keeps the neighbour lists this process has touched in
# process memory, so expanding a node costs neither a YottaDB read nor the two
# `$id` / `$level` subscript strings nor the `unpackLinks` copy that goes with
# them. It is filled on demand - `linksKnown` returning -1 means "ask YottaDB",
# which is also what keeps a node another process added correct - and it never
# has to be invalidated wholesale, because every write this process makes goes
# through `putLinks` or `linksForget`.

func linksIdx(ix: HnswIndex, id, level: int): int {.inline.} =
  id * ix.links.stride + level

func linksGrow(ix: HnswIndex, id, level: int) =
  ## Make the table cover `(id, level)`.
  ##
  ## Both dimensions grow *geometrically*, and that is the whole point of this
  ## procedure. `add` hands out a fresh id for every node, so a build reaches here
  ## once per insert; growing `ids` to "one more row" rebuilds all three tables -
  ## three `newSeq`s plus a copy of `ids * stride` entries - on every one of them,
  ## which is O(n^2) over the build and spends most of its life in the allocator
  ## re-growing a table that is about to be discarded. Doubling makes it amortised
  ## O(1) instead. Measured before this: 1559 us/insert at 10k nodes, 4804 us at
  ## 40k (4x the nodes for 3.1x the cost per insert).
  ##
  ## Only a change of `stride` has to re-index, because that is the one that moves
  ## existing entries - the row length is part of every index. It happens when a
  ## node appears on a layer above every one seen so far, which is logarithmic in
  ## `count`. A longer table is just a `setLen`: rows keep their place, and only
  ## the new ones have to be marked unknown.
  if id < ix.links.ids and level < ix.links.stride and ix.links.lens.len > 0:
    return

  if level >= ix.links.stride:
    let newStride = max(max(level + 1, ix.links.stride * 2), 1)
    let rows = ix.links.ids
    var lens = newSeq[int32](rows * newStride)
    var caps = newSeq[int32](rows * newStride)
    var offset = newSeq[int32](rows * newStride)
    for i in 0 ..< lens.len:
      lens[i] = -1
      offset[i] = -1
    for i in 0 ..< rows:
      for l in 0 ..< ix.links.stride:
        lens[i * newStride + l] = ix.links.lens[i * ix.links.stride + l]
        caps[i * newStride + l] = ix.links.caps[i * ix.links.stride + l]
        offset[i * newStride + l] = ix.links.offset[i * ix.links.stride + l]
    ix.links.lens = lens
    ix.links.caps = caps
    ix.links.offset = offset
    ix.links.stride = newStride

  if id >= ix.links.ids:
    let newIds = max(max(id + 1, ix.links.ids * 2), 64)
    let have = ix.links.ids * ix.links.stride
    ix.links.lens.setLen(newIds * ix.links.stride)
    ix.links.caps.setLen(newIds * ix.links.stride)
    ix.links.offset.setLen(newIds * ix.links.stride)
    for i in have ..< ix.links.lens.len:
      ix.links.lens[i] = -1
      ix.links.offset[i] = -1
    ix.links.ids = newIds

func linksKnown(ix: HnswIndex, id, level: int): int32 {.inline.} =
  ## What the mirror knows about `(id, level)`: -1 not mirrored, 0 mirrored and
  ## empty, n > 0 mirrored with n ids.
  if not ix.params.cacheLinks or id < 0 or level < 0 or
     id >= ix.links.ids or level >= ix.links.stride:
    return -1
  ix.links.lens[linksIdx(ix, id, level)]

func linksStore(ix: HnswIndex, id, level: int, links: openArray[int]) =
  ## Record what `(id, level)` holds. Both callers pass the truth - `putLinks`
  ## after a successful write, the fill path with whatever YottaDB returned - so
  ## an empty list is recorded as empty rather than erased: "no links" is a real
  ## answer and must not be read back.
  if not ix.params.cacheLinks or id < 0 or level < 0:
    return
  ix.linksGrow(id, level)
  let i = linksIdx(ix, id, level)
  if links.len == 0:
    ix.links.lens[i] = 0
    return
  if ix.links.offset[i] < 0 or links.len > int(ix.links.caps[i]):
    # First write for this entry - or, only for a list the index would never
    # write itself, one longer than the room reserved, which then moves.
    let cap = max(2 * ix.params.M, links.len)
    ix.links.offset[i] = int32(ix.links.arena.len)
    ix.links.caps[i] = int32(cap)
    ix.links.arena.setLen(ix.links.arena.len + cap)
  ix.links.lens[i] = int32(links.len)
  copyMem(ix.links.arena[int(ix.links.offset[i])].addr, links[0].unsafeAddr,
          links.len * sizeof(int))

func linksForget(ix: HnswIndex, id, level: int) =
  ## `(id, level)` has no list any more - `unlink` deleting the node it emptied.
  ## The slot is kept and recorded empty, so a later read of it does not go back
  ## to YottaDB to be told the same thing.
  if not ix.params.cacheLinks or id < 0 or level < 0:
    return
  if id < ix.links.ids and level < ix.links.stride:
    ix.links.lens[linksIdx(ix, id, level)] = 0

func linksForgetAll(ix: HnswIndex, id: int) =
  ## Drop every layer of `id`: the mirror's side of deleting the node's
  ## `^...LINKS` subtree.
  if not ix.params.cacheLinks or id < 0 or id >= ix.links.ids:
    return
  for l in 0 ..< ix.links.stride:
    ix.links.lens[linksIdx(ix, id, l)] = 0

proc ydbLinks(ix: HnswIndex, id, level: int): seq[int] =
  ## One neighbour list straight from YottaDB - the read the mirror saves.
  unpackLinks(ix.params.linkFmt, ydb_get(ix.params.globalLinks, @[$id, $level]))

proc putLinks(ix: HnswIndex, id, level: int, links: openArray[int]) =
  if links.len > 0:
    ydb_set(ix.params.globalLinks, @[$id, $level],
            packLinks(ix.params.linkFmt, links))
    ix.linksStore(id, level, links)

proc linksAt(ix: HnswIndex, id, level: int,
             spare: var seq[int]): tuple[p: ptr UncheckedArray[int], n: int] =
  ## The neighbour list of `(id, level)` as a borrowed slice, for the walk.
  ##
  ## The mirror is asked first; a subscript it does not know is read from YottaDB
  ## once and then remembered, empty or not. `spare` backs that one read - the
  ## caller owns the buffer and reuses it across calls - so once the mirror is
  ## warm a walk allocates nothing here at all.
  ##
  ## `n` is the number of ids and `p` points at the first of them; `n == 0` means
  ## the list is empty and `p` must not be dereferenced. What `p` points at stays
  ## valid until the next call that *writes* a link list, which a search never
  ## makes.
  if ix.params.cacheLinks:
    if ix.linksKnown(id, level) < 0:
      spare = ix.ydbLinks(id, level)
      ix.linksStore(id, level, spare)
    let n = int(ix.linksKnown(id, level))
    if n <= 0:
      return (nil, 0)
    let off = int(ix.links.offset[linksIdx(ix, id, level)])
    return (cast[ptr UncheckedArray[int]](ix.links.arena[off].unsafeAddr), n)
  spare = ix.ydbLinks(id, level)
  if spare.len == 0:
    return (nil, 0)
  (cast[ptr UncheckedArray[int]](spare[0].unsafeAddr), spare.len)

proc getLinks(ix: HnswIndex, id, level: int): seq[int] =
  ## One neighbour list as a fresh seq: what the paths that go on to *change* it
  ## need (`connect` pruning an overflowing list, `unlink` removing an id). The
  ## walk does not use this - it borrows the list where it lies, through
  ## `linksAt`.
  var spare: seq[int]
  let (p, n) = ix.linksAt(id, level, spare)
  if n == 0:
    return @[]
  result = newSeqUninit[int](n)
  copyMem(result[0].addr, p, n * sizeof(int))

func refreshLinks*(ix: HnswIndex) =
  ## Drop every mirrored neighbour list, so the next read of each comes from
  ## YottaDB again - what a long-running reader does once a *second* process has
  ## relinked nodes in the same index.
  ##
  ## The mirror is written only by this process's own `putLinks` / `linksForget`,
  ## which is exactly why it cannot see another writer: it is a copy of
  ## `^...LINKS`, and `^...LINKS` carries no version to compare against. A process
  ## that deliberately writes those globals itself (a migration, or a test) wants
  ## `cacheLinks = false`; one that merely shares the index with another process
  ## wants this. It costs nothing but the refill, one read per list actually
  ## walked. `preloadVectors` is the same idea for the vectors.
  ix.links = LinkStore()

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

proc bumpLive(ix: HnswIndex, delta: int) =
  ## Advance the shared `live` counter by `delta`, mirroring the result in
  ## `ix.live`.
  ##
  ## `$INCREMENT` is atomic across processes; `inc ix.live; metaSet("live", ...)`
  ## is not. With two writers each keeping its own copy, that read-modify-write
  ## loses one update in two, so the recorded `live` drifts far below the number
  ## of nodes that are really there - which is exactly the "122901 vector(s) of
  ## 349950 id(s)" report, where the preload sweep then counted `count` vectors
  ## and proved every id was present. `count` was already made atomic for the
  ## same reason; this is its counterpart.
  ##
  ## An index built before `live` existed has no META key, and `$INCREMENT` on a
  ## missing node starts from 0 - so seed it from what `openHnsw` worked out
  ## (`live` == `count` in that case) before the first bump.
  if ydb_data(ix.params.globalMeta, @["live"]) == 0:
    ix.metaSet("live", $ix.live)
  ix.live = ydb_increment(ix.params.globalMeta, @["live"], delta)

func maxLinks*(ix: HnswIndex, level: int): int =
  ## Level 0 keeps twice as many links - that is where the whole graph gets
  ## walked, so it pays for itself there.
  if level == 0: 2 * ix.params.M else: ix.params.M



# ---------------------------------------------------------------------------
# in-process vector cache
# ---------------------------------------------------------------------------

func cacheOn(ix: HnswIndex): bool =
  ## Whether this index mirrors its vectors in process memory - the `cacheVectors`
  ## setting, not "the id map has entries".
  ##
  ## The distinction matters for an index opened while still empty: it has a
  ## cache, it is merely empty, and the first `add` has to be able to fill it.
  ## Keying this on `idToSlot.len` is what made a from-scratch build run its
  ## entire first pass with the cache silently off (`cachePut` returning early)
  ## and only start caching after a reopen.
  ix.params.cacheVectors

func cacheInit(ix: HnswIndex) =
  ## Bring the cache arena in line with the index's dimension.
  ##
  ## For an index opened with vectors the two already agree: `preloadVectors`
  ## sized the arena from `ix.dim`, which came from META. The one case they do not
  ## is an index that was *empty* when opened - `preloadVectors` had nothing to
  ## size itself from and returned - and is being filled now. Without this the
  ## first mirror write would use an arena `dim` of 0: `cacheReserve` would
  ## allocate nothing and every vector would be copied to offset 0.
  if ix.params.cacheVectors and ix.dim > 0 and ix.cache.dim != ix.dim:
    ix.cache.dim = ix.dim

func cacheIds(ix: HnswIndex, count: int) =
  ## Make the id -> slot map cover every id below `count`. This is the one part
  ## still keyed by id and so the one part that grows with `count` - which is why
  ## it is a plain int seq and not a second arena.
  if count <= ix.cache.idToSlot.len:
    return
  let have = ix.cache.idToSlot.len
  ix.cache.idToSlot.setLen(count)
  for i in have ..< count:
    ix.cache.idToSlot[i] = -1

func cacheReserve(ix: HnswIndex, slots: int) =
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

func cachePut(ix: HnswIndex, id: int, v: StoredVec) =
  ## Mirror `v` into the cache. Called from `putVec`, so every write this index
  ## makes keeps the cache coherent and an insert never invalidates it.
  if not ix.cacheOn:
    return
  
  ix.cacheInit()
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

func cacheFree(ix: HnswIndex, id: int) =
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


proc cacheVector(ix: HnswIndex, id: int): bool =
  ## Mirror the vector YottaDB holds at `id` into the cache, returning whether
  ## one was loaded.
  ##
  ## Safe to call for any id, in any order, and idempotently: it grows `idToSlot`
  ## to cover `id` and leaves an id that is already mirrored alone. Both matter
  ## once a second process writes the same index - the ids it reserved arrive out
  ## of step with our own `count`, and an id can already be present from our own
  ## `putVec`. Writing `idToSlot[id]` without the growth first is exactly the
  ## IndexDefect that a refresh past our own `count` used to hit.
  ##
  ## An id that is reserved but not written yet (the other process is mid-`add`)
  ## yields false and takes no slot; `loadVec` falls back to YottaDB for it and a
  ## later sweep picks it up.
  if not ix.params.cacheVectors or id < 0:
    return false
  ix.cacheInit()
  if id >= ix.cache.idToSlot.len:
    ix.cacheIds(id + 1)
  if ix.cache.idToSlot[id] >= 0:
    return false
  let s = ydb_get(ix.params.globalNode, @[$id])
  if s.len == 0:
    return false
  # Validate the blob against the mode before committing a slot, so a malformed
  # value cannot leave a slot with no owner.
  case ix.params.quant
  of vqNone:
    if s.len != ix.dim * sizeof(float32): return false
  of vqInt8:
    if s.len != sizeof(float32) + ix.dim: return false
  of vqInt8Fixed:
    if s.len != ix.dim: return false
  let slot = ix.cache.nSlots
  ix.cacheReserve(slot + 1)
  case ix.params.quant
  of vqNone:
    copyMem(ix.cache.floats[slot * ix.dim].addr, s[0].unsafeAddr, s.len)
  of vqInt8:
    copyMem(ix.cache.scales[slot].addr, s[0].unsafeAddr, sizeof(float32))
    copyMem(ix.cache.codes[slot * ix.dim].addr, s[sizeof(float32)].unsafeAddr, ix.dim)
  of vqInt8Fixed:
    copyMem(ix.cache.codes[slot * ix.dim].addr, s[0].unsafeAddr, ix.dim)
  ix.cache.idToSlot[id] = slot
  ix.cache.slotToId[slot] = id
  ix.cache.nSlots = slot + 1
  # The id is there now; if an earlier search memoised it as absent (it was
  # reserved but not written then), that verdict is no longer true.
  ix.absentIds.excl(id)
  true

proc preloadLinks(ix: HnswIndex): int =
    if ix.dim <= 0 or ix.count == 0:
      return 0

    echo "Preloading up to ", ix.live, " links of ", ix.count, " id(s)"
    let t1 = now()

    let gbl = ix.params.globalLinks
    for keys in QueryItr @gbl.keys:
        let id = parseInt(keys[0])
        let level = parseInt(keys[1])
        let links = unpackLinks(ix.params.linkFmt, ydb_get(ix.params.globalLinks, @[$id, $level]))
        ix.linksStore(id, level, links)
        inc result

    echo "'Links' cache loaded with ", result, " entries in ", $(now() - t1).inMilliseconds, " ms."
    return result

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
  let t1 = now()
  for id in 0 ..< ix.count:
    if ix.cacheVector(id):
      inc result
  ix.cache.complete = true
  # The sweep just counted exactly how many ids carry a vector - which is what
  # `live` is meant to mean. Put the shared counter right if an earlier run's
  # racy read-modify-write had let it drift; a consistent index is left alone.
  if ix.live != result:
    ix.live = result
    ix.metaSet("live", $result)
  echo "'Vector' cache loaded with ", result, " entries in ", $(now() - t1).inMilliseconds, " ms."

func cachedVectors*(ix: HnswIndex): int =
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

func toStored(ix: HnswIndex, v: seq[float32]): StoredVec =
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
# lifecycle
# ---------------------------------------------------------------------------
var modelLoader: ModelLoader = nil

proc setModelLoader*(loader: ModelLoader) =
  ## Install the way `openHnsw` loads the model named in `HnswParams.model` - see
  ## `ModelLoader`. `bert_nim.loadModel` is what the embedding side registers.
  modelLoader = loader

func deriveGlobalNames(p: var HnswParams) =
  ## Fill in the `^...NODE` / `^...KEY` / `^...META` names that belong to
  ## `p.global`, unless a caller named them itself.
  if p.globalNode.len == 0:
    p.globalNode = p.global & "NODE"
    p.globalLinks = p.global & "LINKS"
    p.globalLevel = p.global & "LEVEL"
    p.globalKey = p.global & "KEY"
    p.globalMeta = p.global & "META"
    p.globalQueue = p.global & "QUEUE"

func hnswParams*(global: string, model = DefaultModel,
                 M = 16, efConstruction = 200, efSearch = 64, seed = 1234,
                 dim = 0, quant = vqNone, quantScale = 0.0'f32,
                 linkFmt = lfUInt32,
                 cacheVectors = true, cacheLinks = true,
                 batchSize = 128): HnswParams =
  ## Configure one index: `global` is the YottaDB global (`^HNSWxxx`) and the
  ## `NODE` / `KEY` / `META` names are derived from it, so no caller spells the
  ## layout out a second time.
  ##
  ## `model`, `dim`, `quant`, `quantScale` and `linkFmt` only have to be right
  ## the first time; an index that already exists keeps what META says (see
  ## `openHnsw`) - which is what keeps a reopen from reinterpreting blobs that
  ## were written in the other link width.
  ##
  ## `cacheVectors` and `cacheLinks` are not stored in META and are honoured on
  ## every open, so they are the two things here that always take effect: they say
  ## what the process mirrors in memory (`preloadVectors`, `linksAt`), not what
  ## the index is.
  result = HnswParams(global: global, model: model, M: M,
                      efConstruction: efConstruction, efSearch: efSearch,
                      seed: seed, dim: dim, quant: quant, quantScale: quantScale,
                      linkFmt: linkFmt,
                      cacheVectors: cacheVectors, cacheLinks: cacheLinks,
                      batchSize: batchSize)
  deriveGlobalNames(result)

proc inferDim(ix: HnswIndex): int =
  ## The width an index's own blobs imply.
  ##
  ## `META("dim")` is written when the index is created, from whatever the
  ## *creating* process was told - and every builder in this repo opens without a
  ## `dim` (the queue processor, `lj`, `hnsw_search`), so 0 goes in. `add` settles
  ## the real dimension from the first vector and, until this, kept it to itself.
  ## What is left behind is an index that can hold a million correctly shaped
  ## vectors and still not know how wide they are, which is not a state anything
  ## downstream can use: `bytesPerVector` comes out 0 or 4, `preloadVectors`
  ## returns immediately (so `cacheVectors` silently does nothing) and
  ## `blobVector` rejects the very blobs the index wrote.
  ##
  ## The blobs themselves say it. `bytesPerVector` is a function of the mode and
  ## the dimension and nothing else, so it inverts exactly - there is nothing to
  ## guess: a `vqNone` blob is a whole number of float32, a `vqInt8` blob is those
  ## codes behind one float32 scale, a `vqInt8Fixed` blob is the codes alone.
  ##
  ## The width has to be the *majority* width and not merely the first one read:
  ## this is written back to the index, and an index that is mid-rebuild, or that
  ## has a stray value in it, must not be able to talk an open into a dimension
  ## its vectors do not have - a wrong one is worse than none, because it is then
  ## believed by every later open. Three quarters of the blobs have to agree, and
  ## a mixed index is refused, leaving the explicit "expects 0 at dim 0" error in
  ## place. (A first attempt at this did believe a two-blob reading of 101-byte
  ## values in an index whose 742,850 vectors are all 388 bytes and wrote dim 97
  ## into META, which is what the majority rule is here to prevent.)
  ##
  ## Only the low 64 ids are read: ids are handed out in order, so node 0 - or the
  ## hole just after it - is where the first vector is, and 64 reads is nothing
  ## next to the `preloadVectors` this unblocks.
  ##
  ## Call this only once `ix.params.quant` is the *index's* mode and not the
  ## caller's: the mode is what says where the dimension sits in a blob, and the
  ## two disagree by the scale - `vqNone` over a `vqInt8` blob divides by four. A
  ## wrong dimension from the wrong mode is written to META and then believed by
  ## every later open, which is worse than no dimension at all.
  if ix.count <= 0:
    return 0
  var lens: seq[int]
  for id in 0 ..< min(ix.count, 64):
    let s = ydb_get(ix.params.globalNode, @[$id])
    if s.len > 0:
      lens.add s.len        # a hole is a reserved-but-unwritten or deleted id
  if lens.len < 2:
    return 0
  var want = 0              # the width most of them have
  var best = 0
  for n in lens:
    var c = 0
    for m in lens:
      if m == n: inc c
    if c > best:
      best = c
      want = n
  if best * 4 < lens.len * 3:
    echo "NODE blobs have mixed widths (", lens.len, " read, ", best, " agree); leaving META without a dimension"
    return 0
  case ix.params.quant
  of vqNone:
    if want mod sizeof(float32) == 0: want div sizeof(float32) else: 0
  of vqInt8:
    if want > sizeof(float32): want - sizeof(float32) else: 0
  of vqInt8Fixed:
    want

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
  result.absentIds = initHashSet[int]()

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

  # The dimension is the one piece of the index META can be missing outright, and
  # it is only recoverable *here*, below the storage mode: the mode is what says
  # where the dimension sits inside a blob, so inferring it while `quant` is still
  # the caller's default reads a `vqInt8` blob as float32 and divides by four.
  if result.dim <= 0 and result.count > 0:
    # No usable dimension in META, but vectors that have one. Recover it instead
    # of opening an index that cannot read its own blobs, and record it: the 0
    # that is there is the dimension the creator was *asked* for, and every later
    # open - this process or another - would otherwise start from 0 again. See
    # `inferDim`. Idempotent, so a reader racing a builder writes the same value
    # the builder would.
    result.dim = result.inferDim()
    if result.dim > 0:
      result.metaSet("dim", $result.dim)
      echo "META records no dimension; ", result.dim, " inferred from the stored vectors"
  # `params.dim` is the dimension this process works with - META's, or the one
  # just recovered. It is what `$ix.params` reports and what the init block below
  # writes, so leaving the caller's 0 in it described the index wrongly.
  result.params.dim = result.dim

  # The link width is the same kind of thing as the storage mode: it says how the
  # blobs in LINKS decode, so it belongs to the index and META wins over `p`.
  # A non-empty index without the key was written before the width was
  # configurable, i.e. as int64.
  let storedLinkFmt = result.metaGetName("linkfmt", "")
  if storedLinkFmt.len > 0:
    result.params.linkFmt = parseEnum[LinkFmt](storedLinkFmt, lfInt)
  elif result.count > 0:
    result.params.linkFmt = lfInt
  else:
    result.params.linkFmt = p.linkFmt

  if result.params.quant == vqInt8Fixed and result.params.quantScale <= 0:
    raise newException(ValueError,
      "vqInt8Fixed needs a positive quantScale, e.g. quantizeScale(sample)")

  # The embedding model belongs to the index's identity, like its dimension and
  # its storage mode: a vector is only comparable to one from the same model, so
  # META wins over `p` and a reopen loads the model the vectors were built with.
  # The load happens here, at the end, so a config error above fails before the
  # weights are pulled in. Without a loader (see `setModelLoader`) the name is
  # carried along but nothing is loaded - the tests never import Python.
  if result.params.model.cmpIgnoreCase("NONE") != 0:
      result.params.model = result.metaGetName("model", p.model)
      if result.params.model.len > 0 and not modelLoader.isNil:
        # The loader reports back the name it resolved to (a short id, a local path,
        # the default) - that resolved name is what the index is described by.
        result.params.model = modelLoader(result.params.model)

  if result.params.verbose:
    echo result.params
    
  # The vector cache is not part of an index's identity the way M, dim and the
  # storage mode are - it mirrors what is already there and can be thrown away -
  # so it is the one thing that comes from `p` and not from META. Reading the
  # vectors back is O(count) round trips, hence the explicit opt-in.
  if result.params.cacheVectors and result.params.preloadCacheVectors:
    discard result.preloadVectors()
  if result.params.cacheLinks and result.params.preloadCacheLinks:
    discard result.preloadLinks()
  

  let gbl = result.params.globalMeta
  let p = result.params
  if ydb_data(gbl, @["M"]) == 0:  # initialize only once
    ydb_set(gbl, @["M"], $p.M)
    ydb_set(gbl, @["efConstruction"], $p.efConstruction)
    ydb_set(gbl, @["dim"], $p.dim)
    ydb_set(gbl, @["quant"], $p.quant)
    if p.quant == vqInt8Fixed:
        ydb_set(gbl, @["qscale"], $p.quantScale)
    ydb_set(gbl, @["linkfmt"], $p.linkFmt)    
    if p.model.len > 0:
        ydb_set(gbl, @["model"], p.model)    



proc recordLinkFmt*(ix: HnswIndex, fmt: LinkFmt) =
  ## Declare the width of the neighbour lists that are already in the database,
  ## recording it in META and in `ix` - for an index whose `^...LINKS` blobs were
  ## converted in place (every value rewritten from int64 to uint32) and which
  ## therefore has nothing in META to say so.
  ##
  ## This writes the key and nothing else: it does not convert one blob. The
  ## index has to *be* in `fmt` already, because reading a uint32 list as `lfInt`
  ## does not raise - it returns half as many ids, each one a pair of real ones,
  ## and the graph quietly stops being the one that was built. Ordinary indexes
  ## never need this: `add` pins the width with the first node.
  ix.params.linkFmt = fmt
  ix.metaSet("linkfmt", $fmt)
  # Every list in the mirror was decoded with the old width, so drop it rather
  # than answer from blobs that have just been reinterpreted underneath it.
  ix.links = LinkStore()


proc hasId*(ix: HnswIndex, id: int): bool =
  ## Whether a live node exists at `id`. One YottaDB `data` call, no value read -
  ## and no call at all once `preloadVectors` has made the cache complete.
  if ix.cache.complete and id >= 0 and id < ix.cache.idToSlot.len:
    return ix.cache.idToSlot[id] >= 0
  ydb_data(ix.params.globalNode, @[$id]) != 0

func liveCount*(ix: HnswIndex): int =
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

func randomLevel(ix: HnswIndex): int =
  ## Exponentially decaying level draw: most nodes land on layer 0, a few get
  ## promoted, which is what gives the structure its log-time descent.
  let mL = 1.0 / ln(ix.params.M.float)
  # rand(r, 1.0) is uniform in [0, 1); guard the log against an exact zero.
  let u = max(rand(ix.rng, 1.0), 1e-12)
  int(floor(-ln(u) * mL))

func seenOrMark(ix: HnswIndex, over: var seq[int], e: int, epoch: int32): bool {.inline.} =
  ## True when node `e` is already visited under `epoch`; marks it and returns
  ## false otherwise.
  ##
  ## Node ids are dense integers `0 ..< count`, so `visitStamp[e] == epoch`
  ## answers "seen this node?" in one array load. The HashSet this replaces
  ## (`hashWangYi1` + `rawGet` + `containsOrIncl`) was ~25% of a build profile -
  ## hashing a value that is already an index. A fresh search just bumps the
  ## epoch instead of clearing the array, so it is O(1), not O(count).
  ##
  ## An id at or past our `count` is a node another process allocated after we
  ## opened (a link list can already name it). Those are rare and few, so they go
  ## into a linear-scan `over` list rather than growing the array to their id.
  if e >= 0 and e < ix.visitedStamp.len:
    if ix.visitedStamp[e] == epoch: return true
    ix.visitedStamp[e] = epoch
    return false
  for x in over:
    if x == e: return true
  over.add e
  false

proc searchLayer(ix: HnswIndex, q: StoredVec, entryPoints: openArray[int],
                 ef: int, level: int): NeighborView =
  ## Greedy best-first search on one layer, against the *stored* vectors (codes
  ## for a quantized index). Returns the `ef` closest nodes found, nearest first,
  ## as a view into `ix.scratch.found` - valid until the next call.
  ##
  ## Candidates and their neighbour lists are both read through the in-process
  ## mirrors (`distanceAt`, `linksAt`), so a vector or a link list they hold is
  ## used where it already sits - no per-candidate allocation, no copy, no YottaDB
  ## read. The query is borrowed into a view once, before the loop, because it is
  ## the same vector for every comparison this call makes.
  ##
  ## The heaps, the two small buffers and the result buffer are the index's
  ## scratch (`SearchScratch`), reset here instead of allocated: a query runs this
  ## once per layer and a build once per layer per insert, and the four seqs plus
  ## their growth were pure churn. Resetting is `n = 0` because `n` is the only
  ## thing a heap reads, and everything below it is written before it is read.
  if ix.visitedStamp.len < ix.count:
    ix.visitedStamp.setLen(ix.count)   # new slots read 0, every epoch is >= 1
  inc ix.visitEpoch
  if ix.visitEpoch <= 0:               # wrapped after ~2^31 searches: start over
    for i in 0 ..< ix.visitedStamp.len: ix.visitedStamp[i] = 0
    ix.visitEpoch = 1
  let epoch = ix.visitEpoch

  # The query is the same vector for every comparison in this call, so its view
  # is taken once, here, rather than per candidate.
  let qv = viewOf(q)

  ix.scratch.candidates.n = 0
  ix.scratch.results.n = 0
  ix.scratch.over.setLen(0)

  var d: float32
  for ep in entryPoints:
    if ep < 0 or ep >= ix.count or ix.seenOrMark(ix.scratch.over, ep, epoch):
      continue
    if not ix.distanceAt(qv, ep, d):
      continue                        # deleted id (e.g. a stale entry point)
    ix.scratch.candidates.heapPush(d, ep)
    ix.scratch.results.heapPush(-d, ep)

  while ix.scratch.candidates.n > 0:
    let c = ix.scratch.candidates.heapPop()
    # Everything left is farther than the worst result we keep -> done.
    if ix.scratch.results.n >= ef and c.dist > ix.scratch.results.worst:
      break
    let (lp, ln) = ix.linksAt(c.id, level, ix.scratch.spare)
    for k in 0 ..< ln:
      let e = lp[k]
      if ix.seenOrMark(ix.scratch.over, e, epoch):
        continue
      if not ix.distanceAt(qv, e, d):
        # Freed id, reached through a link that neighbour pruning left one-way.
        # Treating it as "skip" keeps deleted nodes out of the result set.
        continue
      if ix.scratch.results.n < ef or d < ix.scratch.results.worst:
        ix.scratch.candidates.heapPush(d, e)
        ix.scratch.results.heapPush(-d, e)
        if ix.scratch.results.n > ef:
          # min-heap on -dist: drops the farthest
          discard ix.scratch.results.heapPop()
  # The heap pops farthest first; fill from the end so the caller gets ascending
  # order without a separate reverse pass.
  let n = ix.scratch.results.n
  ix.scratch.found.setLen(n)
  if n == 0:
    return NeighborView(p: nil, n: 0)
  var i = n - 1
  while ix.scratch.results.n > 0:
    let r = ix.scratch.results.heapPop()
    ix.scratch.found[i] = (dist: -r.dist, id: r.id)
    dec i
  NeighborView(p: cast[ptr UncheckedArray[Neighbor]](ix.scratch.found[0].addr), n: n)

proc selectNeighbors(ix: HnswIndex, candidates: seq[Neighbor], m: int,
                     provided: openArray[VecView] = [],
                     keepPruned = false): seq[int] =
  ## Malkov & Yashunin Algorithm 4, run on stored vectors. Candidates are
  ## accepted only when they are closer to the base vector than to every
  ## neighbour already accepted; this spreads the links in different directions
  ## instead of clustering them all in one spot. The distances to the base are
  ## already in `candidates`, so the heuristic only needs the vectors to compare
  ## candidates with each other.
  ##
  ## The candidates are examined in increasing distance and their vectors are
  ## borrowed as views, one at a time: a cached vector is compared where it lies
  ## (no allocation, no copy), and only the candidates the heuristic actually
  ## reaches are loaded at all - it stops as soon as `m` have been accepted,
  ## which is usually well before the `efConstruction`-wide list is used up. The
  ## old version loaded every candidate up front, as an owning `StoredVec`.
  ##
  ## `provided`, when it holds one vector per candidate *in candidate order*, is
  ## used instead of loading them: `connect`'s pruning path has already loaded
  ## the list to compute the distances to `n`. The caller keeps those views - and
  ## whatever they point at - alive for the duration of the call.
  ##
  ## Borrowing the bytes is safe for the same reason it is in `searchLayer`: this
  ## is a read-only pass, so nothing reaches `cacheReserve` / `setLen` while the
  ## views are in use.
  var order = newSeq[int](candidates.len)
  for i in 0 ..< order.len:
    order[i] = i
  # Sorting positions rather than the candidates keeps the vectors aligned with
  # them. `sort` is stable, so candidates at equal distance keep the order they
  # arrived in - the same order the old in-place sort produced.
  order.sort(proc(a, b: int): int = cmp(candidates[a].dist, candidates[b].dist))

  let haveProvided = provided.len == candidates.len
  var views = newSeq[VecView](order.len)
  var owned: seq[StoredVec]
  if haveProvided:
    for i in 0 ..< order.len:
      views[i] = provided[order[i]]
  else:
    # One allocation holding the fallback loads, instead of one per candidate.
    owned = newSeq[StoredVec](order.len)

  var chosen: seq[int]
  var pruned: seq[int]
  for i in 0 ..< order.len:
    if chosen.len >= m:
      break
    let ci = order[i]
    if not haveProvided:
      views[i] = ix.vecViewAt(candidates[ci].id, owned[i])
    var keep = true
    for j in chosen:
      if distanceView(views[i], views[j]) < candidates[ci].dist:
        keep = false
        break
    if keep:
      chosen.add i
    else:
      pruned.add i

  for i in chosen:
    result.add candidates[order[i]].id
  if keepPruned:
    # Fill up with the discarded ones rather than returning fewer than `m`.
    for i in pruned:
      if result.len >= m:
        break
      result.add candidates[order[i]].id

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
      # Re-run the heuristic from n's point of view. Each candidate vector is
      # loaded once - the distance to `n` needs it, and so does the selection -
      # and `n` itself is a view rather than a copy. The old version loaded `n`
      # plus every candidate twice more inside `selectNeighbors`.
      var nHolder: StoredVec
      let base = ix.vecViewAt(n, nHolder)
      var owned = newSeq[StoredVec](nLinks.len)
      var views = newSeq[VecView](nLinks.len)
      var cands = newSeq[Neighbor](nLinks.len)
      for i in 0 ..< nLinks.len:
        views[i] = ix.vecViewAt(nLinks[i], owned[i])
        cands[i] = (dist: distanceView(base, views[i]), id: nLinks[i])
      nLinks = ix.selectNeighbors(cands, limit, views)
    ix.putLinks(n, level, nLinks)

  ep = w[0].id

proc add*(ix: HnswIndex, vec: openArray[float32], key = ""): int =
  ## Insert one vector as a new node and link it into the graph, returning its
  ## id. The id is reserved atomically from META("count"), so `add` is safe to
  ## call from several processes that share the same index.
  ##
  doAssert key.len > 0, "add: empty key"

  ## Passing `key` records the key -> node id mapping used by `lookup`.
  ## `vec` is copied before it is normalised, so the caller's seq is untouched.
  ##
  ## The id comes from an atomic `$INCREMENT` on META("count"), never from our own
  ## `ix.count`: several processes can be adding to the same index, each with its
  ## own copy of it, and two of them reserving from a stale local count would
  ## hand out the same id and overwrite each other's node. `count` is "next free
  ## id", so the id we own is the incremented value minus one; `ix.count` is then
  ## pulled up to it so the rest of this call sees a coherent id space.
  let lastCount = ix.count
  let id = ydb_increment(ix.params.globalMeta, @["count"]) - 1
  if id + 1 > ix.count:
    ix.count = id + 1
  if id > lastCount and ix.params.cacheVectors:
    # Somebody else reserved ids we have not mirrored, so the store no longer
    # covers the whole index: `hasId` has to ask YottaDB from here on. Only
    # `preloadVectors` sets this back to true.
    ix.cache.complete = false

  # Under `lfUint32` the id has to fit in a link. Ids are handed out in
  # increasing order and never reused, so this one check covers every later link
  # write as well: a neighbour's id is always below the id being inserted.
  if ix.params.linkFmt == lfUint32 and uint64(id) > uint64(high(uint32)):
    raise newException(ValueError,
      "id " & $id & " does not fit linkFmt = lfUint32; reopen the index with " &
      "lfInt, or rebuild it into a new global")

  var v = @vec
  if ix.dim == 0:
    # The first vector defines the index's dimension - a caller's `dim` is only a
    # promise, and nothing checks it against the data. This is also the one moment
    # the dimension is knowable, so META is told here: the create path in
    # `openHnsw` can only record what the caller *asked* for, which is 0 for every
    # builder that passes no `dim`, and an index whose dimension is not in META
    # cannot be read back at all (see `inferDim`). Once per process: the branch
    # needs `ix.dim` to still be 0.
    ix.dim = v.len
    if ix.dim > 0 and ix.metaGet("dim", 0) <= 0:
      ix.metaSet("dim", $ix.dim)
  doAssert v.len == ix.dim, "vector has " & $v.len & " dims, index expects " & $ix.dim
  normalize(v)

  let level = ix.randomLevel()

  # Encode once, store that, and build the graph from it - the neighbours are
  # then the ones the stored codes actually have. `v` is L2-normalised by now,
  # which is what makes the quantization grid meaningful.
  let sv = ix.toStored(v)

  # The geometry is pinned in META by `openHnsw`, and only while the index is
  # being created: the graph is valid only for the M and efConstruction it was
  # built with, the storage mode decides how every vector blob in it decodes, and
  # the model decides what the vectors mean - so none of them may change on a
  # reopen. The dimension is the one piece that is not known at creation, so
  # `add` records it when it settles it, above.

  # The node's own data first: later steps look their neighbours up by id.
  ix.putVec(id, sv)
  ix.putLevel(id, level)
  ix.putKey(key, id)

  # `count` is the id allocator: it only ever grows, so ids freed by delete()
  # stay holes and are never handed out again. It was already advanced by the
  # `$INCREMENT` above, i.e. in the shared global rather than in a private copy.
  # `live` is bumped atomically too - see `bumpLive`.
  ix.bumpLive(1)

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

  # The gap `lastCount ..< id` is ids other processes reserved between our last
  # add and this one. Their nodes are reachable through links and `loadVec` falls
  # back to YottaDB for any it misses, but that read is exactly what the cache is
  # there to avoid - so mirror them now. Our own id is already mirrored, by the
  # `putVec` above, which is why the range stops before `id`.
  #
  # A reserved id whose vector is not written yet (the other process is still
  # inside its own `add`) loads nothing; that leaves the cache short of vectors,
  # which is why `complete` stays false and `hasId` keeps asking YottaDB instead
  # of trusting a table with holes in it.
  if ix.params.cacheVectors and lastCount != id:
    for i in lastCount ..< id:
        discard ix.cacheVector(i)

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
        ix.linksForget(n, l)
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
  ix.linksForgetAll(id)
  ydb_delete(ix.params.globalNode, @[$id], YDB_DEL_NODE)

  # The id is gone from the cache too. The store is dense, so this leaves no hole
  # behind: the slot goes back and the next `cachePut` reuses it. Recording the id
  # as absent is the same fact for a reader without a cache - and it spares the
  # deletes below, and every later stale link, a YottaDB read that could only
  # confirm the NODE value is no longer there.
  ix.cacheFree(id)
  ix.absentIds.incl(id)

  ix.bumpLive(-1)

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


proc searchStored(ix: HnswIndex, q: StoredVec, k, ef: int): seq[Hit] =
  ## The walk itself: greedy descent through the layers above 0 with `ef = 1`,
  ## then best-first on layer 0 with `ef`/`k` candidates.
  ##
  ## Shared by `search` and `searchInt8`, which differ only in how `q` got onto
  ## the index's grid - keeping one copy is what stops the two entry points from
  ## drifting apart in what they return.
  if ix.entry < 0 or ix.count == 0:
    return @[]
  let width = max(if ef > 0: ef else: ix.params.efSearch, k)

  var ep = ix.entry
  for l in countdown(ix.maxLevel, 1):
    let w = ix.searchLayer(q, [ep], 1, l)
    if w.len > 0:
      ep = w[0].id
  let w = ix.searchLayer(q, [ep], width, 0)
  for i in 0 ..< min(k, w.len):
    result.add Hit(id: w[i].id, dist: w[i].dist)

proc search*(ix: HnswIndex, vec: openArray[float32], k = 5, ef = 0): seq[Hit] =
  ## The `k` closest nodes to `vec`, nearest first.
  ##
  ## `ef` is the search candidate width (>= k); it defaults to the index's
  ## efSearch. Larger values trade speed for recall.
  ##
  ## `vec` is a raw embedding; it is normalised and quantized here. When the
  ## caller already holds the codes - a benchmark of the raw walk, or a host that
  ## keeps its vectors quantized - go straight to `searchInt8` and skip both.
  if ix.entry < 0 or ix.count == 0:
    return @[]
  var v = @vec
  doAssert v.len == ix.dim, "vector has " & $v.len & " dims, index expects " & $ix.dim
  normalize(v)
  # The query goes onto the same grid as the stored vectors - otherwise the codes
  # compared in `searchLayer` would not mean the same magnitudes.
  let q = ix.toStored(v)
  ix.searchStored(q, k, ef)


proc searchInt8*(ix: HnswIndex, codes: openArray[int8], k = 5, ef = 0,
                 scale = 0.0'f32): seq[Hit] =
  ## `search` for a query that is *already* quantized: `codes` is what
  ## `quantizeInt8` produced from an L2-normalised float32 query, as a host that
  ## receives int8 vectors from its embedding side has in hand.
  ##
  ## Nothing is normalised or quantized here, which is the point: a benchmark of
  ## this entry point measures the graph walk plus the int8 kernels, not the
  ## pipeline that feeds them.
  ##
  ## Only an int8 storage mode can use this, and the codes have to be on the grid
  ## the index stores: `quantScale` for `vqInt8Fixed`, the vector's own
  ## `max|x| / 127` grid for `vqInt8`. Both the mode and the dimension are
  ## checked, because the failure without them is not an exception - codes
  ## compared against float32 vectors reach `simdDotI8` through an empty seq and
  ## read past it.
  ##
  ## `scale` is the query's own quantization scale, and it only sets the
  ## magnitude of the returned distances, never which ids come back or in what
  ## order: `dist` is `1 - scale * storedScale * sum`, so a positive `scale` is
  ## one constant factor across every candidate, and every decision in
  ## `searchLayer` is a comparison between two of those distances. `0` - the
  ## default - asks for the scale the index can supply itself, which is
  ## `quantScale` for `vqInt8Fixed` (query and stored codes share the grid, so
  ## the distances are the true cosine distances, identical to what `search`
  ## reports for the same query) and `1` for `vqInt8`, where the grid is per
  ## vector and only the ordering is meaningful unless the caller passes the
  ## scale it quantized with.
  if ix.params.quant == vqNone:
    raise newException(ValueError,
      "searchInt8 needs an int8 index, " & $ix.params.global & " stores " &
      $ix.params.quant & "; build the index quantized, or use search()")
  if codes.len != ix.dim:
    raise newException(ValueError,
      "query has " & $codes.len & " codes, index expects " & $ix.dim)
  if scale < 0:
    raise newException(ValueError,
      "searchInt8 scale is the query's quantization scale, so > 0 (or 0 to " &
      "take the index's own)")
  let q = StoredVec(codes: @codes,
                    scale: if scale > 0: scale
                           elif ix.params.quant == vqInt8Fixed: ix.params.quantScale
                           else: 1.0'f32)
  ix.searchStored(q, k, ef)


proc blobVector(ix: HnswIndex, blob: string): StoredVec =
  ## Decode a stored vector blob - a `^...NODE` value as it comes out of YottaDB -
  ## the way `ix.params.quant` says it was written. Exactly the decode `loadVec`
  ## does for a neighbour: the two must not drift, or a query would be compared
  ## on a different grid than the vectors it is compared against.
  ##
  ## The length has to match the mode exactly. That check is the difference
  ## between "your blob is the wrong thing" and a walk over 388 codes that the
  ## 384-dim kernels are then happy to read past.
  if blob.len != ix.bytesPerVector:
    raise newException(ValueError,
      "query blob is " & $blob.len & " bytes, index expects " & $ix.bytesPerVector &
      " for " & $ix.params.quant & " at dim " & $ix.dim &
      (if ix.dim <= 0:
         "; the index records no dimension (" & ix.params.globalMeta &
         "(\"dim\") is missing or 0) and it could not be read off the stored blobs"
       else: ""))
  case ix.params.quant
  of vqNone:
    result.floats = unpackFloats(blob)
  of vqInt8:
    result = unpackQVec(blob)
  of vqInt8Fixed:
    result.codes = unpackInt8(blob)
    result.scale = ix.params.quantScale


proc searchBlob*(ix: HnswIndex, blob: string, k = 5, ef = 0): seq[Hit] =
  ## `search` with a query that is already in the form the index keeps its own
  ## vectors in: the `^...NODE` value itself, so nothing is decoded to float32
  ## and nothing is normalised or quantized on the way in. Mode-agnostic - it
  ## reads `vqNone` (the float32 vector), `vqInt8` (float32 scale, then codes) and
  ## `vqInt8Fixed` (codes alone) alike.
  ##
  ## This is the entry point for walking the index with a vector it already
  ## holds, and the one that makes a benchmark measure the walk alone. Note what
  ## it does *not* do: a `vqNone` blob is stored L2-normalised, so it is used as
  ## it is rather than normalised again (`search` normalises its copy, and a
  ## second pass over an already-normalised vector shifts the last bits and can
  ## round a code differently); a `vqInt8` blob carries its own scale, so the
  ## distances come out with the same magnitudes `search` reports.
  ##
  ## The blob must be exactly `bytesPerVector` long for the index's mode - see
  ## `blobVector`, which does the check and the decode.
  ix.searchStored(ix.blobVector(blob), k, ef)


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


# ---------------------------------------------------------------------------
# Maintenence
# ---------------------------------------------------------------------------

proc dropIndex*(params: HnswParams) =
    for gbl in [params.globalNode, params.globalKey, params.globalMeta, params.globalLevel, params.globalKey, params.globalLinks, params.globalQueue]:
        ydb_delete(gbl, @[], YDB_DEL_TREE)
        echo gbl, " deleted"
