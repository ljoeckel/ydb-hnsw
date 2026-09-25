## Build (or extend) the HNSW index over the article embeddings, with the graph
## persisted in YottaDB.
##
##   nimble runHnsw               # extend the default index
##   ./hnsw_articles --help       # all options
##
## An index keeps the vector storage mode of its *first* vector, so quantizing an
## existing index means building into a fresh global:
##
##   ./hnsw_articles --index=^HNSWArticlesQ --quant=int8fixed --scale=auto
##
## Needs the venv on PYTHONPATH (nimble runHnsw does that):
##   PYTHONPATH=/home/ljoeckel/bert_env/lib/python3.12/site-packages ./hnsw_articles

import std/[os, strformat, strutils, times]
import bert_nim     # nimpy glue: embed(); its own demo stays dormant
import hnsw
import rss_bridge
import yottadb      # only for --reset (dropping the target index)

const
    DefaultIndex = "^HNSWArticles"

    Usage = """
usage: hnsw_articles [option=value]...

  --index=GLOBAL   index to build (default ^HNSWArticles). YottaDB names allow
                   letters, digits and % only - no underscores.
  --quant=MODE     none | int8 | int8fixed (default none, i.e. float32).
                   An index keeps the mode of its first vector.
  --model=NAME     sentence-transformers model the vectors are built with.
                   Default """ & DefaultModel & """ - an index keeps the model
                   of its first vector, because vectors of two models do not mix.
  --scale=FLOAT    int8 grid for --quant=int8fixed. "auto" (the default in that
                   mode) derives it from --sample real embeddings.
  --sample=N       titles embedded to derive that grid (default 5000).
  --limit=N        stop after N articles from the RSS iterator (trial run).
                   --limit=0 with --reset empties the target and builds nothing.
  --reset          delete the target index before building. Requires an explicit
                   --index=..., because it removes every node of it.
  --help           this text.

Stored size per vector: none 4*dim B, int8 4+dim B, int8fixed dim B.
"""

type
    Options = object
        index: string
        quant: VecQuant
        quantGiven: bool
        model: string
        modelGiven: bool
        scale: float32
        scaleAuto: bool
        reset: bool
        limit: int
        sample: int
        help: bool

proc parseMode(s: string): VecQuant =
    case s.toLowerAscii
    of "none", "float", "float32": vqNone
    of "int8", "sq": vqInt8
    of "int8fixed", "fixed": vqInt8Fixed
    else: quit &"unknown --quant={s} (none | int8 | int8fixed)", 2

proc parseArgs(): Options =
    result.index = DefaultIndex
    result.limit = int.high
    result.sample = 5000
    result.scaleAuto = true
    result.model = DefaultModel
    for i in 1 .. paramCount():
        let a = paramStr(i)
        let eq = a.find('=')
        let key = if eq < 0: a else: a[0 ..< eq]
        let val = if eq < 0: "" else: a[eq + 1 .. ^1]
        case key
        of "--index":
            if val.len == 0: quit "--index needs a global name, e.g. --index=^HNSWQ", 2
            result.index = val
        of "--model":
            if val.len == 0: quit "--model needs a name, e.g. --model=all-MiniLM-L6-v2", 2
            result.model = val
            result.modelGiven = true
        of "--quant":
            result.quant = parseMode(val)
            result.quantGiven = true
            result.scaleAuto = val.toLowerAscii in ["int8fixed", "fixed"]
        of "--scale":
            if val.toLowerAscii == "auto": result.scaleAuto = true
            else:
                result.scale = float32(parseFloat(val))
                result.scaleAuto = false
        of "--sample": result.sample = parseInt(val)
        of "--limit": result.limit = parseInt(val)
        of "--reset": result.reset = true
        of "--help", "-h": result.help = true
        else: quit &"unknown option {a} - try --help", 2

proc dropIndex(params: HnswParams) =
    ## `--reset`: the three globals of one index, and nothing else. The names
    ## come from the params, so they cannot drift from the ones `openHnsw` uses.
    for g in [params.globalNode, params.globalKey, params.globalMeta]:
        ydb_delete(g, @[], YDB_DEL_TREE)

# proc bytesPerVector(ix: HnswIndex): int =
#     ## What one stored vector costs, from the mode and the dimension alone - the
#     ## blobs do not have to be read back for this.
#     case ix.params.quant
#     of vqNone: 4 * ix.dim
#     of vqInt8: 4 + ix.dim
#     of vqInt8Fixed: ix.dim

proc embedNormalized(title: string): seq[float32] =
    ## Exactly what `add` stores: the vector of the normalised title, L2
    ## normalised. The int8 grid has to be measured on this, not on the raw
    ## embedding, or the codes would be scaled against the wrong magnitudes.
    result = embed(@[hnswNormalize(title)])[0]
    normalize(result)

when isMainModule:
    let opts = parseArgs()
    if opts.help:
        echo Usage
        quit 0

    if opts.reset and opts.index == DefaultIndex:
        quit &"--reset deletes every node of the target, and {DefaultIndex} is the live " &
             &"index - name a target explicitly (--index=...), or build the new one " &
             &"alongside it as --index={DefaultIndex}Q", 2

    # Graph parameters are fixed for this program; the command line only picks
    # the index, the embedding model, the storage mode and the grid. `hnswParams`
    # derives the `^...NODE/KEY/META` names, so `dropIndex` and `openHnsw` (and
    # the KEY- and NODE-side helpers below) all address the same globals, and
    # `openHnsw` loads `params.model` (see `bert_nim.loadModel`).
    var params = hnswParams(opts.index, model = opts.model,
                            M = 16, efConstruction = 200, efSearch = 64,
                            quant = opts.quant, quantScale = opts.scale)

    if opts.reset:
        dropIndex(params)
        echo &"deleted {opts.index} (NODE/KEY/META)"

    var ix: HnswIndex
    try:
        ix = openHnsw(params)
    except ValueError as e:
        # vqInt8Fixed with neither --scale nor a grid in META: openHnsw refuses,
        # because a grid has to exist before the first vector is written. Measure
        # one from real embeddings. The build loop embeds those titles a second
        # time - --sample is small next to a full run.
        if opts.quant != vqInt8Fixed or not opts.scaleAuto:
            raise e
        let sampleStart = epochTime()
        var sample: seq[seq[float32]]
        for (_, title) in RSSItemIter(opts.sample):
            sample.add embedNormalized(title)
        if sample.len == 0:
            quit "no articles to sample - is ^RSSItem populated?", 1
        let grid = quantizeScale(sample)
        echo &"grid from {sample.len} sample titles: {grid:.8f} " &
             &"({epochTime() - sampleStart:.1f} s)"
        params.quantScale = grid
        ix = openHnsw(params)

    # A non-empty index keeps the mode (and the grid) of its first vector. Say so
    # rather than quietly building more of the same kind - "I asked for int8 and
    # got float32 blobs" would otherwise only show up in the blob sizes.
    if opts.quantGiven and ix.params.quant != opts.quant:
        quit &"{opts.index} already holds {ix.params.quant} vectors and keeps them. " &
             &"Build into a fresh global (--index=^SomethingElse --quant={opts.quant}), " &
             &"or wipe the target with --reset --index={opts.index}", 2
    if not opts.scaleAuto and opts.scale > 0 and ix.params.quant == vqInt8Fixed and
       ix.params.quantScale != opts.scale:
        echo &"note: {opts.index} keeps its grid {ix.params.quantScale:.8f}, " &
             &"not --scale={opts.scale:.8f}"
    if opts.modelGiven and ix.params.model.len > 0 and ix.params.model != opts.model:
        echo &"note: {opts.index} holds vectors of {ix.params.model} and keeps that " &
             &"model, not --model={opts.model}"

    echo &"{opts.index}: {ix.params.model}, {ix.params.quant}, {ix.liveCount} nodes" &
         (if ix.dim > 0: &", dim {ix.dim}, {bytesPerVector(ix)} B/vector" else: "")

    let started = epochTime()
    var cnt = 0
    var added = 0
    for (idxref, title) in RSSItemIter():
        if cnt >= opts.limit:
            break
        inc cnt
        # One Data call on this index's key global: an article that is already
        # keyed is never re-embedded.
        if not ix.contains(idxref):
            discard ix.add(embedNormalized(title), idxref)
            inc added
            if added mod 500 == 0:
                echo &"  {added} added, {cnt} seen, {ix.liveCount} nodes, {epochTime() - started:.0f} s"
                updateDBStats("hnsw_articles")

    echo &"done: {added} added, {cnt} seen, live={ix.liveCount} count={ix.count} in {epochTime() - started:.0f} s"
    if ix.dim > 0:
        echo &"  {ix.params.quant}, {bytesPerVector(ix)} B/vector, {bytesPerVector(ix) * ix.liveCount} B for this index, {ix.clamped} clamped"
    echo &"  {levelsSummary(ix)}"
    updateDBStats("hnsw_articles")


