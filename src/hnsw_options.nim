import std/[parseopt, strutils, strformat, os]


const
    DefaultModel = "paraphrase-multilingual-MiniLM-L12-v2"
    DefaultIndex = "^HNSWArticles"


var Usage = """
usage: <program> [option=value]...

  --index=GLOBAL      index to build (default ^HNSWArticles). YottaDB names allow
                      letters, digits and % only - no underscores.
  --quant=MODE        none | int8 | int8fixed (default none, i.e. float32).
                      An index keeps the mode of its first vector.
  --m                 Number of connections per node
  --efConstruction    Size of the candidate pool at building
  --efSearch          Size of the candidate pool when querying
  --linkFmt=MODE      The dataformat for the links. Defaults to lfUint32.
                      lfInt for int64 or lfUint32 for uint32
                      lfUint32 max indexsize is 4.294.967.295 entries.
  --model=NAME        sentence-transformers model the vectors are built with.
                      Default """ & DefaultModel & """ - an index keeps the model
                      of its first vector, because vectors of two models do not mix.
                      Use 'NONE' to indicate not to load the model.
  --scale=FLOAT       int8 grid for --quant=int8fixed. "auto" (the default in that
                      mode) derives it from --sample real embeddings.
  --cache=BOOL        Enable/disable both caches (Vector/Links) 
  --cacheVectors=BOOL Cache all vectors
  --preloadCacheVectors=BOOL Preload at start (defaults to true)
  --cacheLinks=BOOL   Cache the links
  --preloadCacheLinks=BOOL   Preload links at start (defaults to true)
  --sample=N          titles embedded to derive that grid (default 5000).
  --limit=N           stop after N records (trial run).
                      --limit=0 with --reset empties the target and builds nothing.
  --reset             delete the target index before building. Requires an explicit
                      --index=..., because it removes every node of it.
  --batchSize         Number of items per GPU batch. Defaults to 128
  --help              this text.
  --verbose           Show configuration

Stored size per vector: none 4*dim B, int8 4+dim B, int8fixed dim B.
"""

type
    VecQuant* = enum
        ## How an index stores (and compares) its vectors.
        vqNone      ## float32, 4 bytes per dimension - default, lossless
        vqInt8      ## scalar-quantized, grid stored with each vector (4 + dim bytes)
        vqInt8Fixed ## scalar-quantized on one grid for the whole index (dim bytes)

    LinkFmt* = enum
        ## How an index stores a neighbour id in `^...LINKS`, see `HnswParams.linkFmt`.
        ##
        ## Both are decoded back into `int`; the choice only decides how many bytes a
        ## link costs in YottaDB.
        lfInt     ## int64, 8 bytes per id - default, and what older indexes hold
        lfUint32  ## uint32, 4 bytes per id - halves the link tree, ids < 2^32

    HNSWOptions* = object
        index: string = DefaultIndex
        model: string = DefaultModel
        M: int = 16
        efConstruction: int = 200
        efSearch: int = 64
        seed: int = 1234
        dim: int = 0
        quant: VecQuant = vqInt8
        quantScale: float32 = 0.0'f32
        quantGiven: bool
        modelGiven: bool = true
        scale: float32
        scaleAuto: bool = true
        reset: bool
        limit: int = int.high
        sample: int 
        batchSize: int = 128
        help: bool
        linkFmt: Linkfmt = LinkFmt.lfUInt32
        cacheVectors: bool = true
        preloadCacheVectors: bool = true
        cacheLinks: bool = true
        preloadCacheLinks: bool = true
        verbose: bool = false


    HnswParams* = object
        model*: string = DefaultModel   ## The hugging face model name
        global*: string                 ## YottaDB global basename ^HNSWxxx
        globalNode*: string             ## YottaDB global ^HNSWxxxNODE
        globalLinks*: string            ## YottaDB global ^HNSWxxxLINKS
        globalLevel*: string            ## YottaDB global ^HNSWxxxLEVEL
        globalKey*: string              ## YottaDB global ^HNSWxxxKEY
        globalMeta*: string             ## YottaDB global ^HNSWxxxMETA
        globalQueue*: string            ## YottaDB global ^HNSWxxxQUEUE
        M*: int                         ## links per node above layer 0
        efConstruction*: int            ## candidate width while inserting
        efSearch*: int                  ## default candidate width while searching
        seed*: int                      ## RNG seed for the random level draw
        dim*: int                       ## Dimension
        quant*: VecQuant                ## vector storage / quantization mode
        quantScale*: float32            ## the int8 grid for vqInt8Fixed, ignored otherwise
        linkFmt*: LinkFmt               ## neighbour id width stored in LINKS, see `LinkFmt`
        cacheVectors*: bool             ## cache every new vector into memory
        preloadCacheVectors*: bool      ## Preload all index vectors at start
        cacheLinks*: bool               ## keep every neighbour list this process touches in memory, see `linksAt`; the link half of `cacheVectors`
        preloadCacheLinks*: bool        ## Preload all links at start
        batchSize*: int                 ## Default batch size for g.e. batchedIterator, etc.
        verbose*: bool
        reset*: bool
        limit*: int


proc parseMode(s: string): VecQuant =
    case s.toLower
    of "none", "float", "float32": vqNone
    of "int8", "sq": vqInt8
    of "int8fixed", "fixed": vqInt8Fixed
    else: quit &"unknown --quant={s} (none | int8 | int8fixed)", 2

proc parseLinkFmt(s: string): LinkFmt =
    case s.toLower
    of "lfuint32": lfUint32
    of "lfInt": lfInt
    else: quit &"unknown --linkFmt={s} (lfUint32 | lfInt)", 2

proc parseBool(s: string): bool =
    case s.toLower
    of "t", "true" : true
    of "f", "false": false
    else: false

proc parseArgs(args: seq[string], global: string, model: string): HNSWOptions =
    result = HNSWOptions()
    result.index = global
    result.model = model

    for kind, key, val in getopt(args):  
      case kind
      of cmdEnd: break
      of cmdShortOption, cmdLongOption:
        case toLowerAscii(key)
        of "index":
            result.index = if val.startsWith("^"): val else: "^" & val
            if result.index.len == 1: quit &"Missing index", 2
        of "model":
            result.model = val.toLowerAscii()                
            result.modelGiven = result.model != "none"
        of "quant":
            result.quant = parseMode(val)
            result.quantGiven = true
            result.scaleAuto = val.toLowerAscii in ["int8fixed", "fixed"]
        of "linkfmt":
            result.linkFmt = parseLinkFmt(val)
        of "m":
            result.M = parseInt(val)
        of "efconstruction":
            result.efConstruction = parseInt(val)
        of "efsearch":
            result.efSearch = parseInt(val)
        of "scale":
            if val.toLowerAscii == "auto": result.scaleAuto = true
            else:
                result.scale = float32(parseFloat(val))
                result.scaleAuto = false
        of "cache":
            let caching = parseBool(val)
            result.cacheVectors = caching
            result.cacheLinks = caching
            result.preloadCacheVectors = caching
            result.preloadCacheLinks = caching
        of "cachevectors": 
            result.cacheVectors = parseBool(val)
        of "preloadcachevectors": 
            result.preloadCacheVectors = parseBool(val)
        of "cachelinks": 
            result.cacheLinks = parseBool(val)
        of "preloadcachelinks": 
            result.preloadCacheLinks = parseBool(val)
        of "sample": 
            result.sample = parseInt(val)
        of "batchsize": 
            result.batchSize = parseInt(val)
        of "limit": 
            result.limit = parseInt(val)
        of "reset": 
            result.reset = true
        of "help", "h":
            quit &"{Usage}", 0
        of "verbose", "v":
            result.verbose = true
        else: echo &"Unknown HnswOption key:{key} val:{val}. Ignored"
      else: discard

# proc parseArgs(global: string, model: string): HNSWOptions =
#     ## The command line as the index tools have always read it.
#     parseArgs(commandLineParams(), global, model)


proc deriveGlobalNames(p: var HnswParams) =
  ## Fill in the `^...NODE` / `^...KEY` / `^...META` names that belong to
  ## `p.global`, unless a caller named them itself.
  if p.globalNode.len == 0:
    p.globalNode = p.global & "NODE"
    p.globalLinks = p.global & "LINKS"
    p.globalLevel = p.global & "LEVEL"
    p.globalKey = p.global & "KEY"
    p.globalMeta = p.global & "META"
    p.globalQueue = p.global & "QUEUE"


proc getHnswParams*(global: string = DefaultIndex, model: string = DefaultModel,
                    args: seq[string] = commandLineParams()): HnswParams =
    ## `args` is the index's argument list. It defaults to the process command
    ## line; a host that has options of its own (the HTTP service's `--port`)
    ## passes the arguments that are meant for the index instead.
    let options = parseArgs(args, global, model)
    result = HnswParams()
    result.global = options.index
    deriveGlobalNames(result)
    result.model = options.model
    result.M = options.M
    result.efConstruction = options.efConstruction
    result.efSearch = options.efSearch
    result.seed = options.seed
    result.dim = options.dim
    result.quant = options.quant
    result.quantScale = options.quantScale
    result.linkFmt = options.linkFmt
    result.cacheVectors = options.cacheVectors
    result.preloadCacheVectors = if options.cacheVectors: options.preloadCacheVectors else: false
    result.cacheLinks = options.cacheLinks
    result.preloadCacheLinks = if options.cacheLinks: options.preloadCacheLinks else: false
    result.batchSize = options.batchSize
    result.verbose = options.verbose
    result.reset = options.reset
    result.limit = options.limit