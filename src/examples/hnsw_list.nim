import std/[strformat, parseopt, strutils, times, random, sets]
import ydbhnsw
import hnsw_common       # getRssRef / getRssTitle / getRssDescription

type AppOptions = object
    bySequence: bool = true
    byRandom: bool = false
    k: int = 5

var Usage = """
usage: <program> [option=value]...

  --[HnswOption]      Will be processed by HnswOption
  --bySequence        Search from 0 .. n in ascending sequence (Default)
  --byRandom          Search randomly
"""

const
    MinSim = 0.90'f32

var cnt, notSelfCount, hitsCount, t100, totalTime = 0
var ydbkey = initYdbKey("^HNSWArticlesNODE")

proc nextItem(ix: HnswIndex, id: int, options: AppOptions ) =
    ydbkey.setSub(0, id)
    let vs = ydb_get(ydbkey)
    let tx1 = getTime()
    let hits = ix.searchBlob(vs, k = options.k)
    let delta = (getTime() - tx1).inMicroseconds

    if ix.params.verbose:
        echo &"{id:>8} {delta:>5} µs : {getRssTitle(id)}"
        for hit in hits:
            if hit.sim() > MinSim:
                 echo &"         {hit.sim():>3.3f} {hit.id:>8} {getRssTitle(hit.id)}"

    inc(t100, delta)
    inc(totalTime, delta)
    inc(hitsCount, hits.len)

    inc cnt
    inc(hitsCount, hits.len)

    if cnt mod 100 == 0:
        let avg100 = t100 / 100
        let avg = totalTime / cnt
        echo &"cnt:{cnt} hits:{hitsCount} notSelf:{notSelfCount} avg100: {avg100:>5.0f} µs., avg: {avg:>5.0f} µs."
        t100 = 0


proc inSequence(ix: HnswIndex, options: AppOptions, limit: int) =
    var item = 0
    for subs in QueryItr ^HNSWArticlesNODE.keys:
        let id = parseInt(subs[0])
        nextItem(ix, id , options)
        inc item
        if item > limit: break
    echo fmt"cnt: {cnt}, hits: {hitsCount}, notSelfCount: {notSelfCount}"


proc inRandom(ix: HnswIndex, options: AppOptions, limit: int) =
    var visited = HashSet[int]()
    var item = 0
    for i in 0 .. ix.live:
        let id = rand(0 ..< ix.live)
        if not visited.contains(id):
            nextItem(ix, id, options)
            visited.incl(id)
        inc item
        if item > limit: break
    echo fmt"cnt: {cnt}, hits: {hitsCount}, notSelfCount: {notSelfCount}"

proc getAppOptions(): AppOptions =
    result = AppOptions()

    for kind, key, val in getopt():  
      case kind
      of cmdEnd: break
      of cmdShortOption, cmdLongOption:
        case toLowerAscii(key)
        of "bysequence": result.bySequence = true
        of "byrandom": result.byRandom = true   
        of "help": echo Usage
        else: discard
      else:
        discard

when isMainModule:
    let hnswParams = getHnswParams()
    let options = getAppOptions()
    if hnswParams.verbose:
        echo "AppOptions: ", options

    let ix = openHnsw(hnswParams)

    if options.byRandom: inRandom(ix, options, hnswParams.limit)
    elif options.bySequence: inSequence(ix, options, hnswParams.limit)
    else:
        echo "Specifiy --byRandom or --bySequence"
    
