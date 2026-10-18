## Build (or extend) the HNSW index over the article embeddings, with the graph
## persisted in YottaDB.

import std/[strformat, strutils, times, enumerate]
import ydbhnsw
import hnsw_common
import types

var started: float

proc logInfo(cnt: int) =
    let ips = cnt.float / (epochTime() - started)
    echo &"added to queue: {cnt} with {ips:5.2f} nodes/s"


when isMainModule:
    var params = getHnswParams()
    if params.reset:
        dropIndex(params)
        quit(0)

    var ix = openHnsw(params)
    var total = 0
    started = epochTime()

    for (cnt, iterResult) in enumerate(batchedRSSTitleIter(ix)):
        if cnt >= params.limit:  break
        var workItem = WorkItem(idxref: iterResult.idxref, vec: packFloats(iterResult.vec))
        saveObject[WorkItem](iterResult.idxref, workItem, globalName=ix.params.globalQueue)
        if cnt mod 100 == 0: logInfo(cnt)
        total = cnt

    logInfo(total)    


