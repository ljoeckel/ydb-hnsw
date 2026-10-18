import std/[strformat, strutils, times, enumerate]
import ydbhnsw
import hnsw_common


when isMainModule:
    var params = getHnswParams()
    if params.reset:
        dropIndex(params)
        quit(0)

    var ix = openHnsw(params)

    let started = epochTime()
    var added, cnt = 0

    for (cnt, iterResult) in enumerate(batchedRSSTitleIter(ix)):        
        if cnt >= params.limit: break
        if ix.dim > 0 and ix.dim != iterResult.vec.len:
            raise newException(ValueError, "Embedded vector dimension != Index vector dimension")
 
        if params.verbose:
            echo cnt, " ", iterResult.idxref, " ", iterResult.title

        # add the Title
        discard ix.add(iterResult.vec, iterResult.idxref)
        # add the Description
        #discard ix.add(iterResult.vec2, iterResult.idxref)

        inc added
        if added mod 100 == 0:
            let ips = cnt.float / (epochTime() - started)
            echo &"  {added} added, {cnt} seen, {ix.liveCount} nodes, {ips:5.2f} nodes/s"

    echo &"done: {added} added, {cnt} seen, live={ix.liveCount} count={ix.count} in {epochTime() - started:.0f} s"
    if ix.dim > 0:
        echo &"  {ix.params.quant}, {bytesPerVector(ix)} B/vector, {bytesPerVector(ix) * ix.liveCount} B for this index, {ix.clamped} clamped"
    echo &"  {levelsSummary(ix)}"
    #updateDBStats("hnsw_articles")


