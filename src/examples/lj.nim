import std/[strformat, strutils, times]
import ../ydbhnsw
import hnsw_common       # getRssRef / getRssTitle / getRssDescription

const
    BatchSize = 128
    MinSim = 0.90'f32


when isMainModule:
    var cnt, notSelfCount, hitsCount = 0
    let params = hnswParams("^HNSWArticles", cacheVectors=true)
    let ix = openHnsw(params)
    echo $ix.params
    echo ix.levelsSummary()
    
    let t1 = getTime() # set start for caclulation
    var t0 = gettime() # for each 100 segment
    for (subs, vs) in QueryItr ^HNSWArticlesNODE.sv: 
        let id = Get ^HNSWArticlesKEY(subs[0])

        # The stored value is the query: `searchBlob` decodes it exactly the way
        # `loadVec` decodes a neighbour and walks - no float32, no normalise, no quantize.
        let hits = ix.searchBlob(vs)

        if hits.len == 0:
            echo "    Self not found: ", id
            inc notSelfCount
        else:
            discard
        #    for hit in hits:
        #        if hit.sim() > MinSim:
        #             echo &"    {hit.sim()} {hit.id} {getRssTitle(hit.id)}"

        inc cnt
        inc(hitsCount, hits.len)

        if cnt mod 100 == 0:
            let avg100 = (getTime() - t0).inMicroseconds / 100
            let avg = (getTime() - t1).inMicroseconds / cnt
            echo &"cnt:{cnt} hits:{hitsCount} notSelf:{notSelfCount} avg100: {avg100}, avg: {avg}"
            t0 = getTime()

    echo fmt"cnt: {cnt}, hits: {hitsCount}, notSelfCount: {notSelfCount}"
