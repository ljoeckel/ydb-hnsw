import std/[strformat, strutils, times]
import ../ydbhnsw
import hnsw_common       # getRssRef / getRssTitle / getRssDescription

const
    BatchSize = 128
    MinDescriptionLen = 35
    MinSim = 0.90'f32


when isMainModule:
    var cnt, hits = 0
    let params = hnswParams("^HNSWArticles", cacheVectors=true, quant=vqInt8)
    let ix = openHnsw(params)
    
    let t1 = getTime() # set start for caclulation

    for (subs, vs) in QueryItr ^HNSWArticlesNODE.sv: 
        let id = Get ^HNSWArticlesKEY(subs[0])
        #echo getRssTitle(id)

        # The stored value is the query: `searchBlob` decodes it exactly the way
        # `loadVec` decodes a neighbour and walks - no float32, no normalise, no quantize.
        for hit in ix.searchBlob(vs):
            if hit.sim() > MinSim:
                #echo &"   {hit.sim()} {hit.id} {getRssTitle(hit.id)}"
                inc hits
        inc cnt

        if cnt mod 100 == 0:
            let avg = (getTime() - t1).inMicroseconds / cnt
            echo &"cnt:{cnt} hits:{hits} avg:{avg}"
            
    echo "cnt=", cnt, " hits=", hits
