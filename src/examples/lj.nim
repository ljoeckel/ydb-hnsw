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

    #for (subs, vs) in QueryItr ^HNSWArticlesNODE.sv: 
    for subs in QueryItr ^HNSWArticlesNODE.keys: 
        let id = parseInt(subs[0])
        let vec = ix.loadVec(id)
        echo vec
        # let id = Get ^HNSWArticlesKEY(subs[0])
        # #echo getRssTitle(id)
        # #let vs = Get ^HNSWArticlesNODE(subs)
        # #let vec = unpackFloats(vs)
        # let vec = unpackInt8(vs)
        # inc cnt
           
        # collect articles that seams are related
        # for hit in ix.search(vec, k = 3):
        #     if hit.sim() > MinSim:
        #         echo &"   {hit.sim()} {hit.id} {getRssTitle(hit.id)}"
        #         inc hits

        # if cnt mod 100 == 0:
        #     let avg = (getTime() - t1).inMicroseconds / cnt
        #     echo &"cnt:{cnt} hits:{hits} avg:{avg}"
        #     updateDBStats("hnsw_search")
            
    echo "cnt=", cnt, " hits=", hits
#for (k,v) in QueryItr ^RSSItemVec.kv:
#    echo k, ", ", v.len
