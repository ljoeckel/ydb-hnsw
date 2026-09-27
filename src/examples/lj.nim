import std/[strformat, times]
import ../ydbhnsw
import hnsw_common       # getRssRef / getRssTitle / getRssDescription

when isMainModule:
    var params = hnswParams("^HNSWArticles", cacheVectors=true)
    echo &"Opening the HNSW index with {params}"
    var ix = openHnsw(params)

    var cnt = 0
    var hits = 0
    var t1 = getTime()
  
    for subs in QueryItr ^HNSWArticlesNODE.keys:
        let id = Get ^HNSWArticlesKEY(subs[0])
        echo getRssTitle(id)
        
        let v = Get ^HNSWArticlesNODE(subs)
        let n = v.len div sizeof(float32)
        let vec = newSeqUninit[float32](n) 
        copyMem(vec[0].addr, v[0].unsafeAddr, n * sizeof(float32))
        inc cnt
           
        # collect articles that seams are related
        for hit in ix.search(vec, k = 3):
            if hit.sim() > 0.7:
                echo &"   {hit.sim()} {hit.id} {getRssTitle(hit.id)}"
                inc hits

        if cnt mod 100 == 0:
            let avg = (getTime() - t1).inMicroseconds / cnt
            echo &"cnt:{cnt} hits:{hits} avg:{avg}"
            #updateDBStats("hnsw_search")
            




#for (k,v) in QueryItr ^RSSItemVec.kv:
#    echo k, ", ", v.len