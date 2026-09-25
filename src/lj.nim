import yottadb
import ydbutils

import std/[strformat, strutils, tables, enumerate, times]

import bert_nim     # nimpy glue: embed(); its own demo stays dormant
import hnsw
import rss_bridge
import common       # getRssRef / getRssTitle / getRssDescription
import rsstypes

when isMainModule:
    let params = hnswParams("^HNSWArticles", cacheVectors=true)
    echo &"Opening the HNSW index with {params}"
    var ix = openHnsw(params)


    var cnt = 0
    var hits = 0
    var t1 = getTime()
  
    for k,v in QueryItr ^RSSItemVec.kv:
        let n = v.len div sizeof(float32)
        let vec = newSeqUninit[float32](n) 
        copyMem(vec[0].addr, v[0].unsafeAddr, n * sizeof(float32))

        inc cnt
            
        # collect articles that seams are related

        for hit in ix.search(vec, k = 3):
            if hit.sim() > 0.7:
                #echo "   ", hit.sim()," ", getRssTitle(hit.id)
                inc hits

        if cnt mod 100 == 0:
            let avg = (getTime() - t1).inMicroseconds / 100
            t1 = getTime()
            echo &"cnt:{cnt} hits:{hits} avg:{avg}"
            updateDBStats("hnsw_search")




#for (k,v) in QueryItr ^RSSItemVec.kv:
#    echo k, ", ", v.len