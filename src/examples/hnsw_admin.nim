import std/[strutils, strformat]
import ../ydbhnsw

let params = hnswParams("^HNSWArticles")
echo &"Opening the HNSW index with {params}"
var ix = openHnsw(params)

proc repair() =
    echo ix.levelsSummary()
    let cnt = ix.repair()
    echo "repair cnt=", cnt
    echo ix.levelsSummary()

proc checkKey() =
    var cnt = 0
    for k,v in QueryItr ^HNSWArticlesKEY("4,0").kv:
        echo k,"=",v
        inc cnt
        if cnt > 100: break

proc deleteIndex(indexName: string) =
    let parts = @["KEY", "LEVEL", "LINKS", "META", "NODE"]
    for part in parts:
        let gbl = indexName & part
        Kill: @gbl
        echo "Killed ", gbl


#repair()
#checkKey()
deleteIndex("^HNSWArticles")