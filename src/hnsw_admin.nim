import std/[strutils, strformat]
import ydbhnsw

let params = hnswParams("^HNSWArticles")
echo &"Opening the HNSW index with {params}"
var ix = openHnsw(params)

echo ix.levelsSummary()

let cnt = ix.repair()
echo "repair cnt=", cnt

echo ix.levelsSummary()