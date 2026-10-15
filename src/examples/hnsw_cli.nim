## Build (or extend) the HNSW index over the article embeddings, with the graph
## persisted in YottaDB.
##
##   nimble runHnsw               # extend the default index
##   ./hnsw_articles --help       # all options
##
## An index keeps the vector storage mode of its *first* vector, so quantizing an
## existing index means building into a fresh global:
##
##   ./hnsw_articles --index=^HNSWArticlesQ --quant=int8fixed --scale=auto
##
## Needs the venv on PYTHONPATH (nimble runHnsw does that):
##   PYTHONPATH=/home/ljoeckel/bert_env/lib/python3.12/site-packages ./hnsw_articles

import std/[os, strformat, strutils, times, enumerate]
import ../ydbhnsw
import hnsw_common

const
    DefaultIndex = "^HNSWArticles"
    MinSim = 0.50'f32

proc embedNormalized(title: string): seq[float32] =
    result = embed(@[hnswNormalize(title)])[0]

when isMainModule:
    var params = hnswParams(DefaultIndex, model = DefaultModel, M = 16, efConstruction = 200, efSearch = 256, cacheVectors = false)
    let ix: HnswIndex = openHnsw(params)

    while true:
        stdout.write "Search for: "    
        var searchFor = readLine(stdin)
        if searchFor == "": quit(0)
        
        let vec = embedNormalized(searchFor)
        let hits = ix.search(vec, k=20)

        echo "\nSearch results for: ", searchFor
        for hit in hits:
            if hit.sim() > MinSim:
                 echo &"  {hit.sim()} {hit.id} {getRssTitle(hit.id)}"
                 echo &"     {getRssDescription(hit.id)}"

