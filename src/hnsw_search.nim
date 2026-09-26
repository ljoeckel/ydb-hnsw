import std/[strformat, strutils, tables, enumerate]

import bert_nim     # nimpy glue: embed(); its own demo stays dormant
import hnsw
import rss_bridge
import common       # getRssRef / getRssTitle / getRssDescription
import yottadb
import rsstypes

const BatchSize = 256

var empty, duplicates, vectors = 0

proc sanitize(label: string): bool =
    if label.isEmptyOrWhitespace(): return true
    if label == "mehr": return true
    false


proc delete(ix: HnswIndex, hit: Hit) =
    # Get rssref to access ydb
    let rssref = getRssRef(hit.id)
    if rssref.len == 0:
        echo &"Could not find storage reference {rssref}"
        return

    # Remove form HNSW vector db
    let rc = ix.delete(hit.id)
    if not rc:
        echo &"Could not delete {hit.id} from HNSW index"
        return

    # Remove from YottaDB
    deleteObject[RSSItem](rssref)


proc updateVecEntry(key: string, vec: seq[float32]): bool =
    if 0 == Data ^RSSItemVec(key):
        var packed = newString(vec.len * sizeof(float32))
        copyMem(packed[0].addr, vec[0].unsafeAddr, vec.len * sizeof(float32))
        Set: ^RSSItemVec(key, "vec") = packed
        return true
    return false


proc findDuplicates(ix: HnswIndex, vec: seq[float32], k = 5, remove = false) =
    var t = newTable[string, seq[Hit]]()

    # collect articles that seams are related
    for hit in ix.search(vec, k = k):
        if hit.sim() > 0.7:
            t.mgetOrPut(getRssTitle(hit.id), @[]).add(hit)

    for title, hits in t:
        var lastDescription = ""
        for hit in hits:
            let description = getRssDescription(hit.id)
            if lastDescription != "" and description == lastDescription:
                inc duplicates
                if remove: delete(ix, hit)
            if sanitize(description):
                inc empty
                if remove: delete(ix, hit)

            lastDescription = description


iterator processTitles(titles: seq[string], keys: seq[string], cnt: int): (string, string, seq[float32]) =
    ## Embed one batch of titles and yield the vectors.
    ##
    ## Deliberately top-level, taking `titles` as a parameter: as a nested
    ## iterator that *captured* `xIter`'s locals it became a closure iterator,
    ## and Nim 2.2.12 fails to give that closure's environment a location when
    ## the enclosing inline iterator also inlines `RSSItemIter` (which contains
    ## `if reverse: ...yield... else: ...yield...`). The result is
    ## `internal error: expr: var not init :env_<id>` at the `for ... in
    ## processTitles()` site.
    let stop = min(BatchSize, titles.len)
    let flat = embedFlat(titles)
    let dim = flat.len div stop
    for i in 0 ..< stop:
        let vec = flat[i * dim ..< (i + 1) * dim]
        yield (keys[i], titles[i], vec)


iterator batchedRSSItemIter(batchSize: int): (string, string, seq[float32]) =
    var titles = newSeqOfCap[string](batchSize)
    var keys = newSeqOfCap[string](batchSize)
    var cnt = 0

    for (key, title) in RSSItemIter(reverse=true):
        keys.add(key)
        titles.add(hnswNormalize(title))
        inc cnt
        if cnt mod batchSize == 0:
            for (key, title, vec) in processTitles(titles, keys, cnt):
                yield (key, title, vec)
            titles.setLen(0)
            keys.setLen(0)

    if titles.len > 0:
        for (key, title, vec) in processTitles(titles, keys, cnt):
            yield (key, title, vec)


when isMainModule:
    let params = hnswParams("^HNSWArticles", cacheVectors=true, batchSize=BatchSize)
    echo &"Opening the HNSW index with {params}"
    var ix = openHnsw(params)

    for (cnt, key, title, vec) in enumerate(batchedRSSItemIter(params.batchSize)):
        #echo cnt, " ", key, " ", title
        if updateVecEntry(key, vec): inc vectors

        findDuplicates(ix, vec, k=3, remove=true)
        if cnt mod params.batchSize == 0:
            echo &"findDuplicates cnt:{cnt}, vectors:{vectors}, empty:{empty}, duplicates:{duplicates}"
            updateDBStats("hnsw_clean")        
    
    echo "New Vectors: ", vectors
    echo "      Empty: ", empty
    echo " Duplicates: ", duplicates
    updateDBStats("hnsw_clean")