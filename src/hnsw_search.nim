import std/[strformat, strutils, tables]

import bert_nim     # nimpy glue: embed(); its own demo stays dormant
import hnsw
import rss_bridge
import common       # getRssRef / getRssTitle / getRssDescription
import yottadb
import rsstypes


var empty, duplicates = 0

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


proc findDuplicates(ix: HnswIndex, vec: seq[float32], k = 5, remove = false) =
    var t = newTable[string, seq[Hit]]()

    # collect articles that seams are related
    for hit in ix.search(vec, k = k):
        if hit.sim() > 0.6:
            t.mgetOrPut(getRssTitle(hit.id), @[]).add(hit)

    for title, hits in t:
        # sanitize
        for hit in hits:
            let description = getRssDescription(hit.id)
            if remove and sanitize(description):
                inc empty
                echo &"EMPTY       id:{hit.id}, rssRef:{getRssRef(hit.id)}  {title}"
                delete(ix, hit)
                discard

        var lastDescription = ""
        for hit in hits:
            let description = getRssDescription(hit.id)
            if remove and lastDescription != "" and description == lastDescription:
                echo &"REDUNDANT   id:{hit.id}, rssRef:{getRssRef(hit.id)} {title}"
                echo &"            {description}"
                inc duplicates
                delete(ix, hit)
            lastDescription = description



when isMainModule:
    const BatchSize = 128
    echo "Scanning Articles"

    var titles: seq[string]
    for (_, title) in RSSItemIter(5000, reverse=true):
        titles.add(hnswNormalize(title))
    echo &"Have {titles.len} titles"

    let params = hnswParams("^HNSWArticles")
    echo &"Opening the HNSW index with {params}"

    var ix = openHnsw(params)

    for start in countup(0, titles.len - 1, BatchSize):
        let stop = min(start + BatchSize, titles.len)
        let flat = embedFlat(titles[start ..< stop])
        if flat.len == 0: continue
        let dim = flat.len div (stop - start)
        for i in start ..< stop:
            findDuplicates(ix, flat[(i - start) * dim ..< (i - start + 1) * dim], k=5, remove=true)
        echo stop - 1, " ", titles[stop - 1]
        updateDBStats("hnsw_clean")
    
    echo "     Empty: ", empty
    echo "Duplicates: ", duplicates

    updateDBStats("hnsw_clean")