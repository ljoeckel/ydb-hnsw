import std/[os, strformat, strutils, tables, enumerate]
#import std/[htmlparser, xmltree]

import bert_nim     # nimpy glue: embed(); its own demo stays dormant
import hnsw
import rss_bridge
import yottadb
import rsstypes
import common


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


proc findAndRemoveDuplicates(ix: HnswIndex, headline: string, k = 5) =
    var t = newTable[string, seq[Hit]]()
    let title = hnswNormalize(headline)
    let q = embed(@[title])[0]

    # collect articles that seams are related
    for hit in ix.search(q, k = k):
        #let sim = 1.0'f32 - hit.dist
        if hit.sim() > 0.6:
            t.mgetOrPut(getRssTitle(hit.id), @[]).add(hit)
     
    for title, hits in t:
        # sanitize
        for hit in hits:
            let description = getRssDescription(hit.id)
            let title = getRssTitle(hit.id)
            if sanitize(description):
                echo &"    EMPTY id:{hit.id} {hit.sim()} {title}, {description}"
                delete(ix, hit)

        var lastDescription = ""
        for hit in hits:
            let description = getRssDescription(hit.id)
            let title = getRssTitle(hit.id)
            if lastDescription != "" and description == lastDescription:
                echo &"     REDUN id:{hit.id} {hit.sim()} {title}, {description}"
                #delete(ix, hit)
            lastDescription = description



when isMainModule:
    var ix = openHnsw(HnswParams(global: "^HNSWArticles"))

    for (cnt, idxref, title) in enumerate(RSSItemIter(reverse=true)):
        echo cnt, " ", title
        findAndRemoveDuplicates(ix, title, k=10)
