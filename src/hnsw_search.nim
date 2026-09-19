import std/[os, strformat, strutils, tables, enumerate]
import std/[htmlparser, xmltree]

import bert_nim     # nimpy glue: embed(); its own demo stays dormant
import hnsw
import rss_bridge
import yottadb
import rsstypes


const Index = "^HNSWArticles"

proc getRssRef(id: int): seq[string] =
    let rssref = Get ^HNSWArticlesKEY(id)
    if rssref.len > 0 and rssref.contains(","):
        rssref.split(",")
    else:
        @[]

proc getTitle(id: int): string =
    let rssref = getRssRef(id)
    let title = Get ^RSSItem(rssref, "title")
    return hnswNormalize(title)


proc getDescription(id: int): string = 
    let rssRef = getRssRef(id)
    let s = Get ^RSSItem(rssref, "description")
    let description = s.parseHtml().innerText
    return hnswNormalize(description)


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
    else:
        echo &"Removed {hit.id} from HNSW index"

    # Remove from YottaDB
    deleteObject[RSSItem](rssref)
    echo &"Removed {rssref} from YDB"


proc findAndRemoveDuplicates(ix: HnswIndex, headline: string, k = 5) =
    var t = newTable[string, seq[Hit]]()
    let title = hnswNormalize(headline)
    let q = embed(@[title])[0]

    # collect articles that seams are related
    for hit in ix.search(q, k = k):
        let sim = 1.0'f32 - hit.dist
        if sim > 0.6:
            t.mgetOrPut(getTitle(hit.id), @[]).add(hit)
     
    for title, hits in t:
        if hits.len > 1:
            # sanitize
            for hit in hits:
                let description = getDescription(hit.id)
                if sanitize(description):
                    echo &"EMPTY       id:{hit.id} {description}"
                    delete(ix, hit)

            var lastDescription = ""
            for hit in hits:
                let description = getDescription(hit.id)
                if lastDescription != "" and description == lastDescription:
                    echo &"REDUNDANT   id:{hit.id} {description}"
                    delete(ix, hit)
                lastDescription = description



when isMainModule:
    var ix = openHnsw(Index, M = 16, efConstruction = 200, efSearch = 64)

    # let t = "das naechste gaspreis hoch bei equinor klingelt die kasse weiter"
    # let tds = findDuplicates(ix, t)

    # for hit in tds:
    #     if delete(ix, hit):
    #         echo &"Removed {hit.id} from HNSW and YottaDB"
    #     else:
    #         echo &"Could not remove {hit.id} from HNSW and YottaDB"

    

    for (cnt, idxref, title) in enumerate(RSSItemIter()):
        findAndRemoveDuplicates(ix, title, k=10)
        if cnt mod 100 == 0:
            echo cnt, " ", title
            updateDBStats("hnsw_clean")

    updateDBStats("hnsw_clean")
