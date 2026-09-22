import std/[os, strformat, strutils, tables]
import std/[htmlparser, xmltree]

import bert_nim     # nimpy glue: embed(); its own demo stays dormant
import hnsw
import rss_bridge
import yottadb
import rsstypes


var empty, duplicates = 0

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

    # Remove from YottaDB
    deleteObject[RSSItem](rssref)


proc findDuplicates(ix: HnswIndex, vec: seq[float32], k = 5, remove = false) =
    var t = newTable[string, seq[Hit]]()

    # collect articles that seams are related
    for hit in ix.search(vec, k = k):
        if hit.sim() > 0.6:
            t.mgetOrPut(getTitle(hit.id), @[]).add(hit)

    for title, hits in t:
        # sanitize
        for hit in hits:
            let description = getDescription(hit.id)
            if remove and sanitize(description):
                inc empty
                echo &"EMPTY       id:{hit.id}, rssRef:{getRssRef(hit.id)}  {title}"
                delete(ix, hit)
                discard

        var lastDescription = ""
        for hit in hits:
            let description = getDescription(hit.id)
            if remove and lastDescription != "" and description == lastDescription:
                echo &"REDUNDANT   id:{hit.id}, rssRef:{getRssRef(hit.id)} {title}"
                echo &"            {description}"
                inc duplicates
                delete(ix, hit)
            lastDescription = description



when isMainModule:
    #var ix = openHnsw(HnswParams(global: "^HNSWArticlesQ", quant: vqInt8))
    #var ix = openHnsw(Index, M = 16, efConstruction = 200, efSearch = 64)

    #let t = hnswNormalize("Anschläge auf Umspannwerke: Verband: Brauchen Backup-System für Notfälle im Stromnetz")
    #let t = hnswNormalize("Sabotage - Polizei findet zwölf Sprengsätze an Stromtrassen in Sachsen - Fahndung mit Foto nach Tatverdächtigem aus NRW")
    #findDuplicates(ix, t)

    # for hit in tds:
    #     if delete(ix, hit):
    #         echo &"Removed {hit.id} from HNSW and YottaDB"
    #     else:
    #         echo &"Could not remove {hit.id} from HNSW and YottaDB"

    

    # Embedding one headline per call costs ~12 ms of pure per-call overhead
    # (Python dispatch, tokenizer, kernel launches) and no number of CPU
    # threads changes that - SBERT_THREADS only parallelises the matrix work
    # *inside* one call, which is negligible for a batch of 1. The same model
    # does ~0.3 ms per text in batches, so collect the titles first and embed
    # them in bulks. Measured on this box: 2000 single calls 23.3 s vs 3.6 s
    # for the batched version, i.e. most of the run time of this program.
    const BatchSize = 128
    echo "Scanning Articles"

    var titles: seq[string]
    for (_, title) in RSSItemIter(5000, reverse=true):
        titles.add(hnswNormalize(title))
    echo &"Have {titles.len} titles"

    let params = HnswParams(global: "^HNSWArticles")
    #let params = HnswParams(global: "^HNSWArticlesQ", quant: vqInt8)
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