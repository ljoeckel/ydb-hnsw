import std/[os, strformat, strutils, tables, enumerate]
import std/[htmlparser, xmltree]

import bert_nim     # nimpy glue: embed(); its own demo stays dormant
import hnsw
import rss_bridge
import yottadb
import types


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


proc showHit(hit: Hit) =
    let title = getTitle(hit.id)
    let sim = 1.0'f32 - hit.dist
    echo &"  {hit.id:8d}  sim={sim:.4f}  {title}"


proc findDuplicates(ix: HnswIndex, label: string, q: seq[float32], k = 5) =
    var t = newTable[string, seq[Hit]]()

    for hit in ix.search(q, k = k):
        let sim = 1.0'f32 - hit.dist
        if sim > 0.5:
            t.mgetOrPut(getTitle(hit.id), @[]).add(hit)
 
    
    for title, hits in t:
        if hits.len > 1:
            echo title
            for hit in hits:
                let rssRef = getRssRef(hit.id)
                let rssItem = loadObject[RSSItem](rssRef)
                let opthtml = getOption(rssItem.description)
                let description = opthtml.parseHtml().innerText
                echo "  id:", hit.id, " rssRef:", rssRef, "  ", description


proc showHits(ix: HnswIndex, label: string, q: seq[float32], k = 5) =
  echo label
  for hit in ix.search(q, k = k):
    showHit(hit)



when isMainModule:
    var ix = openHnsw(Index, M = 16, efConstruction = 200, efSearch = 64)

    for (cnt, idxref, title) in enumerate(RSSItemIter(25)):
        let nTitle = hnswNormalize(title)
        findDuplicates(ix, title, embed(@[nTitle])[0] , k=10)
        #if cnt mod 1000 == 0:
        #    updateDBStats("hnsw_search")

    updateDBStats("hnsw_search")

