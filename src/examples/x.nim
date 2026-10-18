#import yottadb
import rsstypes
import std/[strformat]
import ydbhnsw
import hnsw_common

proc scanTitleDescription() =
    var params = hnswParams("^HNSWArticles", model = DefaultModel, M = 16, efConstruction = 200, efSearch = 256)
    let ix: HnswIndex = openHnsw(params)

    for rssid in OrderItr ^RSSItem:
        for articleId in OrderItr ^RSSItem(rssid,""):
            let item = loadObject[RSSItem](@[rssid, articleId])
            let title = hnswNormalize(getOption(item.title))
            let description = hnswNormalize(getOption(item.description))
            let vectors = embed(@[title, description])

            echo "title:", title
            for hit in ix.search(vectors[0]):
                echo &"    {hit.sim()} {hit.id} {getRssTitle(@[rssid, articleId])}"

            echo "description:", description
            for hit in ix.search(vectors[1]):
                echo &"    {hit.sim()} {hit.id} {getRssTitle(@[rssid, articleId])}"
        

proc scanLinks() =
    var params = getHnswParams(model="none")
    params.cacheVectors = false

    params.verbose = true
    let ix = openHnsw(params)
    let gbl = params.globalLinks
    for keys in QueryItr @gbl.keys:
        let id = keys[0]
        let level = keys[1]
        let blob = Get @gbl(keys)
        echo "id:", id, " level:", level, " blob:", blob.len

if isMainModule:
    scanLinks()