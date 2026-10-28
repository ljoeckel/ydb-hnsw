#import yottadb
import rsstypes
import std/[strformat]
import ydbhnsw
import hnsw_common


proc findDuplicates(ix: HnswIndex, self: int, title: string, vec: seq[float32], k = 10, minSim = MinSim, remove = false) =
    let hits = ix.search(vec, k = k, ef = max(k * 4, ix.params.efSearch))

    var seen = initHashSet[string]()
    seen.incl getRssDescription(self)

    var found = 0
    for hit in hits:
        if hit.id == self:
            continue                    # the article itself, similarity ~= 1.0
        if hit.sim() < minSim:
            continue                    # nearest, but not similar
        if found == 0:
            echo title
            inc similar
        inc found

        let description = getRssDescription(hit.id)
        echo &"   {hit.sim():.4f} {hit.id} : {getRssTitle(hit.id)}"
        #echo &"                     - {description}"

        if sanitize(description):
            inc empty
            if remove: delete(ix, hit)
        elif description in seen:
            inc duplicates
            if remove: delete(ix, hit)
        else:
            seen.incl description



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
        


if isMainModule:
    scanTitleDescription()