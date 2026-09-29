import std/[strformat, strutils, sets, enumerate]
import ../ydbhnsw
import hnsw_common
import rsstypes

const
    BatchSize = 128
    MinDescriptionLen = 35
    MinSim = 0.90'f32
    RemoveDups = false

## Report counters
var empty, duplicates, skipped, queries, similar = 0


proc sanitize(description: string): bool =
    if description.len < MinDescriptionLen: return true
    if description == "mehr": return true
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



when isMainModule:
    let params = hnswParams("^HNSWArticles", cacheVectors=true, batchSize=BatchSize)
    echo &"Opening the HNSW index with {params}"
    let ix = openHnsw(params)

    for (cnt, id, title, vec) in enumerate(batchedRSSItemIter(ix)):
        findDuplicates(ix, id, title, vec, remove=RemoveDups)
        inc queries

        if cnt mod params.batchSize == 0:
            echo &"findDuplicates cnt:{cnt}, queries:{queries}, similar:{similar}, " &
                 &"skipped:{skipped}, empty:{empty}, duplicates:{duplicates}"
            #updateDBStats("hnsw_clean") 

        #if cnt > BatchSize: break
    
    echo "   Queries: ", queries
    echo "   Similar: ", similar
    echo "   Skipped: ", skipped
    echo "     Empty: ", empty
    echo "Duplicates: ", duplicates
    #updateDBStats("hnsw_clean")