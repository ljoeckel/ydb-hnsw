import std/[strformat, strutils, sets, enumerate]

import bert_nim     # nimpy glue: embed(); its own demo stays dormant
import hnsw
import rss_bridge
import common       # getRssRef / getRssTitle / getRssDescription
import yottadb
import rsstypes

const BatchSize = 128
const MinDescriptionLen = 35

const MinSim = 0.90'f32
    ## Similarity a neighbour has to reach to be reported as related, and the
    ## shortlist the duplicate test runs over. `search` returns the k *nearest*,
    ## never the k most similar, so without this "no results" and "no similar
    ## results" are the same thing. 0.6 - the value that used to sit in the loop
    ## as a commented-out filter - is far too low to mean anything on these
    ## MiniLM vectors.

## Report counters. `queries` is how many articles were actually looked at and
## `similar` how many had at least one neighbour above `MinSim`; together they
## say whether the search is finding anything at all. `skipped` counts articles
## an earlier iteration already deleted as somebody else's duplicate - the
## article iterator walks ^RSSItem and still hands them out.
var empty, duplicates, skipped, queries, similar = 0

proc sanitize(description: string): bool =
    if description.len < MinDescriptionLen: return true
    if description == "mehr": return true
    false


proc delete(ix: HnswIndex, hit: Hit) =
    echo "delete ", hit.id
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


proc findDuplicates(ix: HnswIndex, key, title: string, vec: seq[float32],
                    k = 10, minSim = MinSim, remove = false) =
    ## Report one article and the neighbours close to it.
    ##
    ## The query *is* the article, so the header a group prints under is what the
    ## lines below it are similar to. The previous version keyed its table on
    ## `getRssTitle(hit.id)` and then re-printed that same title for every hit, so
    ## each hit landed in a group named after itself: a group could only ever
    ## collect nodes carrying an *identical* title. Related articles with
    ## different wording - the very thing the report exists for - were spread
    ## over singleton groups, which is what made it look like "not enough similar
    ## results".
    let self = ix.lookup(key)
    if self < 0:
        # Already deleted as somebody else's duplicate on an earlier iteration.
        inc skipped
        return
    inc queries

    # `ef` is widened with `k`: the index default is tuned around k ~ 5, and
    # asking for more neighbours without a wider candidate window would just
    # return more of the same few.
    let hits = ix.search(vec, k = k, ef = max(k * 4, ix.params.efSearch))

    # Similarity only *shortlists*. What makes two articles duplicates is that
    # they carry the same description - the syndication case - or none at all.
    # Seeding with the query's own description is what `lastDescription` meant to
    # do; comparing against the previous hit *inside one title bucket* could only
    # ever fire when that bucket already held two identical titles.
    var seen = initHashSet[string]()
    seen.incl getRssDescription(self)

    var found = 0
    for hit in hits:
        if hit.id == self:
            continue                    # the article itself, similarity ~= 1.0
        let sim = hit.sim()
        if sim < minSim:
            continue                    # nearest, but not similar
        if found == 0:
            echo title
            inc similar
        inc found

        let description = getRssDescription(hit.id)
        echo &"   {sim:.4f} {hit.id} : {getRssTitle(hit.id)}"
        #echo &"                     - {description}"

        if sanitize(description):
            inc empty
            if remove: delete(ix, hit)
        elif description in seen:
            inc duplicates
            if remove: delete(ix, hit)
        else:
            seen.incl description


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
        #if updateVecEntry(key, vec): inc vectors

        findDuplicates(ix, key, title, vec, remove=true)
        if cnt mod params.batchSize == 0:
            echo &"findDuplicates cnt:{cnt}, queries:{queries}, similar:{similar}, " &
                 &"skipped:{skipped}, empty:{empty}, duplicates:{duplicates}"
            updateDBStats("hnsw_clean") 

        if cnt > 1000: break       
    
    echo "   Queries: ", queries
    echo "   Similar: ", similar
    echo "   Skipped: ", skipped
    echo "     Empty: ", empty
    echo "Duplicates: ", duplicates
    updateDBStats("hnsw_clean")