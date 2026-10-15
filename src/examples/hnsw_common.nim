import std/strutils
import std/strformat
import std/[htmlparser, xmltree]
import std/times
import ../ydbhnsw
import rsstypes


iterator RSSItemIter*(reverse: bool = false): RSSItem =
    if reverse:
        for rssId in OrderItr ^RSSItem.reverse:
            for idxref in OrderItr ^RSSItem(rssId, "").keys:
                yield loadObject[RSSItem](idxref)
    else:
        for rssId in OrderItr ^RSSItem:
            for idxref in OrderItr ^RSSItem(rssId, "").keys:
                yield loadObject[RSSItem](idxref)


iterator RSSTitleIter*(reverse: bool = false): (string, string) =
    proc getItem(idxref: seq[string]): (string, string) =
        let title = Get ^RSSItem(idxref, "title")
        return (idxref.join(","), title)

    if reverse:
        for rssId in OrderItr ^RSSItem.reverse:
            for idxref in OrderItr ^RSSItem(rssId, "").keys:
                yield getItem(idxref)
    else:
        for rssId in OrderItr ^RSSItem:
            for idxref in OrderItr ^RSSItem(rssId, "").keys:
                yield getItem(idxref)


iterator processTitles(titles: seq[string], keys: seq[int], cnt: int, batchSize: int): (int, string, seq[float32]) =
    ## Embed one batch of titles and yield the vectors.
    ##
    ## Deliberately top-level, taking `titles` as a parameter: as a nested
    ## iterator that *captured* `xIter`'s locals it became a closure iterator,
    ## and Nim 2.2.12 fails to give that closure's environment a location when
    ## the enclosing inline iterator also inlines `RSSItemIter` (which contains
    ## `if reverse: ...yield... else: ...yield...`). The result is
    ## `internal error: expr: var not init :env_<id>` at the `for ... in
    ## processTitles()` site.
    let stop = min(batchSize, titles.len)
    let flat = embedFlat(titles)
    let dim = flat.len div stop
    for i in 0 ..< stop:
        let vec = flat[i * dim ..< (i + 1) * dim]
        yield (keys[i], titles[i], vec)

iterator processTitles(keys: seq[string], titles: seq[string], descriptions: seq[string], cnt: int, batchSize: int): (string, string, seq[float32], seq[float32]) =
    ## Embed one batch of titles and yield the vectors.
    ##
    ## Deliberately top-level, taking `titles` as a parameter: as a nested
    ## iterator that *captured* `xIter`'s locals it became a closure iterator,
    ## and Nim 2.2.12 fails to give that closure's environment a location when
    ## the enclosing inline iterator also inlines `RSSItemIter` (which contains
    ## `if reverse: ...yield... else: ...yield...`). The result is
    ## `internal error: expr: var not init :env_<id>` at the `for ... in
    ## processTitles()` site.
    let stop = min(batchSize, titles.len)
    var t1 = now()
    let flatTitles = embedFlat(titles)
    echo "GPU embed ", titles.len, " titles in ", (now() - t1)
    t1 = now()
    echo "GPU embed ", descriptions.len, " descriptions in ", (now() - t1)
    let flatDescriptions = embedFlat(descriptions)

    let dim = flatTitles.len div stop
    for i in 0 ..< stop:
        let vec = flatTitles[i * dim ..< (i + 1) * dim]
        let vec2 = flatDescriptions[i * dim ..< (i + 1) * dim]
        yield (keys[i], titles[i], vec, vec2)


iterator batchedRSSNewTitleIter*(ix: HnswIndex): (string, string, seq[float32], seq[float32]) =
    let batchSize = ix.params.batchSize
    var titles = newSeqOfCap[string](batchSize)
    var descriptions = newSeqOfCap[string](batchSize)
    var keys = newSeqOfCap[string](batchSize)

    var cnt = 0
    var t1 = now()

    #for (idxref, title) in RSSTitleIter(reverse=true):
    #for (idxref, title) in RSSTitleIter():
    for item in RSSItemIter(reverse=true):
        let idxref = item.idxref
        if ix.lookup(idxref) > 0:  # Already in the vector-db
            continue

        keys.add(idxref)
        let title = getOption(item.title)
        titles.add(hnswNormalize(title))
        let description = getOption(item.description)
        descriptions.add(hnswNormalize(description))

        inc cnt
        if cnt mod batchSize == 0:
            for (idxref, title, vec, vec2) in processTitles(keys, titles, descriptions, cnt, batchSize):
                yield (idxref, title, vec, vec2)
            titles.setLen(0)
            descriptions.setLen(0)
            keys.setLen(0)

    if titles.len > 0:
        for (key, title, vec, vec2) in processTitles(titles, descriptions, keys, cnt, batchSize):
            yield (key, title, vec, vec2)



iterator batchedRSSTitleIter*(ix: HnswIndex): (int, string, seq[float32]) =
    let batchSize = ix.params.batchSize
    var titles = newSeqOfCap[string](batchSize)
    var keys = newSeqOfCap[int](batchSize)
    var cnt = 0

    for (key, title) in RSSTitleIter(reverse=true):
        let id = ix.lookup(key)
        if id < 0:
            # Not yet in the vector-db or deleted as somebody else's duplicate on an earlier iteration
            continue

        keys.add(id)
        titles.add(hnswNormalize(title))

        inc cnt
        if cnt mod batchSize == 0:
            for (key, title, vec) in processTitles(titles, keys, cnt, batchSize):
                yield (key, title, vec)
            titles.setLen(0)
            keys.setLen(0)

    if titles.len > 0:
        for (key, title, vec) in processTitles(titles, keys, cnt, batchSize):
            yield (key, title, vec)


proc getRssRef*(id: int): seq[string] =
    let rssref = Get ^HNSWArticlesKEY(id)
    if rssref.len > 0 and rssref.contains(","):
        rssref.split(",")
    else:
        @[]

proc getRssTitle*(subs: seq[string]): string = 
    let title = Get ^RSSItem(subs, "title")
    if title.len > 0:
        return hnswNormalize(title)
    else:
        return ""

proc getRssTitle*(rssref: string): string = 
    if rssref.len > 0 and rssref.contains(","):
        let subs = rssref.split(",")
        getRssTitle(subs)
    else:
        return ""

proc getRssTitle*(id: int): string =
    getRssTitle(getRssRef(id))


proc getRssDescription*(subs: seq[string]): string = 
    let s = Get ^RSSItem(subs, "description")
    let description = s.parseHtml().innerText
    return hnswNormalize(description)

proc getRssDescription*(rssref: string): string = 
    if rssref.len > 0 and rssref.contains(","):
        let subs = rssref.split(",")
        getRssDescription(subs)
    else:
        return ""

proc getRssDescription*(id: int): string = 
    getRssDescription(getRssRef(id))
