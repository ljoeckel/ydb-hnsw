import std/strutils
import std/[htmlparser, xmltree]
import ../ydbhnsw


iterator RSSItemIter*(maxitems: int = int.high, reverse: bool = false): (string, string) =
    proc getItem(idxref: seq[string]): (string, string) =
        #let item = loadObject[RSSItem](idxref)
        #let title = getOption(item.title)
        let title = Get ^RSSItem(idxref, "title")
        return (idxref.join(","), title)

    var cnt = maxitems

    if reverse:
        for item in OrderItr ^RSSItem.reverse:
            for idxref in OrderItr ^RSSItem(item, "").keys:
                yield getItem(idxref)
                dec cnt
                if cnt <= 0: break

            if cnt <= 0: break
    else:
        for item in OrderItr ^RSSItem:
            for idxref in OrderItr ^RSSItem(item, "").keys:
                yield getItem(idxref)
                dec cnt
                if cnt <= 0: break

            if cnt <= 0: break


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


iterator batchedRSSItemIter*(ix: HnswIndex): (int, string, seq[float32]) =
    let batchSize = ix.params.batchSize
    var titles = newSeqOfCap[string](batchSize)
    var keys = newSeqOfCap[int](batchSize)
    var cnt = 0

    for (key, title) in RSSItemIter(reverse=true):
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

proc getRssTitle*(rssref: string): string = 
    if rssref.len > 0 and rssref.contains(","):
        let subs = rssref.split(",")
        let title = Get ^RSSItem(subs, "title")
        return hnswNormalize(title)
    else:
        return ""

proc getRssTitle*(id: int): string =
    let rssref = getRssRef(id)
    let title = Get ^RSSItem(rssref, "title")
    return hnswNormalize(title)


proc getRssDescription*(id: int): string = 
    let rssRef = getRssRef(id)
    let s = Get ^RSSItem(rssref, "description")
    let description = s.parseHtml().innerText
    return hnswNormalize(description)
