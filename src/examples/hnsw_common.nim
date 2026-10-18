import std/[strutils, times, htmlparser, xmltree]
import ydbhnsw
import rsstypes

type 
    IterResult* = object
        idxref*: string
        title*: string
        description*: string
        vec*: seq[float32]
        vec2*: seq[float32]


iterator RSSItemIter*(reverse: bool = false): RSSItem =
    if reverse:
        for rssId in OrderItr ^RSSItem.reverse:
            for idxref in OrderItr ^RSSItem(rssId, "").keys:
                yield loadObject[RSSItem](idxref)
    else:
        for rssId in OrderItr ^RSSItem:
            for idxref in OrderItr ^RSSItem(rssId, "").keys:
                yield loadObject[RSSItem](idxref)


iterator RSSTitleIter*(reverse: bool = false): IterResult =
    proc getItem(idxref: seq[string]): IterResult =
        result.title = Get ^RSSItem(idxref, "title")
        result.idxref = idxref.join(",")

    if reverse:
        for rssId in OrderItr ^RSSItem.reverse:
            for idxref in OrderItr ^RSSItem(rssId, "").keys:
                yield getItem(idxref)
    else:
        for rssId in OrderItr ^RSSItem:
            for idxref in OrderItr ^RSSItem(rssId, "").keys:
                yield getItem(idxref)


iterator processTitles(keys: seq[string], titles: seq[string], cnt: int, batchSize: int): IterResult =
    ## Embed one batch of titles and yield the vectors.
    let stop = min(batchSize, titles.len)
    let flat = embedFlat(titles)
    let dim = flat.len div stop
    for i in 0 ..< stop:
        var result = IterResult()
        result.vec = flat[i * dim ..< (i + 1) * dim]
        result.idxref = keys[i]
        result.title = titles[i]
        yield result

iterator processTitles(keys: seq[string], titles: seq[string], descriptions: seq[string], cnt: int, batchSize: int): IterResult =
    ## Embed one batch of titles and yield the vectors.
    var t1 = now()
    let flatTitles = embedFlat(titles)
    let flatDescriptions = embedFlat(descriptions)
    echo "GPU embeded ", titles.len, " title/descriptions in ", (now() - t1)

    let stop = min(batchSize, titles.len)
    let dim = flatTitles.len div stop
    for i in 0 ..< stop:
        var result = IterResult()
        result.idxref = keys[i]
        result.title = titles[i]
        result.vec = flatTitles[i * dim ..< (i + 1) * dim]
        result.vec2 = flatDescriptions[i * dim ..< (i + 1) * dim]
        yield result


iterator batchedRSSNewTitleIter*(ix: HnswIndex): IterResult =
    let batchSize = ix.params.batchSize
    var titles = newSeqOfCap[string](batchSize)
    var descriptions = newSeqOfCap[string](batchSize)
    var keys = newSeqOfCap[string](batchSize)
    var cnt = 0

    for item in RSSItemIter(reverse=true):
        let idxref = item.idxref
        if ix.lookup(idxref) > -1: continue  # Already in the vector-db

        keys.add(idxref)
        let title = getOption(item.title)
        titles.add(hnswNormalize(title))
        let description = getOption(item.description)
        descriptions.add(hnswNormalize(description))

        inc cnt
        if cnt mod batchSize == 0:
            for iterResult in processTitles(keys, titles, descriptions, cnt, batchSize):
                yield iterResult
            titles.setLen(0)
            descriptions.setLen(0)
            keys.setLen(0)

    if titles.len > 0:
        for iterResult in processTitles(keys, titles, descriptions, cnt, batchSize):
            yield iterResult



iterator batchedRSSTitleIter*(ix: HnswIndex): IterResult =
    let batchSize = ix.params.batchSize
    var titles = newSeqOfCap[string](batchSize)
    var keys = newSeqOfCap[string](batchSize)
    var cnt = 0

    for iterResult in RSSTitleIter(reverse=true):
        if ix.lookup(iterResult.idxref) > -1: continue # Already in the vector-db

        keys.add(iterResult.idxref)
        titles.add(hnswNormalize(iterResult.title))

        inc cnt
        if cnt mod batchSize == 0:
            for iterResult in processTitles(keys, titles, cnt, batchSize):
                yield iterResult
            titles.setLen(0)
            keys.setLen(0)

    if titles.len > 0:
        for iterResult in processTitles(keys, titles, cnt, batchSize):
            yield iterResult


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
