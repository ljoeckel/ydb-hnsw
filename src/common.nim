import std/strutils
import std/[htmlparser, xmltree]

import yottadb
import hnsw

proc getRssRef*(id: int): seq[string] =
    let rssref = Get ^HNSWArticlesKEY(id)
    if rssref.len > 0 and rssref.contains(","):
        rssref.split(",")
    else:
        @[]

proc getRssTitle*(id: int): string =
    let rssref = getRssRef(id)
    let title = Get ^RSSItem(rssref, "title")
    return hnswNormalize(title)


proc getRssDescription*(id: int): string = 
    let rssRef = getRssRef(id)
    let s = Get ^RSSItem(rssref, "description")
    let description = s.parseHtml().innerText
    return hnswNormalize(description)
