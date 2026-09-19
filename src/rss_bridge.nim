import yottadb
import rsstypes

proc isInDB*(idxref: string): bool =
    0 != Data ^HNSWArticlesKEY(idxref)
    
iterator RSSItemIter*(maxitems: int = int.high, reverse: bool = false): (string, string) =
    proc getItem(idxref: seq[string]): (string, string) =
        let item = loadObject[RSSItem](idxref)
        let title = getOption(item.title)
        return (item.idxref, title)

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


if isMainModule:
    var cnt = 0
    for (k,v) in QueryItr ^HNSWARTICLES.kv:
        echo k,"=",v
        inc cnt
        #if cnt == 100: break


    # for (idxref, title) in RSSItemIter():
    #     echo idxref, " ", title
    #     break

    