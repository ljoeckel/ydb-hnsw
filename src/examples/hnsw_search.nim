import std/[strformat, wordwrap, parseopt, strutils, sets, enumerate]
import ../ydbhnsw
import hnsw_common
import rsstypes

const
    MinDescriptionLen = 35
    MinSim = 0.90'f32
    RemoveDups = false

    Usage = """
    usage: <program> [option=value]...

    --[HnswOption]      Will be processed by HnswOption
    --id, -i            Lookup by Hnsw Index id
    """

type
    AppOptions = object
        id: int


## counters
var idxid, empty, duplicateTitles, duplicateDescriptions, queries, similar = 0


proc getAppOptions(): AppOptions =
    result = AppOptions()

    for kind, key, val in getopt():  
      case kind
      of cmdEnd: break
      of cmdShortOption, cmdLongOption:
        case toLowerAscii(key)
        of "id", "i": result.id = parseInt(val)
        of "help", "h": echo Usage
        else: discard
      else:
        discard


proc logCounters() =
    echo &"LastId: {idxid}, Queries: {queries}, Similar: {similar} Empty: {empty}, dupTitles: {duplicateTitles}, dupDesc: {duplicateDescriptions}"


proc sanitize(description: string): bool =
    if description.len < MinDescriptionLen: return true
    if description == "mehr": return true
    false


proc deleteItem(ix: HnswIndex, id: int) =
    # Get rssref to access ydb
    let rssref = getRssRef(id)
    if rssref.len == 0:
        echo &"Could not find storage reference {rssref}"
        return

    # Remove form HNSW vector db
    let rc = ix.delete(id)
    if not rc:
        echo &"Could not delete {id} from HNSW index"
        return

    # Remove from YottaDB
    deleteObject[RSSItem](rssref)
    if ix.params.verbose:
        echo &"RSSItem {rssref} and HNSW index entry {id} deleted"


proc loadRSSItem(idxref: string): RSSItem =
    result = loadObject[RSSItem](idxref.split(","))    


proc searchDuplicates(ix: HnswIndex, self: int, k: int = 5, ef = max(k * 4, ix.params.efSearch)) =
    var info: seq[string]

    let gblNode = ix.params.globalNode
    let selfTitle = getRSSTitle(self)
    let selfDescription = getRssDescription(self)

    let vec = Get @gblNode(self)
    let hits = ix.searchBlob(vec, k=k, ef=ef)
    for hit in hits:
        if hit.id == self: continue
        if hit.sim() < MinSim: continue
        let title = getRSSTitle(hit.id)
        let description = getRSSDescription(hit.id)

        if title == selfTitle:
            inc duplicateTitles
            if selfDescription == description:
                inc duplicateDescriptions
                deleteItem(ix, hit.id) # delete real duplicate              
            else:
                info.add(&"{hit.id:>8} {title} {hit.sim():.4f}")
                info.add(description.wrapWords(maxLineWidth = 80).indent(9))

    if info.len > 0 and ix.params.verbose:
        echo &"{self:<8} {selfTitle}"
        echo selfDescription.wrapWords(maxLineWidth = 80).indent(9)
        for line in info:
            echo line


when isMainModule:
    let options = getAppOptions()
    var params = getHnswParams()

    let ix = openHnsw(params)
    let gblKey = ix.params.globalKey

    if options.id > 0:
        ix.searchDuplicates(options.id)
        quit(0)

    for cnt, id, idxref in enumerate(OrderItr @gblKey.kv.reverse):
        if id.contains(","): continue
        
        idxid = parseInt(id)
        let title = getRSSTitle(idxref)
        let description = getRssDescription(idxref)

        if sanitize(description):
            inc empty
            deleteItem(ix, idxid)
            continue
         
        ix.searchDuplicates(idxid)

        inc queries

        if cnt mod params.batchSize == 0:
            logCounters()
    
    logCounters()