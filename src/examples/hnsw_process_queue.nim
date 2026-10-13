import std/[cmdline, os, strformat, times]
import ../ydbhnsw
import types
import ydbutils

const
    gbl = "^HNSWArticles"
    gblQueue = gbl & "QUEUE"


proc claimNext(): string =
    ## Claim the first item nobody holds yet and return its idxref ("" when there
    ## is nothing to do).
    ##
    ## The claim is a `$INCREMENT` on the item's "work" node: it is atomic across
    ## processes, so when two processors race for the same item exactly one sees
    ## the marker go 0 -> 1 and wins, while the other sees 2 and moves on. No
    ## lock is involved on purpose - the DSL caps a lock timeout at ~2.1 s and the
    ## scan can have to walk a long run of already-claimed items, so a lock here
    ## just moves the failure from "empty workItem" to YDB_LOCK_TIMEOUT.
    ##
    ## "vec" is the last field the producer writes, so testing for it means we
    ## only ever claim a fully written item; a half-written one is simply skipped
    ## this pass and picked up on a later one.
    var idxref = Order @gblQueue("")
    while idxref.len > 0:
        if 0 != Data @gblQueue(idxref, "vec"):
            if 1 == Increment @gblQueue(idxref, "work"):
                return idxref
        idxref = Order @gblQueue(idxref)


proc processor() =
    let params = hnswParams(gbl, cacheVectors=true)
    let ix = openHnsw(params)
    echo $ix.params
    var cnt = 0
    var avgDurationInMs = 0
    var t1 = now()

    while true:
        let idxref = claimNext()
        if idxref.len == 0:
            echo "Wait for work"
            sleep(1000)
            continue

        # let vec = ydb_get(ix.params.globalQueue, @[idxref])
        # if vec.len == 0:
        #     echo "WARN: giving back empty idxref:", idxref
        #     Killnode @gblQueue(idxref, "work")
        # else:
        #     discard ix.add(unpackFloats(vec), idxref)
        #     Kill @gblQueue(idxref)      # done - drop the whole item
        #     inc cnt
        #     if cnt mod 100 == 0: echo cnt
            
        let workItem = loadObject[WorkItem](idxref, globalName=gblQueue)
        if workItem.idxref == "":
            # Rare: the item was deleted between the scan and our claim, so the
            # `$INCREMENT` rebuilt a lone "work" node. Give the claim back; the
            # scan now skips the item because it has no "vec".
            echo "WARN: giving back empty idxref:", idxref
            Killnode @gblQueue(idxref, "work")
        else:
            let vec = unpackFloats(workItem.vec)
            discard ix.add(vec, workItem.idxref)
            Kill @gblQueue(idxref)      # done - drop the whole item
            inc cnt
            if cnt mod 100 == 0:
                let t2 = (now() - t1).inMilliseconds
                avgDurationInMs += t2
                t1 = now()
                echo $t1, " cnt:", cnt, " avg. duration (last 100) for 1 : ", t2 / 100, "ms.  sliding avg. last ", cnt, " adds:", avgDurationInMs / cnt, " ms."




proc starter(count: int) =
    echo "Launching ", count, " processes"
    for i in 1..count:
        discard job("./hnsw_process_queue", @["--processor"])
    

proc check() =
    for cnt in QueryItr @gblQueue.count:
        echo fmt"{gblQueue} cnt={cnt}"



if isMainModule:
    for param in commandLineParams():
        if param == "--check":
            check()
        elif param == "--processor":
            processor()
        else:
            echo "Valid commands are:"
            echo "  --processor           Run one processor in foreground"
            echo "  --theads=n            Start n threads"
            echo "  --check               Check current status"
            quit(0)