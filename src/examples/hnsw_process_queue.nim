import std/[cmdline, os, strformat, times]
import ../ydbhnsw
import types
import ydbutils

const
    gbl = "^HNSWArticles"
    gblQueue = gbl & "QUEUE"


proc claimNext(): string =
    var idxref = Order @gblQueue("")
    while idxref.len > 0:
        if 0 != Data @gblQueue(idxref, "vec"):
            if 1 == Increment @gblQueue(idxref, "work"):
                return idxref
        idxref = Order @gblQueue(idxref)


proc processor() =
    let params = getHnswParams()
    let ix = openHnsw(params)
    echo $ix.params
    
    var cnt = 0
    var avgDurationInMs = 0
    var t1 = now()

    let started = epochTime()

    proc logInfo(cnt: int) =
        let ips = cnt.float / (epochTime() - started)
        echo &"Processed: {cnt} with {ips:5.2f} nodes/s"

    while true:
        let idxref = claimNext()
        if idxref.len == 0:
            echo "Wait for work"
            sleep(250)
            continue

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
                logInfo(cnt)
                let t2 = (now() - t1).inMilliseconds
                avgDurationInMs += t2
                t1 = now()
                echo $t1, " cnt:", cnt, " avg. duration (last 100) for 1 : ", t2 / 100, "ms.  sliding avg. last ", cnt, " adds:", avgDurationInMs / cnt, " ms."




proc starter(count: int) =
    echo "Launching ", count, " processes"
    for i in 1..count:
        discard job("./hnsw_process_queue", @["--processor"])
    

if isMainModule:
    processor()
