
import std/[strformat, times]
import ydbhnsw
import hnsw_common

when isMainModule:
    var params = getHnswParams()
    let ix: HnswIndex = openHnsw(params)

    while true:
        stdout.write "Search for: "    
        var searchFor = readLine(stdin)
        if searchFor == "": quit(0)
        
        let t0 = getTime()
        let vec = embed(@[hnswNormalize(searchFor)])[0]
        let t1 = getTime()
        let hits = ix.search(vec, k=20)
        let t2 = getTime()

        for hit in hits:
            echo &"  {hit.sim()} {hit.id} {getRssTitle(hit.id)}"
            echo &"     {getRssDescription(hit.id)}"

        let delta1 = (t1 - t0)
        let delta2 = (t2 - t1)
        let embedTime = if delta1.inMilliseconds > 0: &"{delta1.inMilliseconds} ms." else: &"{delta1.inMicroseconds} µs."
        let searchTime = if delta2.inMilliseconds > 0: &"{delta1.inMilliseconds} ms." else: &"{delta2.inMicroseconds} µs."
        echo &"\nEmbed in {embedTime} Search in {searchTime}"