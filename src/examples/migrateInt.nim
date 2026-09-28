import ../ydbhnsw
import std/[strutils]

proc convertInt64ToUint32(global: string) =
    for (k,v) in QueryItr @global.sv:
        let ints64 = unpackInts(v)
        var uints32 = newSeq[uint32](ints64.len)
        for i in 0 ..< ints64.len:
            uints32[i] = cast[uint32](ints64[i])
        let v32 = packInts32(uints32)
        
        echo k,", ", v.len, ", ", ints64.len, " ", uints32.len, " ", v32.len
        for i in 0 ..< ints64.len:
            echo ints64[i] , " ", uints32[i]
            #assert ints64[i] == uints32[i].int

        Set: @global(k) = v32


proc list32(global: string, maxcnt: uint32 = uint32.high) =
    let links = global & "LINKS"
    let meta = global & "META"
    let metacnt = Get @meta("count").uint32 # the current count of the vector index

    var cnt = maxcnt
    for (k,v) in QueryItr @links.sv:
        let uints32 = unpackInts32(v)
        echo k,", ", v.len, ", ", uints32.len
        for i in 0 ..< uints32.len:  
            if uints32[i].uint32 > metacnt.uint32:
                echo "ERROR: Invalid value ", uints32[i], " maxcnt=", maxcnt
        dec cnt
        if cnt <= 0: break


#convertInt64ToUint32("^HNSWArticlesLINKS")
list32("^HNSWArticles")

