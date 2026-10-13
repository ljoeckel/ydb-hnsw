type
    WorkItem* = object
        idxref*: string
        #title*: string
        vec*: string   # packed float32 blob (see packFloats / unpackFloats)