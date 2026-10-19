Remove unused neighbor-pruning bookkeeping. selectNeighbors records rejected candidates in pruned even though both call sites use the default keepPruned = false. Since the function is private and neither caller requests the fallback, removing keepPruned, pruned, and the fill-back loop would simplify the function and avoid storing rejected indices during graph construction.

Consider reusing construction scratch buffers. selectNeighbors also creates an index array and view array on each call, and may allocate owned vectors while selecting. If insertion or repair profiling shows allocator overhead is significant, reusable scratch storage could reduce churn. This is a more involved change because vector views have lifetime constraints; I’d profile before taking it on.

