Consider reusing construction scratch buffers. selectNeighbors also creates an index array and view array on each call, and may allocate owned vectors while selecting. If insertion or repair profiling shows allocator overhead is significant, reusable scratch storage could reduce churn. This is a more involved change because vector views have lifetime constraints; I’d profile before taking it on.

