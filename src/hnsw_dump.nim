import yottadb
var cnt = 0
for (k, v) in QueryItr ^HNSWArticlesNODE.kv:
    echo k,"=",v.len
    inc cnt
    if cnt == 200: break