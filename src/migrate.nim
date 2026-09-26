import ydbhnsw

for keys in QueryItr ^HNSWArticlesNODE.keys:
    echo keys
    # if keys[1] == "links":
    #     let value = Get ^HNSWArticlesNODE(keys)
    #     Set: ^HNSWArticlesLINKS(keys[0], keys[2]) = value
    #     Kill ^HNSWArticlesNODE(keys)
    #     echo keys

    # if keys[1] == "level":
    #     let value = Get ^HNSWArticlesNODE(keys)
    #     Set: ^HNSWArticlesLEVEL(keys[0]) = value
    #     Kill ^HNSWArticlesNODE(keys)
    #     echo keys    

    # if keys[1] == "vec":
    #     let value = Get ^HNSWArticlesNODE(keys)
    #     Set: ^HNSWArticlesNODE(keys[0]) = value
    #     Kill ^HNSWArticlesNODE(keys)
    #     echo keys        