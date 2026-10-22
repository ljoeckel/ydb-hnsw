## Article records for the HNSW web service.
##

import std/strutils
import yottadb
import hnsw_common

type
  Article* = object
    idxref*: string       ## caller's id for the article, unique per index
    title*: string
    description*: string
    sim*: float


proc articleSet*(index: string, id: int, article: Article) =
  echo "no implemented"
#   let node = $id
#   Set: @records(node, "idxref") = article.idxref
#   Set: @records(node, "title") = article.title
#   Set: @records(node, "description") = article.description
#   Set: @ids(article.idxref) = node

proc articleGet*(id: int): Article =
  let idxref = getRssRef(id)
  result.idxref = idxref.join(",")
  result.title = getRssTitle(idxref)
  result.description = getRssDescription(idxref)

proc articleUnset*(index: string, previousId: int) =
    echo "not implemented"

proc articleDrop*(index: string) =
  echo "not implemented"
  ## Delete every article record of `index` - the `--reset` counterpart of
  ## `dropIndex`, which only knows the vector globals.
#   let (records, ids) = articleGlobals(index)
#   Kill: @records
#   Kill: @ids

