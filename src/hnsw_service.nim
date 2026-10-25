## HTTP/JSON service in front of the HNSW article index.
##
## Two endpoints (`/` answers with a status line), one model, one index:
##
##   POST /embed    {"idxref": "...", "title": "...", "description": "..."}
##                  -> 201 (new) / 200 (replaced) with the stored article
##
##   GET  /search?q=<text>&k=<n>
##   POST /search   {"query": "<text>", "k": <n>}
##                  -> {"query": ..., "count": n,
##                      "results": [{"idxref", "title", "description", "score"}]}
##
## Embedding is `title + ". " + description`, normalised exactly like the bulk
## importers (`hnswNormalize`): lower case, transliterated, single-spaced. That
## is what a query is normalised with too, so both sides meet on the same text.
##
## `/embed` is an upsert. A known `idxref` has its old vector deleted and is
## re-embedded, because the title or description may have changed and a vector
## is only a function of the text it came from. Ids are never recycled, so a
## replacement gets a fresh node id and leaves the old one as a hole in the
## graph's id space - `delete` already links the neighbours it strands.
##
## Threading: Mummy dispatches handlers to worker threads, but an `HnswIndex`
## keeps scratch buffers and caches per instance, so index access sits behind one
## lock (`svc.lock`). The embedding call runs before that lock is taken: the
## llama.cpp context has its own lock inside `llama_embed` (a context is not
## thread-safe), and a failed model call must not leave a half-done upsert
## behind.
##
## Persistence: the metadata comes from `article_store`, and vectors and records
## both go to YottaDB, so a restart reopens the index and keeps answering.
##
## Run:
##   nim c --threads:on --mm:orc -d:release src/hnsw_service.nim
##   HNSW_GGUF=~/models/gguf/<model>.gguf ./src/hnsw_service --port=8080
##
## The embedding model is a GGUF file loaded by llama.cpp; `HNSW_GGUF` or
## `HNSW_GGUF_DIR` name it (see `llama_embed`). No Python, venv or PYTHONPATH.
##
## Index options (`--index`, `--model`, `--quant`, `--cache`, ...) are the ones
## every tool in this repo takes; see `hnsw_options`. `--port` and `--bind` are
## the service's own and are filtered out before the index options are parsed.

import std/[json, locks, os, strutils, times]

import mummy

import ydbhnsw              # hnsw, hnsw_options, llama_embed, yottadb
import hnsw_common
import article_store

## Routing is done by hand instead of with `mummy/routers`. That router is part
## of the mummyDS fork and calls `getSignals(request)` - the Datastar SSE path -
## for *every* request before it dispatches, which parses the body as signals.
## A plain JSON API then gets its malformed bodies turned into a 500 before the
## handler can answer 400. Three exact paths are not worth that.

const
  DefaultPort = 8080
  DefaultK = 10
  MaxK = 100
  ServiceWorkerThreads = 4
    ## Mummy's default is `10 * cores`; every handler queues on one lock here,
    ## so a handful of threads is plenty and keeps the stack memory down.

type
  Service = object
    ix: HnswIndex
    lock: Lock

var svc: ptr Service
  ## The one index and the lock that guards it.
  ##
  ## A `ptr` and not a plain global object: Nim's thread analysis rejects access
  ## to a global holding GC'ed memory from a `{.gcsafe.}` handler, and it cannot
  ## see that `svc.lock` is what makes the access safe. A pointer is not GC'ed
  ## memory, so the analysis stays quiet - and the object behind it is
  ## process-lifetime state allocated once at startup, never freed, which is what
  ## the pointer says. Every access below happens under `svc.lock`.

template withLock(lock: Lock, body: untyped) =
  acquire(lock)
  try:
    body
  finally:
    release(lock)

template guarded(body: untyped) =
  ## What the handlers do with the index is safe, but Nim's thread analysis
  ## cannot see it: `hnsw` keeps its YottaDB keys in `{.threadvar.}`s, which the
  ## analysis counts as global GC'ed state, and `HnswIndex` carries scratch and
  ## cache buffers that one lock serialises. This block is that assertion - it
  ## covers the index access, not the request parsing around it.
  {.cast(gcsafe).}:
    body

var boundIndex {.threadvar.}: string
  ## The index this thread's `hnsw` YottaDB keys point at. `openHnsw` binds only
  ## the thread that opened the index, and Mummy runs handlers on its workers.

proc bindThread() =
  ## Bind this worker's YottaDB keys to the served index, once per thread.
  if boundIndex != svc.ix.params.global:
    bindGlobals(svc.ix)
    boundIndex = svc.ix.params.global

# ---------------------------------------------------------------------------
# request / response helpers
# ---------------------------------------------------------------------------

proc respondJson(request: Request, status: int, body: JsonNode) =
  var headers: HttpHeaders
  headers["Content-Type"] = "application/json; charset=utf-8"
  request.respond(status, headers, $body)

proc respondError(request: Request, status: int, message: string) =
  request.respondJson(status, %*{"error": message})

proc requiredString(node: JsonNode, name: string): string =
  ## The string field `name` of a request object, or a `ValueError` whose message
  ## is meant for the client.
  if not node.hasKey(name):
    raise newException(ValueError, "missing field '" & name & "'")
  if node[name].kind != JString:
    raise newException(ValueError, "field '" & name & "' must be a string")
  node[name].getStr()

proc parseArticle(body: string): Article =
  ## Validate an `/embed` body. `description` may be empty - the vector is then
  ## the title alone - but the field has to be there, like the other two.
  let node = parseJson(body)
  if node.kind != JObject:
    raise newException(ValueError, "request body must be a JSON object")
  result.idxref = requiredString(node, "idxref").strip()
  result.title = requiredString(node, "title").strip()
  result.description = requiredString(node, "description").strip()
  if result.idxref.len == 0:
    raise newException(ValueError, "field 'idxref' must not be empty")
  if result.title.len == 0:
    raise newException(ValueError, "field 'title' must not be empty")

func articleText(article: Article): string =
  ## What gets embedded: title and description as one passage, so a query can
  ## match either. Normalisation happens at the call site, next to the query's.
  if article.description.len == 0: article.title
  else: article.title & ". " & article.description

proc checkK(k: int): int =
  if k < 1 or k > MaxK:
    raise newException(ValueError, "k must be between 1 and " & $MaxK)
  k

proc parseIntOr(value: string, default: int): int =
  if value.len == 0:
    return default
  try:
    result = parseInt(value)
  except ValueError:
    raise newException(ValueError, "query parameter 'k' must be an integer")
  result = checkK(result)

proc parseIntOr(node: JsonNode, name: string, default: int): int =
  if not node.hasKey(name):
    return default
  case node[name].kind
  of JInt: 
    result = node[name].getInt()
  of JString:
    result = parseIntOr(node[name].getStr(), default)
  else:
    result = default
  result = checkK(result)


# ---------------------------------------------------------------------------
# handlers
# ---------------------------------------------------------------------------

proc embedHandler(request: Request) {.gcsafe.} =
  var article: Article
  try:
    article = parseArticle(request.body)
  except JsonParsingError as e:
    request.respondError(400, "invalid JSON body: " & e.msg)
    return
  except ValueError as e:
    request.respondError(400, e.msg)
    return

  # Embed first, outside the index lock: a failing model call must not have
  # already deleted the article it was replacing, and the index lock should not
  # be held while llama.cpp runs (the embedding context serialises itself).
  var vector: seq[float32]
  try:
    vector = embedOne(hnswNormalize(articleText(article)))
  except CatchableError as e:
    request.respondError(500, "embedding failed: " & e.msg)
    return

  var id = -1
  var replaced = false
  try:
    guarded:
      bindThread()
      withLock svc.lock:
        # Upsert: drop the old vector and its record first, so the new id is the
        # only one a search can return for this idxref.
        let previous = svc.ix.lookup(article.idxref)
        if previous >= 0:
          discard svc.ix.delete(previous)
          #discard articleUnset(svc.ix.params.global, previous)
          replaced = true
        id = svc.ix.add(vector, article.idxref)
        articleSet(svc.ix.params.global, id, article)
  except CatchableError as e:
    request.respondError(500, "embed failed: " & e.msg)
    return

  request.respondJson(if replaced: 200 else: 201, %*{
    "id": id,
    "idxref": article.idxref,
    "title": article.title,
    "description": article.description,
    "created": not replaced
  })

proc searchHandler(request: Request) {.gcsafe.} =
  var query: string
  var k = DefaultK
  try:
    if request.httpMethod == "POST":
      let node = parseJson(request.body)
      if node.kind != JObject:
        raise newException(ValueError, "request body must be a JSON object")
      query = requiredString(node, "query").strip()
      k = parseIntOr(node, "k", DefaultK)
    else:
      query = request.queryParams["q"].strip()
      k = parseIntOr(request.queryParams["k"], DefaultK)
    if query.len == 0:
      raise newException(ValueError, "missing query text: use ?q=<text> or {\"query\": \"<text>\"}")
  except JsonParsingError as e:
    request.respondError(400, "invalid JSON body: " & e.msg)
    return
  except ValueError as e:
    request.respondError(400, e.msg)
    return

  var vector: seq[float32]
  var t0 = getTime()
  try:
    vector = embedOne(hnswNormalize(query))
  except CatchableError as e:
    request.respondError(500, "embedding the query failed: " & e.msg)
    return
  let td = (getTime() - t0).inMilliseconds

  var rows: seq[Article]
  var vectorSearchTime = 0
  var dbProcessingTime = 0
  try:
    guarded:
      bindThread()
      withLock svc.lock:
        t0 = getTime()
        let hits = svc.ix.search(vector, k)
        vectorSearchTime = (getTime() - t0).inMicroseconds
        t0 = getTime()
        for hit in hits:
          var article = articleGet(hit.id)
          article.sim = hit.sim()
          rows.add (article)
        dbProcessingTime = (getTime() - t0).inMicroseconds
  except CatchableError as e:
    request.respondError(500, "search failed: " & e.msg)
    return

  var results = newJArray()
  for row in rows:
    results.add %*{
      "idxref": row.idxref,
      "title": row.title,
      "description": row.description,
      "sim": row.sim,
    }
  request.respondJson(200, %*{
    "query": query,
    "count": results.len,
    "embedTime (ms)": td,
    "vectorSearchTime (µs)": vectorSearchTime,
    "dbProcessingTime (µs)": dbProcessingTime,
    "results": results
  })

proc indexHandler(request: Request) {.gcsafe.} =
  ## Service description and index stats - enough to see that the process is up
  ## and which index/model it is answering for.
  var live = 0
  var width = 0
  withLock svc.lock:
    live = svc.ix.liveCount
    width = svc.ix.dim
  if width == 0:
    # An index that has no vectors yet records no dimension either; the model
    # knows it, so report that instead of a bare 0.
    try:
      width = dim()
    except CatchableError:
      discard
  request.respondJson(200, %*{
    "status": "ok",
    "index": svc.ix.params.global,
    "model": svc.ix.params.model,
    "dim": width,
    "live": live,
    "endpoints": {
      "GET /": "this description",
      "POST /embed": "{idxref, title, description} -> stored article",
      "GET /search": "?q=<text>&k=<1..100> -> matching articles",
      "POST /search": "{query, k} -> matching articles"
    }
  })

proc notFoundHandler(request: Request) {.gcsafe.} =
  request.respondError(404, "no such endpoint: " & request.path)

proc methodNotAllowedHandler(request: Request) {.gcsafe.} =
  request.respondError(405, request.httpMethod & " is not allowed on " & request.path)

proc dispatch(request: Request) {.gcsafe.} =
  ## The server's one request handler: exact paths, explicit methods.
  case request.path
  of "/":
    if request.httpMethod in ["GET", "HEAD"]: indexHandler(request)
    else: methodNotAllowedHandler(request)
  of "/embed":
    if request.httpMethod == "POST": embedHandler(request)
    else: methodNotAllowedHandler(request)
  of "/search":
    if request.httpMethod in ["GET", "POST"]: searchHandler(request)
    else: methodNotAllowedHandler(request)
  else:
    notFoundHandler(request)

# ---------------------------------------------------------------------------
# startup
# ---------------------------------------------------------------------------

type
  ServiceArgs = object
    port: Port
    address: string
    indexArgs: seq[string]   ## the arguments that are for `getHnswParams`

const Usage = """
usage: hnsw_service [option=value]... [--port=N] [--bind=ADDRESS]

  --port=N            TCP port to listen on (default 8080, env HNSW_PORT)
  --bind=ADDRESS      address to bind (default localhost, env HNSW_BIND;
                      use 0.0.0.0 to accept connections from other hosts)
  --help              this text

Everything else is an index option, identical to the other tools:
  --index=GLOBAL      index to serve (default ^HNSWArticles)
  --model=NAME        sentence-transformers model (default
                      paraphrase-multilingual-MiniLM-L12-v2; an existing index
                      keeps the model recorded in its META)
  --quant=MODE        none | int8 | int8fixed - must match the index
  --cache=BOOL        enable/disable the in-process vector and link caches
  --m, --efConstruction, --efSearch, --linkFmt, --scale, --verbose

The embedding model is a GGUF file run by llama.cpp in-process:
  HNSW_GGUF=~/models/gguf/paraphrase-multilingual-MiniLM-L12-v2-f16.gguf ./hnsw_service
  (or set HNSW_GGUF_DIR to the directory holding <model>.gguf)
"""

proc cliInt(raw: string, what: string): int =
  try:
    result = parseInt(raw)
  except ValueError:
    quit("invalid " & what & ": " & raw, 2)

proc serviceArgs(): ServiceArgs =
  ## `--port` / `--bind` are consumed here and kept *out* of `indexArgs`, so the
  ## shared option parser never sees a flag it would report as unknown.
  result.port = Port(cliInt(getEnv("HNSW_PORT", $DefaultPort), "--port"))
  result.address = getEnv("HNSW_BIND", "localhost")

  let argv = commandLineParams()
  var i = 0
  while i < argv.len:
    let arg = argv[i]
    case arg
    of "--help", "-h":
      echo Usage
      quit(0)
    of "--port", "--bind":
      if i + 1 >= argv.len:
        quit(arg & " needs a value", 2)
      let value = argv[i + 1]
      if arg == "--port":
        result.port = Port(cliInt(value, "--port"))
      else:
        result.address = value
      i += 2
      continue
    else:
      if arg.startsWith("--port="):
        result.port = Port(cliInt(arg.split('=', 1)[1], "--port"))
      elif arg.startsWith("--bind="):
        result.address = arg.split('=', 1)[1]
      else:
        result.indexArgs.add arg
    inc i

when isMainModule:
  let options = serviceArgs()

  # Index first: `openHnsw` loads the embedding model named by the index (META
  # wins for an existing one), through the loader `llama_embed` registers at
  # import time.
  let params = getHnswParams(args = options.indexArgs)

  svc = create(Service)
  svc.ix = openHnsw(params)
  initLock(svc.lock)

  let server = newServer(dispatch, workerThreads = ServiceWorkerThreads)
  echo "hnsw service on http://" & options.address & ":" & $options.port.int &
       ", index " & svc.ix.params.global & ", model " & svc.ix.params.model &
       ", " & $svc.ix.liveCount & " articles"
  server.serve(options.port, options.address)
