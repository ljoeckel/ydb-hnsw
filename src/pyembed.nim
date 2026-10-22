## Thread-safe access to the embedded Python embedding model.
##
## `nimpy` initialises CPython on the thread that first touches it - here the
## main thread, while `bert_nim` is imported - and `Py_InitializeEx` leaves the
## GIL held by that thread forever. Mummy calls its handlers on worker threads,
## so without help the first `embedFlat` on a worker either deadlocks (a
## `PyGILState_Ensure` that the main thread never lets through) or, if no GIL is
## taken at all, runs Python without it.
##
## So the split is:
##
##   * the main thread loads the model (`openHnsw` -> `loadModel`), then hands
##     the GIL back once with `releaseGIL` (`PyEval_SaveThread`);
##   * every worker call goes through `embedOne`, which brackets the Python call
##     with `PyGILState_Ensure` / `PyGILState_Release`.
##
## The guard has to cover the *whole* call, not just the returned value: nimpy
## decrefs Python objects when Nim destroys its wrappers, so those destructors
## run while the call is on the stack.
##
## The GIL entry points are resolved from the libpython handle nimpy already
## dlopened (`pyLib.module`) because nimpy declares neither of them itself; its
## `PyLib` object is the only place that knows which library was loaded.

import std/dynlib

# Aliased: the module is `nimpy/py_lib`, and Nim is style-insensitive, so a bare
# `pyLib` would resolve to the module rather than to the variable of that name.
import nimpy/py_lib as pylib

import bert_nim

type
  PyGILState = cint

var
  gilEnsure: proc(): PyGILState {.cdecl.}
  gilRelease: proc(state: PyGILState) {.cdecl.}
  gilSaveThread: proc(): pointer {.cdecl.}
  gilReady: bool
  ## Written once by `initEmbedding` and only read afterwards. Nim's thread
  ## analysis treats a global of proc type as GC'ed state and cannot know that,
  ## which is what the `{.cast(gcsafe).}` blocks below assert.

proc initEmbedding*() =
  ## Resolve the GIL entry points, once.
  ##
  ## Needs the interpreter, which importing `bert_nim` has already initialised -
  ## so this must run after that import, never before it.
  if gilReady:
    return
  if pylib.pyLib == nil or pylib.pyLib.module == nil:
    raise newException(ValueError, "Python is not initialised; import bert_nim before initEmbedding()")
  {.cast(gcsafe).}:
    gilEnsure = cast[proc(): PyGILState {.cdecl.}](pylib.pyLib.module.symAddr("PyGILState_Ensure"))
    gilRelease = cast[proc(state: PyGILState) {.cdecl.}](pylib.pyLib.module.symAddr("PyGILState_Release"))
    gilSaveThread = cast[proc(): pointer {.cdecl.}](pylib.pyLib.module.symAddr("PyEval_SaveThread"))
    if gilEnsure == nil or gilRelease == nil or gilSaveThread == nil:
      raise newException(ValueError, "libpython does not export the GIL state API (PyGILState_Ensure/Release, PyEval_SaveThread)")
    gilReady = true

proc releaseGIL*() =
  ## Give the GIL back on the thread that initialised CPython.
  ##
  ## Call once, on the main thread, after the model is loaded and before
  ## `serve` can dispatch a handler. From then on Python may only be entered
  ## through `embedOne` - which also applies to the main thread.
  initEmbedding()
  {.cast(gcsafe).}:
    discard gilSaveThread()

proc modelDim*(): int =
  ## Embedding size of the loaded model (384 for the MiniLM default), callable
  ## from any thread. Asked from Python, so it needs a loaded model; an index
  ## that has no vectors yet has no `dim` of its own and the service reports this
  ## one instead.
  initEmbedding()
  {.cast(gcsafe).}:
    let state = gilEnsure()
    try:
      result = dim()
    finally:
      gilRelease(state)

proc embedOne*(text: string): seq[float32] =
  ## Embedding of one text, callable from any thread.
  ##
  ## One text per call is what the HTTP API needs (one article, or one query).
  ## The Python side pays a fixed per-call cost either way, so batching only
  ## helps the bulk importers in `examples/`, not this.
  ##
  ## The cast is for two different globals: this module's GIL hooks, and the
  ## model handle `bert_nim` keeps in a `let` after import. Both are written once
  ## at startup and only read here, and the GIL is what makes the concurrent
  ## Python calls themselves serial.
  initEmbedding()
  {.cast(gcsafe).}:
    let state = gilEnsure()
    try:
      result = embedFlat(@[text])
    finally:
      gilRelease(state)
