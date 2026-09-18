# begin Nimble config (version 2)
when withDir(thisDir(), system.fileExists("nimble.paths")):
  include "nimble.paths"
# end Nimble config

# --- YottaDB linking ---------------------------------------------------------
# nimyottadb declares the YottaDB C API with plain `importc` pragmas (no
# `dynlib`), so the symbols have to be linked in from libyottadb.so. That
# library is not in the loader's default search path, hence the explicit
# -rpath; without it the binary links but fails to start.
# $ydb_dist comes from YottaDB's environment (`ydb_env_set`); the literal is
# just a fallback for shells that do not have it set.
const ydbDist =
  block:
    let fromEnv = getEnv("ydb_dist")
    if fromEnv.len > 0: fromEnv else: "/usr/local/lib/yottadb/r206"

{.passL: "-L" & ydbDist & " -lyottadb -Wl,-rpath," & ydbDist.}

