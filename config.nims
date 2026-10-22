# begin Nimble config (version 2)
when withDir(thisDir(), system.fileExists("nimble.paths")):
  include "nimble.paths"
# end Nimble config

# --- HTTP service (mummyDS) ---------------------------------------------------
# Mummy dispatches requests to worker threads and checks for threads at compile
# time; the flag can only be given here or on the command line, not from inside a
# module. ORC (which Mummy also requires) is Nim 2's default. Harmless for the
# index tools: they already use threadvars and are safe to build with threads on.
switch("threads", "on")

# --- hnsw.nim distance kernels -----------------------------------------------
# `dot`, `dotInt8` and `normalize` use the nimsimd intrinsics; -mavx is what lets
# the C compiler accept them. Build with `-d:hnswNoSimd` (or delete these three
# lines) to get the scalar kernels instead, which compile for any x86-64 - about
# 4x slower on the float32 dot, 6x on the int8 one, 7x on normalize.
if not defined(hnswNoSimd):
  switch("define", "hnswSimd")
  switch("passC", "-mavx")
  # The int8 kernel has a 256-bit body too, but it is opt-in: on Zen 1/+ a
  # 256-bit integer operation is split into two 128-bit halves, and there the
  # AVX2 loop measured 36.3 ns/dot against the 128-bit body's 28.5 on the
  # 384-dim vectors - 28% slower, not faster. Haswell through Skylake and
  # Zen 2+ run 256-bit at full width, so `-d:hnswAvx2` is worth trying there;
  # it brings `-mavx2` with it rather than leaving the flag to the caller.
  if defined(hnswAvx2):
    switch("passC", "-mavx2")

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

