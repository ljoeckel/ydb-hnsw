import std/[os]
import nimpy

# --- Bake sbert_bridge.py into the binary -----------------------------------
# staticRead runs at compile time, relative to src/ (the file with the call),
# so ONE level up reaches the repo root, then into python/.
const sbertSource = staticRead("../python/sbert_bridge.py")

# Only used for tracebacks and __file__; compile-time so it is absolute
# regardless of the working directory the binary is started from.
const sbertVirtualPath =
  currentSourcePath().parentDir.parentDir / "python" / "sbert_bridge.py"


proc installBakedModule(name, source, virtualPath: string): PyObject =
  ## Execute `source` as if it were the module `name`, register it in
  ## sys.modules, and hand the module object back.
  let sys      = pyImport("sys")
  let builtins = pyImport("builtins")

  let module = pyImport("types").ModuleType(name)

  # compile() with a filename keeps tracebacks readable
  # ("File sbert_bridge.py, line 42") instead of the useless "File <string>".
  let code = builtins.compile(source, virtualPath, "exec")

  # So os.path.dirname(__file__) inside the module does not explode.
  # "__file__" is a string literal here, so the lexer is happy.
  discard builtins.setattr(module, "__file__", virtualPath)

  # Register *before* exec: needed if the module imports itself, and it is
  # what makes a later pyImport(name) return this object instead of disk.
  # sys.modules is a dict, so this is Python's `sys.modules[name] = module`.
  # (Writing `.__setitem__` is impossible - see note below.)
  sys.modules[name] = module

  # vars(module) is Python for `module.__dict__`, spelled with a legal name.
  discard builtins.exec(code, builtins.vars(module))

  module

# Runs at module init, i.e. before bert_nim's top-level statements, because
# `import bakery` in bert_nim is executed first.
let bridge* = installBakedModule("sbert_bridge", sbertSource, sbertVirtualPath)