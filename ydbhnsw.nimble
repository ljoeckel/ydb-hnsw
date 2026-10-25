# Package Information
version = "0.0.1"
author = "Lothar Joeckel"
description = "A vector db implementation in Nim with YottaDB as database backend"
license = "GNU General Public License 3.0"
srcDir = "src"
binDir = "bin"
requires "nim >= 2.2.4"

# Dependencies
requires "htmlparser"
requires "malebolgia >=1.3.2"
requires "zippy >=0.10.20"
requires "zstd >=0.9.0"
requires "nimsimd >=1.3.0"      # distance kernels, enabled by -d:hnswSimd
requires "nimyottadb#head"      # change later to a release
requires "https://github.com/ljoeckel/nimlz4.git"

# The HTTP service in src/hnsw_service.nim. mummyDS is the fork of mummy with
# SSE + Datastar support; webby and crunchy are its dependencies.
requires "https://github.com/ljoeckel/mummyDS.git"
requires "webby >= 0.2.1"
requires "crunchy >= 0.1.11"


# The embedding model is a GGUF file read by llama.cpp in-process (src/llama_embed.nim,
# src/llama_shim.c). The build needs a llama.cpp build tree, the run needs the
# GGUF itself:
#   export LLAMA_CPP_DIR=~/git/llama.cpp/build
#   export HNSW_GGUF=~/models/gguf/paraphrase-multilingual-MiniLM-L12-v2.gguf
#   (or HNSW_GGUF_DIR=<dir>, so <model name>.gguf is found; default ~/models/gguf)
# See LLAMA_CPP.md for the conversion recipe (the XLM-R tokenizer needs a one-line
# converter patch) and for the measurements behind this choice.

task runNim, "Build and run the embedding demo (llama.cpp backend)":
  exec "nim c -d:release src/examples/bert_demo.nim"
  exec "./src/examples/bert_demo"

task runHnsw, "Build and run hnsw_search":
  exec "nim c -d:release src/hnsw_search.nim"
  exec "./src/hnsw_search"

task runService, "Build and run the HNSW HTTP service":
  exec "nim c --threads:on --mm:orc -d:release src/hnsw_service.nim"
  exec "./src/hnsw_service"
