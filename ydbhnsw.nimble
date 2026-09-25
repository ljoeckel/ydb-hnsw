# Package Information
version = "0.0.1"
author = "Lothar Joeckel"
description = "A vector db implementation in Nim with YottaDB as database backend"
license = "GNU General Public License 3.0"
srcDir = "src"
binDir = "bin"
requires "nim >= 2.2.4"

# Dependencies
requires "malebolgia >=1.3.2"
requires "zippy >=0.10.20"
requires "zstd >=0.9.0"
requires "nimsimd >=1.3.0"      # distance kernels, enabled by -d:hnswSimd
requires "https://github.com/ljoeckel/nimlz4.git"


# The embedded interpreter is the system libpython3.12 that nimpy dlopens, so
# the venv packages (sentence_transformers, numpy, torch) have to be handed to
# it through PYTHONPATH. Set EDIT if the venv moves (see ~/bert.sh).
const
  VenvSitePackages = "/home/ljoeckel/git/hnsw_env/lib/python3.14/site-packages"

task runNim, "Build and run the Nim host with the venv on PYTHONPATH":
  exec "nimble build"
  exec "PYTHONPATH=" & VenvSitePackages & " ./bert_nim"

task runHnsw, "Build and run hnsw_search with the venv on PYTHONPATH":
  exec "nim c -d:release src/hnsw_search.nim"
  exec "PYTHONPATH=" & VenvSitePackages & " ./src/hnsw_search"