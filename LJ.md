Running it

export LLAMA_CPP_DIR=$HOME/git/llama.cpp/build     # build-time link path
export HNSW_GGUF=$HOME/models/gguf/local6-f16.gguf # or HNSW_GGUF_DIR=<dir>

nim c -d:release src/examples/bert_demo.nim && ./src/examples/bert_demo
nim c --threads:on --mm:orc -d:release src/hnsw_service.nim

HNSW_GGUF wins over everything; otherwise the index's pinned name resolves to HNSW_GGUF_DIR/<name>.gguf (default ~/models/gguf). Other knobs: HNSW_EMBED_THREADS (4), HNSW_EMBED_CTX (512), HNSW_EMBED_SEQS (64), HNSW_EMBED_MAX_TOKENS (128), HNSW_EMBED_POOLING (mean), HNSW_EMBED_GPU_LAYERS (0), HNSW_LLAMA_LOG (off).


HNSW_EMBED_GPU_LAYERS=0 — does it still stall on CPU? That separates CUDA from everything else.
HNSW_LLAMA_LOG=1 — llama.cpp's own log (device selection, tensor assignment).
gdb -p <pid> -batch -ex "thread apply all bt" while it's stuck — that names the frame (YottaDB read, embedding, or CUDA).
