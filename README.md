# ydb-hnsw
A vector db implementation in Nim with YottaDB as database backend.
It is implemented by using a HNSW index.
A Hierarchical Navigable Small World (HNSW) Index is a highly efficient, graph-based algorithm for the approximate search for the nearest neighbors (Approximate Nearest Neighbor, ANN) in high-dimensional vector data. It is mainly used in modern vector databases and AI applications

# Embedding backend: llama.cpp

Embeddings are computed in-process by [llama.cpp](https://github.com/ggml-org/llama.cpp)
from a GGUF file. No Python, PyTorch or virtualenv is involved at runtime.

```bash
# build llama.cpp once (add -DGGML_CUDA=ON for the GPU, see below)
git clone https://github.com/ggml-org/llama.cpp
cmake -B llama.cpp/build -S llama.cpp -DCMAKE_BUILD_TYPE=Release -DLLAMA_CURL=OFF
cmake --build llama.cpp/build -j --target llama-embedding

export LLAMA_CPP_DIR=$PWD/llama.cpp/build   # what config.nims links against
export HNSW_GGUF=~/models/gguf/paraphrase-multilingual-MiniLM-L12-v2.gguf
# alternatively: HNSW_GGUF_DIR=<dir> (default ~/models/gguf), then <model>.gguf
```

GPU: nothing changes in this repo - build llama.cpp with `-DGGML_CUDA=ON`
(`-DCMAKE_CUDA_ARCHITECTURES=<cc>`, e.g. `61` for a GTX 1060). The backend then
offloads the model automatically when a GPU is present and falls back to the CPU
if that load fails; `HNSW_EMBED_GPU_LAYERS=0` pins the CPU. CMake 4.x needs
`bash/llama_cpp_cuda_cmake.patch` for the CUDA build (details in `LLAMA_CPP.md`).

Converting the model is a one-time step and the only thing that still needs
Python (torch + transformers):

```bash
pip install sentencepiece protobuf
python llama.cpp/convert_hf_to_gguf.py \
    sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 \
    --outtype q8_0 --outfile "$HNSW_GGUF"
```

`q8_0` is the recommended file when a GPU is present: llama.cpp's CUDA int8
matmul is ~25% faster than F16 here and the vector error (cos >= 0.99985 vs the
old torch output) is below the `vqInt8` quantisation the index applies anyway.
Use `--outtype f16` on a CPU-only build (or for reference vectors). Either way,
build the index with the same file you later query with - vectors of different
quants differ slightly.

`convert_hf_to_gguf.py` mis-detects this model's XLM-R tokenizer, so apply
`bash/llama_cpp_xlmroberta.patch` to llama.cpp first. `LLAMA_CPP.md` explains
why, shows the verification (the vectors match the old torch path to <= 2.4e-4
per component, top-5 ranking identical) and has the throughput numbers.

# Setup Nim environment
nimble setup
# HTTP service

`src/hnsw_service.nim` puts one HNSW index behind a small JSON API. The server is
[mummyDS](https://github.com/ljoeckel/mummyDS) (mummy plus SSE/Datastar), the
embedding runs in the same process through llama.cpp, and vectors and article
records both live in YottaDB - so a restart reopens the index and keeps
answering.

The service needs the mummyDS checkout and its `webby` / `crunchy` dependencies
(see `requires` in `ydbhnsw.nimble`) and a GGUF model file (`HNSW_GGUF`, or
`HNSW_GGUF_DIR/<model name>.gguf`).

```bash
nim c --threads:on --mm:orc -d:release src/hnsw_service.nim
HNSW_GGUF=~/models/gguf/paraphrase-multilingual-MiniLM-L12-v2.gguf \
  ./src/hnsw_service --index=^HNSWArticles --port=8080
```

`nimble runService` builds and starts it.

## Endpoints

| Route | Body / query | Answer |
| --- | --- | --- |
| `POST /embed` | `{"idxref": "...", "title": "...", "description": "..."}` | the stored article, `201` new / `200` replaced |
| `GET /search?q=<text>&k=<n>` | `k` optional, default 10, max 100 | `{"query", "count", "results": [...]}` |
| `POST /search` | `{"query": "<text>", "k": <n>}` | same as the GET form |
| `GET /` | - | status, index, model, dim, live article count |

Each search result is `{"idxref", "title", "description", "score"}`, nearest
first. `score` is cosine similarity.

```bash
curl -X POST http://localhost:8080/embed \
  -H 'Content-Type: application/json' \
  -d '{"idxref":"art-1",
       "title":"YottaDB adds vector search",
       "description":"The M database gets an HNSW index written in Nim."}'

curl 'http://localhost:8080/search?q=vector%20database&k=5'
```

What gets embedded is `title + ". " + description`, normalised the way the bulk
importers normalise their text (lower case, transliterated, single spaces), and
a query is normalised the same way. `description` may be empty - the vector is
then the title alone - but the field has to be present.

`/embed` is an upsert: a known `idxref` has its old vector deleted and is
re-embedded, so a changed title or description also changes what the article
matches. Ids are never reused, so a replacement gets a fresh node id.

## Options

`--port` (default 8080, env `HNSW_PORT`) and `--bind` (default `localhost`, env
`HNSW_BIND`) belong to the service. Everything else is an index option and means
what it means for the other tools: `--index`, `--model`, `--quant`, `--cache`,
`--m`, `--efConstruction`, `--efSearch`, `--linkFmt`, `--verbose`. `--reset`
drops the index *and* the article records and exits, which is how a test index is
cleaned up.

```bash
PYTHONPATH=... ./src/hnsw_service --index=^HNSWSvcTest --reset
```

`bash/hnsw_service_smoke.sh` builds the service, runs it on a throwaway index,
embeds a few articles, searches, and removes the index again.
