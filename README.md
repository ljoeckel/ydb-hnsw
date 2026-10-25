# ydb-hnsw
A vector db implementation in Nim with YottaDB as database backend.
It is implemented by using a HNSW index.
A Hierarchical Navigable Small World (HNSW) Index is a highly efficient, graph-based algorithm for the approximate search for the nearest neighbors (Approximate Nearest Neighbor, ANN) in high-dimensional vector data. It is mainly used in modern vector databases and AI applications

# Embedding backend: llama.cpp

Embeddings are computed in-process by [llama.cpp](https://github.com/ggml-org/llama.cpp)
from a GGUF file. No Python, PyTorch or virtualenv is involved at runtime.

```bash
# build llama.cpp once (CPU is fine; add -DGGML_CUDA=ON for a CUDA build)
git clone https://github.com/ggml-org/llama.cpp
cmake -B llama.cpp/build -S llama.cpp -DCMAKE_BUILD_TYPE=Release -DLLAMA_CURL=OFF
cmake --build llama.cpp/build -j --target llama-embedding

export LLAMA_CPP_DIR=$PWD/llama.cpp/build   # what config.nims links against
export HNSW_GGUF=~/models/gguf/paraphrase-multilingual-MiniLM-L12-v2.gguf
# alternatively: HNSW_GGUF_DIR=<dir> (default ~/models/gguf), then <model>.gguf
```

Converting the model is a one-time step and the only thing that still needs
Python (torch + transformers):

```bash
pip install sentencepiece protobuf
python llama.cpp/convert_hf_to_gguf.py \
    sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 \
    --outtype f16 --outfile "$HNSW_GGUF"
```

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
