# llama.cpp as the embedding backend — feasibility check

**Verdict: yes.** With a local GGUF conversion, llama.cpp reproduces the
sentence-transformers embeddings to float32 precision, so the index sees the same
vectors. The obstacle is not the runtime but `convert_hf_to_gguf.py`, which
currently mis-converts this model's tokenizer (13-line patch, below).

Everything here was measured on 2026-10-25 against llama.cpp `6050b46`, the
`paraphrase-multilingual-MiniLM-L12-v2` weights, and the venv in `hnsw_env/`.
Harness: `bash/llama_embed_compare.py`.

## Agreement with torch

32 mixed-language texts, mean pooling, L2 normalised:

| GGUF | size | cosine min | max abs diff | top-5 sanity (llama query vs torch corpus) |
|---|---|---|---|---|
| local conversion, F32, patched converter | 477 MB | 0.99999911 | 2.3e-4 | 32/32 identical order |
| local conversion, F16, patched converter | 242 MB | 0.99999905 | 2.4e-4 | 32/32 identical order |
| local conversion, Q8_0, patched converter | 132 MB | 0.99984801 | 3.4e-3 | 32/32 identical order |
| `mykor/...-F32.gguf` (community, 2023) | 477 MB | 0.98216900 | 3.7e-2 | 0.981 overlap, 23/32 identical |

* Rank the F16 file as the deployment artifact: same fidelity as F32 at half the
  bytes. Q8_0 is still fine for cosine search but 15x more drift, and it is not
  faster than F32 on this CPU.
* The community GGUF has the *same tensor bytes* as a local conversion (verified
  tensor by tensor) and tokenises `Bundestag` the same way, yet sits 0.2–1.8 %
  off. Its added-token table (`<mask>` / `[PAD…]`, `token_type_count`) differs;
  do not mix it into an index built from a local conversion.
* `torch mean(all tokens)` is what `--pooling mean` computes — verified against
  `mean`, `mean without specials`, `cls` and `last` variants.

## Why a naive conversion fails (each stage measured)

1. **WordPiece instead of SentencePiece (worst).** The model's `config.json` says
   `BertModel`, but the tokenizer is XLM-R's Unigram model.
   `BertModel.set_vocab` therefore emits a WordPiece vocab (`tokenizer.ggml.model
   = "bert"`, phantom-space rewrite) and every ordinary word becomes `<unk>`:
   cosine collapses to 0.14.
2. **Unigram without scores/charsmap.** Forcing `tokenizer.ggml.model = "t5"`
   but writing the token list only gets Latin text roughly right; without
   `tokenizer.ggml.scores` the Viterbi search degrades to greedy matching
   (mean cosine 0.87, worst 0.20).
3. **Missing tokenizer metadata.** With scores, `Bundestag` still split into
   `Bundes`+`tag` because `tokenizer.ggml.add_space_prefix`,
   `tokenizer.ggml.precompiled_charsmap` and
   `tokenizer.ggml.remove_extra_whitespaces` were absent (mean cosine 0.98).

All three are fixed by routing XLM-R tokenizers to the converter's existing
`_xlmroberta_set_vocab()`: `bash/llama_cpp_xlmroberta.patch` (13 lines, applies
to `conversion/bert.py`).

```sh
cd ~/git/llama.cpp
git apply /path/to/ydb-hnsw/bash/llama_cpp_xlmroberta.patch
./build/bin/llama-embedding --version      # 6050b46 or later

# conversion needs sentencepiece + protobuf in the driver env
pip install sentencepiece protobuf

python convert_hf_to_gguf.py \
    ~/models/hf/paraphrase-multilingual-MiniLM-L12-v2 \
    --outtype f16 \
    --outfile ~/models/gguf/paraphrase-multilingual-MiniLM-L12-v2-f16.gguf
```

## Residual difference: GELU

llama.cpp's BERT graph uses `LLM_FFN_GELU` (`ggml_gelu`, tanh approximation);
HuggingFace BERT uses exact erf. Switching `src/models/bert.cpp` to
`LLM_FFN_GELU_ERF` drops the max component difference from 2.3e-4 to 2.7e-5
(cosine 0.99999911 -> 0.99999988). Optional: both are two orders of magnitude
below the index's own `vqInt8` quantisation noise (~1e-3), so this is a nicety,
not a blocker.

## Throughput (CPU, 240 texts, this box)

| backend | batched (32) | one text per call |
|---|---|---|
| llama.cpp F32, 4 threads | 6.96 ms/text | (in-process; call overhead not isolated) |
| llama.cpp F32, 8 threads | 4.06 ms/text | |
| llama.cpp F16, 8 threads | 5.90 ms/text | |
| llama.cpp Q8_0, 8 threads | 5.42 ms/text | |
| torch, 1 thread | 15.32 ms/text | 32.25 ms/text |
| torch, 4 threads | 6.15 ms/text | 17.55 ms/text |
| torch, 8 threads | 3.03 ms/text | 16.03 ms/text |

Batched, torch with 8 threads is ~1.3x faster; a one-text call (what
`hnsw_service` does) is at most the batched per-text cost in llama.cpp — a single
~10-token sequence is less work than being one of ~50 sequences in a decode —
against torch's measured 16–32 ms, and it needs no Python, GIL or venv. Model
load is ~1.3 s for the F16 file.

## What was implemented

`hnsw.nim` reached the model through `setModelLoader`, so only the leaf module
changed:

- **`src/llama_embed.nim`** - the embedding backend. Same public surface as the
  old `bert_nim` (`loadModel`, `dim`, `embed`, `embedFlat`, `embedOne`, `query`),
  so `hnsw_common`, the examples and the service are untouched. It packs texts
  into one `llama_batch` (many sequences per decode), mean-pools, L2-normalises,
  truncates at 128 tokens, and serialises everything on one `Lock` because a
  `llama_context` is not thread-safe.
- **`src/llama_shim.c`** - compiled into the binary with `{.compile.}`. llama.h
  passes `llama_model_params`, `llama_context_params` and `llama_batch` **by
  value** and their layouts change between releases; the shim is compiled against
  the real header, so Nim only sees opaque pointers and scalars.
- **Deleted**: `src/bert_nim.nim`, `src/pyembed.nim`, `python/sbert_bridge.py`
  and the `nimpy` dependency. The GIL dance is gone with them.
- **`src/hnsw_service.nim`** - no `initEmbedding` / `releaseGIL`; the service
  calls `embedOne` / `dim` directly.
- **`config.nims`** - `LLAMA_CPP_DIR` (default `~/git/llama.cpp/build`) is passed
  as `-d:llamaCppDir`; `llama_embed` adds `-I<build>/../include`,
  `-I<build>/../ggml/include`, `-L<build>/bin -lllama` and an rpath to the build
  tree (the same shape as the existing YottaDB block).
- **Model name -> file**: `HnswParams.model` is still the string pinned in META.
  `llama_embed.resolveModelFile` tries `HNSW_GGUF`, then a `.gguf` path, then
  `HNSW_GGUF_DIR/<name>.gguf` (`-f16`, `-f32` suffixes), with
  `HNSW_GGUF_DIR=~/models/gguf` by default.

Verified after the switch:

- `bert_demo` embeds and ranks (dot(list path, flat path) = 1.000000).
- Vectors from the Nim binding vs `llama-embedding` on the same 32 texts:
  max |diff| 1.0e-7 (JSON print precision, i.e. the same computation); vs torch:
  max |diff| 2.4e-4, cosine 0.9999988 - the F16 GGUF difference.
- `hnsw_service` on the live `^HNSWArticles` (742,854 articles): `/` reports
  `dim 384, live 742854`; a German query answers in 15 ms of embedding plus
  2.9 ms of index search with sensible hits.

Two examples do not compile and did not before this change:
`examples/hnsw_search.nim` (destructures `enumerate(...)` into four names) and
`examples/migrateInt.nim` (`unpackInts` no longer exists).

## Open risks

- **The live index was built with torch.** Vectors from a patched local
  conversion differ by <=2.4e-4 per component, which is far below `vqInt8`
  noise, but run `test/hnsw_recall_check.nim` / `hnsw_recall_bruteforce.nim`
  against `^HNSWArticles` before trusting it. Re-embedding 742k articles is the
  clean alternative.
- **`llama.h` churns between releases** (`llama_decode` -> `llama_process`,
  `llama_load_model_from_file` -> `llama_model_load_from_file`): pin a tag. The
  shim contains the churn; the Nim side only uses scalars and pointers.
- **Truncation**: the model's `max_seq_length` is 128 and `HNSW_EMBED_MAX_TOKENS`
  defaults to it, so the caller no longer has to truncate - but a description cut
  at 128 tokens is still cut.
- **GPU**: this box has a GTX 1060/6 GB; `HNSW_EMBED_GPU_LAYERS` offloads to a
  CUDA build of llama.cpp (there is none yet, so it is untested).
- **Backend knobs** (environment): `HNSW_GGUF`, `HNSW_GGUF_DIR`,
  `HNSW_EMBED_THREADS` (4), `HNSW_EMBED_CTX` (512), `HNSW_EMBED_SEQS` (64),
  `HNSW_EMBED_MAX_TOKENS` (128), `HNSW_EMBED_POOLING` (mean),
  `HNSW_EMBED_GPU_LAYERS` (0), `HNSW_LLAMA_LOG` (off).
