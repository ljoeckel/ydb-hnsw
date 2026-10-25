// Thin C bridge between llama.h and src/llama_embed.nim.
//
// Why a shim instead of importing `llama.h` directly: llama.cpp passes several
// structs *by value* (llama_model_params, llama_context_params, llama_batch),
// and their layout changes between releases. Mirroring them in Nim compiles
// fine and then crashes or silently misreads fields when the installed library
// is a different version. Everything that needs such a struct is therefore
// handled here, compiled against the real header; Nim only sees opaque pointers,
// ints and C strings.
//
// The entry points are named shim_* so they cannot collide with the library's
// own symbols.

#include <llama.h>

#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>

static void shim_set_err(char * err, size_t errlen, const char * fmt, ...) {
    if (err == NULL || errlen == 0) {
        return;
    }
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(err, errlen, fmt, ap);
    va_end(ap);
}

static void shim_log_silent(enum ggml_log_level level, const char * text, void * user_data) {
    (void) level;
    (void) text;
    (void) user_data;
}

// llama.cpp's default logger writes every load/decode message to stderr. A
// library should stay quiet unless the host asks for it.
void shim_backend_init(int quiet) {
    if (quiet) {
        llama_log_set(shim_log_silent, NULL);
    }
    llama_backend_init();
}

void shim_backend_free(void) {
    llama_backend_free();
}

struct llama_model * shim_load_model(const char * path, int n_gpu_layers,
                                     char * err, size_t errlen) {
    struct llama_model_params params = llama_model_default_params();
    params.n_gpu_layers = n_gpu_layers;

    struct llama_model * model = llama_model_load_from_file(path, params);
    if (model == NULL) {
        shim_set_err(err, errlen, "cannot load GGUF model '%s'", path);
    }
    return model;
}

void shim_free_model(struct llama_model * model) {
    if (model != NULL) {
        llama_model_free(model);
    }
}

// `pooling` is a llama_pooling_type value (-1 = model default).  n_batch is also
// used as n_ubatch: for a non-causal (encoder) model every sequence in a batch
// has to be processed in one ubatch.
struct llama_context * shim_new_context(struct llama_model * model,
                                        int n_ctx, int n_batch, int n_seq_max,
                                        int n_threads, int n_threads_batch,
                                        int pooling,
                                        char * err, size_t errlen) {
    struct llama_context_params params = llama_context_default_params();
    params.n_ctx           = (uint32_t) n_ctx;
    params.n_batch         = (uint32_t) n_batch;
    params.n_ubatch        = (uint32_t) n_batch;
    params.n_seq_max       = (uint32_t) n_seq_max;
    params.n_threads       = n_threads;
    params.n_threads_batch = n_threads_batch;
    params.pooling_type    = (enum llama_pooling_type) pooling;
    params.embeddings      = true;
    params.kv_unified      = true;

    struct llama_context * ctx = llama_init_from_model(model, params);
    if (ctx == NULL) {
        shim_set_err(err, errlen, "cannot create llama context (%d threads, n_ctx %d)", n_threads, n_ctx);
    }
    return ctx;
}

void shim_free_context(struct llama_context * ctx) {
    if (ctx != NULL) {
        llama_free(ctx);
    }
}

int32_t shim_n_embd_out(const struct llama_model * model) {
    return llama_model_n_embd_out(model);
}

int shim_pooling_type(const struct llama_context * ctx) {
    return (int) llama_pooling_type(ctx);
}

// Returns the token count, or its negation when the buffer is too small (the
// same convention llama_tokenize uses).
int shim_tokenize(const struct llama_model * model,
                  const char * text, int text_len,
                  int32_t * out, int out_cap,
                  int add_special, int parse_special) {
    const struct llama_vocab * vocab = llama_model_get_vocab(model);
    return llama_tokenize(vocab, text, text_len,
                          (llama_token *) out, out_cap,
                          add_special != 0, parse_special != 0);
}

// Token used to close a truncated sequence (SentencePiece </s> for XLM-R).
int shim_seq_end_token(const struct llama_model * model) {
    const struct llama_vocab * vocab = llama_model_get_vocab(model);
    llama_token tok = llama_vocab_sep(vocab);
    if (tok == LLAMA_TOKEN_NULL) {
        tok = llama_vocab_eos(vocab);
    }
    return (int) tok;
}

// --- batch ---------------------------------------------------------------
// llama_batch_init returns the struct by value; keep it here and hand Nim an
// opaque handle.  `cap` is tracked because llama_batch.n_tokens doubles as the
// fill counter after the first add.
struct shim_batch {
    struct llama_batch batch;
    int32_t            cap;
};

void * shim_batch_new(int n_tokens, int n_seq_max) {
    struct shim_batch * sb = malloc(sizeof(struct shim_batch));
    if (sb == NULL) {
        return NULL;
    }
    sb->batch = llama_batch_init(n_tokens, 0, n_seq_max);
    sb->cap   = n_tokens;
    if (sb->batch.token == NULL) {
        free(sb);
        return NULL;
    }
    return sb;
}

void shim_batch_free(void * handle) {
    if (handle == NULL) {
        return;
    }
    struct shim_batch * sb = handle;
    llama_batch_free(sb->batch);
    free(sb);
}

void shim_batch_reset(void * handle) {
    struct shim_batch * sb = handle;
    sb->batch.n_tokens = 0;
}

int shim_batch_capacity(const void * handle) {
    return ((const struct shim_batch *) handle)->cap;
}

// Append one token: `seq` is the sequence it belongs to, `pos` its position in
// that sequence.  Returns the index, or -1 when the batch is full.
int shim_batch_add(void * handle, int token, int pos, int seq) {
    struct shim_batch * sb = handle;
    const int32_t i = sb->batch.n_tokens;
    if (i >= sb->cap) {
        return -1;
    }
    sb->batch.token[i]     = (llama_token) token;
    sb->batch.pos[i]       = (llama_pos) pos;
    sb->batch.n_seq_id[i]  = 1;
    sb->batch.seq_id[i][0] = (llama_seq_id) seq;
    sb->batch.logits[i]    = 1;   // embeddings are read per sequence below
    sb->batch.n_tokens     = i + 1;
    return (int) i;
}

// Run one batch.  The memory is cleared first: embeddings are stateless, and
// llama_decode requires the context's memory to be empty between calls.
int shim_decode(struct llama_context * ctx, void * handle, char * err, size_t errlen) {
    struct shim_batch * sb = handle;
    llama_memory_clear(llama_get_memory(ctx), true);

    const int32_t rc = llama_decode(ctx, sb->batch);
    if (rc < 0) {
        shim_set_err(err, errlen, "llama_decode failed (%d)", rc);
    }
    return (int) rc;
}

// Pooled embedding of one sequence from the last decode; `n_embd` floats,
// valid until the next decode.  NULL when the sequence produced nothing.
const float * shim_embeddings_seq(struct llama_context * ctx, int seq) {
    return llama_get_embeddings_seq(ctx, (llama_seq_id) seq);
}
