// Thin C bridge between llama.h and src/llama_chat.nim.
//
// Same reason as llama_shim.c on the embedding side: llama.cpp passes
// llama_model_params, llama_context_params, llama_batch and
// llama_sampler_chain_params *by value*, and those layouts move between
// releases. Everything that touches such a struct is done here, compiled
// against the real header, so Nim only ever sees opaque pointers, ints and C
// strings.
//
// The entry points are named chat_* so they cannot collide with the library or
// with the shim_* functions in llama_shim.c.

#include <llama.h>
#include <ggml-backend.h>

#include <stdarg.h>
#include <stddef.h>
#include <stdio.h>

// Mirrors llama_chat_message (two const char *), minus the llama.h dependency.
// The Nim side only ever sees this struct.
typedef struct chat_turn {
    const char * role;
    const char * content;
} chat_turn;

static void chat_set_err(char * err, size_t errlen, const char * fmt, ...) {
    if (err == NULL || errlen == 0) {
        return;
    }
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(err, errlen, fmt, ap);
    va_end(ap);
}

static void chat_log_silent(enum ggml_log_level level, const char * text, void * user_data) {
    (void) level;
    (void) text;
    (void) user_data;
}

// llama.cpp's default logger narrates every tensor it maps; a library should
// stay quiet unless the host asks for it.
void chat_backend_init(int quiet) {
    if (quiet) {
        llama_log_set(chat_log_silent, NULL);
    }
    llama_backend_init();
}

void chat_backend_free(void) {
    llama_backend_free();
}

// Number of GPU devices the loaded backends registered: 0 on a CPU-only build,
// whatever the hardware is.
int chat_gpu_device_count(void) {
    int count = 0;
    const size_t ndev = ggml_backend_dev_count();
    for (size_t i = 0; i < ndev; ++i) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        if (dev != NULL && ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_GPU) {
            ++count;
        }
    }
    return count;
}

// Description of the first GPU device, or "" when there is none. Owned by the
// backend; the caller copies it.
const char * chat_gpu_device_name(void) {
    const size_t ndev = ggml_backend_dev_count();
    for (size_t i = 0; i < ndev; ++i) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        if (dev != NULL && ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_GPU) {
            return ggml_backend_dev_description(dev);
        }
    }
    return "";
}

struct llama_model * chat_load_model(const char * path, int n_gpu_layers,
                                     char * err, size_t errlen) {
    struct llama_model_params params = llama_model_default_params();
    params.n_gpu_layers = n_gpu_layers;

    struct llama_model * model = llama_model_load_from_file(path, params);
    if (model == NULL) {
        chat_set_err(err, errlen, "cannot load model from '%s'", path);
    }
    return model;
}

void chat_free_model(struct llama_model * model) {
    if (model != NULL) {
        llama_model_free(model);
    }
}

struct llama_context * chat_new_context(struct llama_model * model, int n_ctx,
                                        int n_batch, int n_threads,
                                        char * err, size_t errlen) {
    struct llama_context_params params = llama_context_default_params();
    params.n_ctx           = (uint32_t) n_ctx;      // 0 = the model's own context size
    params.n_batch         = (uint32_t) n_batch;
    params.n_ubatch        = (uint32_t) n_batch;
    params.n_seq_max       = 1;
    params.n_threads       = (uint32_t) n_threads;  // 0 = llama.cpp picks
    params.n_threads_batch = (uint32_t) n_threads;
    params.embeddings      = false;
    params.no_perf         = true;

    struct llama_context * ctx = llama_init_from_model(model, params);
    if (ctx == NULL) {
        chat_set_err(err, errlen, "llama_init_from_model failed (n_ctx %d, n_batch %d)",
                     n_ctx, n_batch);
    }
    return ctx;
}

void chat_free_context(struct llama_context * ctx) {
    if (ctx != NULL) {
        llama_free(ctx);
    }
}

int chat_n_ctx(const struct llama_context * ctx) {
    return (int) llama_n_ctx(ctx);
}

// Forget the whole KV cache. The chat loop clears it at every turn and decodes
// the full conversation again - cheaper to reason about than tracking which
// tokens of the re-formatted prompt are still valid.
void chat_clear_kv(struct llama_context * ctx) {
    llama_memory_clear(llama_get_memory(ctx), true);
}

// The model's own chat template (from GGUF metadata), or NULL when it has none.
const char * chat_model_template(const struct llama_model * model) {
    return llama_model_chat_template(model, NULL);
}

// Format `turns` into `out`. Returns the formatted length (may be larger than
// `cap`, in which case the text was truncated), or -1 when llama.cpp does not
// recognise the template.
int chat_format_prompt(const struct llama_model * model,
                       const chat_turn * turns, int n_turns, int add_assistant,
                       char * out, int cap) {
    const char * tmpl = chat_model_template(model);
    // chat_turn and llama_chat_message are layout-compatible: two pointers.
    return llama_chat_apply_template(tmpl,
                                     (const struct llama_chat_message *) turns,
                                     (size_t) n_turns, add_assistant != 0,
                                     out, cap);
}

// Tokenize `text`. Returns the token count, or -needed when `out` is too small.
int chat_tokenize(const struct llama_model * model, const char * text, int text_len,
                  int32_t * out, int cap) {
    const struct llama_vocab * vocab = llama_model_get_vocab(model);
    // Special tokens are parsed (<|im_start|> and friends must stay one token),
    // BOS is not added - the template already decides what the prompt starts with.
    return llama_tokenize(vocab, text, text_len, out, cap, false, true);
}

// Feed `tokens` to the model at positions [pos, pos + n). Logits are computed
// for the last token only, which is all the sampler needs.
static int chat_decode_chunk(struct llama_context * ctx, const int32_t * tokens, int n,
                            int pos, char * err, size_t errlen) {
    struct llama_batch batch = llama_batch_init(n, 0, 1);
    if (batch.token == NULL) {
        chat_set_err(err, errlen, "llama_batch_init(%d) failed", n);
        return -1;
    }

    for (int i = 0; i < n; ++i) {
        batch.token[i]     = tokens[i];
        batch.pos[i]       = pos + i;
        batch.n_seq_id[i]  = 1;
        batch.seq_id[i][0] = 0;
        batch.logits[i]    = 0;
    }
    batch.n_tokens    = n;
    batch.logits[n-1] = 1;

    const int rc = llama_decode(ctx, batch);
    llama_batch_free(batch);

    if (rc != 0) {
        chat_set_err(err, errlen, "llama_decode failed (%d) at position %d", rc, pos);
        return -1;
    }
    return 0;
}

// Decode a prompt of any length, in `n_batch`-sized chunks.
int chat_decode_prompt(struct llama_context * ctx, const int32_t * tokens, int n,
                       int pos, char * err, size_t errlen) {
    const int n_batch = (int) llama_n_batch(ctx);
    for (int i = 0; i < n; i += n_batch) {
        const int n_chunk = (n - i) < n_batch ? (n - i) : n_batch;
        if (chat_decode_chunk(ctx, tokens + i, n_chunk, pos + i, err, errlen) != 0) {
            return -1;
        }
    }
    return 0;
}

int chat_decode_token(struct llama_context * ctx, int32_t token, int pos,
                      char * err, size_t errlen) {
    return chat_decode_chunk(ctx, &token, 1, pos, err, errlen);
}

struct llama_sampler * chat_sampler_new(int greedy, float temperature, int top_k,
                                        float top_p, float min_p, unsigned int seed) {
    struct llama_sampler_chain_params params = llama_sampler_chain_default_params();
    params.no_perf = true;

    struct llama_sampler * chain = llama_sampler_chain_init(params);
    if (chain == NULL) {
        return NULL;
    }

    if (greedy) {
        llama_sampler_chain_add(chain, llama_sampler_init_greedy());
        return chain;
    }

    if (top_k > 0) {
        llama_sampler_chain_add(chain, llama_sampler_init_top_k(top_k));
    }
    if (top_p > 0.0f && top_p < 1.0f) {
        llama_sampler_chain_add(chain, llama_sampler_init_top_p(top_p, 1));
    }
    if (min_p > 0.0f) {
        llama_sampler_chain_add(chain, llama_sampler_init_min_p(min_p, 1));
    }
    llama_sampler_chain_add(chain, llama_sampler_init_temp(temperature));
    llama_sampler_chain_add(chain, llama_sampler_init_dist((uint32_t) seed));
    return chain;
}

void chat_sampler_free(struct llama_sampler * sampler) {
    if (sampler != NULL) {
        llama_sampler_free(sampler);
    }
}

void chat_sampler_reset(struct llama_sampler * sampler) {
    if (sampler != NULL) {
        llama_sampler_reset(sampler);
    }
}

// Pick the next token from the logits of the last decoded token. Accepts the
// token into the chain, so grammar/penalty state stays consistent.
int32_t chat_sample(struct llama_sampler * sampler, struct llama_context * ctx) {
    return llama_sampler_sample(sampler, ctx, -1);
}

// Token id -> text. Returns the byte count, or -needed when `out` is too small.
int chat_token_text(const struct llama_model * model, int32_t token, char * out, int cap) {
    const struct llama_vocab * vocab = llama_model_get_vocab(model);
    return llama_token_to_piece(vocab, token, out, cap, 0, false);
}

int chat_is_eog(const struct llama_model * model, int32_t token) {
    const struct llama_vocab * vocab = llama_model_get_vocab(model);
    return llama_vocab_is_eog(vocab, token) ? 1 : 0;
}
