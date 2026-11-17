## Chat with a GGUF model through llama.cpp - the generation counterpart of
## `llama_embed.nim`.
##
##   let chat = loadChat(ChatOptions(modelPath: "Qwen3-1.7B-Q4_K_M.gguf"))
##   defer: chat.closeChat
##   echo chat.ask("Why is the sky blue?")
##
## `loadChat` opens the model and a decode context, `ask` runs one turn - the
## conversation history is kept inside the `ChatModel`, so a REPL just calls
## `ask` in a loop - and `closeChat` releases everything. Answers are streamed
## token by token to an optional callback, which is what makes the output appear
## while the model is still thinking.
##
## The C side lives in `llama_chat_shim.c`: llama.h passes several structs by
## value and those layouts change between releases, so Nim never mirrors them.
## Linking works the same way as for the embedding backend - `config.nims`
## passes `-d:llamaCppDir=<llama.cpp build dir>` from `LLAMA_CPP_DIR` (default
## `~/git/llama.cpp/build`).

import std/[os, strutils]

# Where llama.cpp lives. `config.nims` passes the build directory through
# `-d:llamaCppDir=<build>`; without it we assume a system-wide install.
const llamaCppDir {.strdefine.} = ""
when llamaCppDir.len > 0:
  const llamaBin = llamaCppDir / "bin"
  # llama.h pulls in ggml.h, which lives in the ggml subdirectory.
  {.passC: "-I" & (llamaCppDir / ".." / "include") &
            " -I" & (llamaCppDir / ".." / "ggml" / "include").}
  # -lggml/-lggml-base are needed explicitly: libllama.so reaches ggml symbols
  # directly and ld wants them named on the link line.
  {.passL: "-L" & llamaBin & " -lllama -lggml -lggml-base -Wl,-rpath," & llamaBin.}
else:
  {.passL: "-lllama".}

{.compile: "llama_chat_shim.c".}

const
  DefaultCtx = 4096
    ## tokens of conversation the KV cache holds; Qwen3-1.7B was trained on 40960
  DefaultBatch = 512
    ## prompt tokens per llama_decode(); must be >= the longest prompt chunk
  DefaultMaxTokens = 512
    ## tokens generated per answer, thinking included
  DefaultTopK = 20
  ThinkOff = "<think>\n\n</think>\n\n"
    ## Qwen3's own template appends this after `<|im_start|>assistant\n` when
    ## `enable_thinking` is false, which switches reasoning off. llama.cpp's
    ## template list has no option for that variable, so it is appended by hand.
  ErrLen = 512
  PieceLen = 64

type
  Model = pointer   ## struct llama_model *
  Context = pointer ## struct llama_context *
  Sampler = pointer ## struct llama_sampler * (a sampler chain)

  ChatTurn {.bycopy.} = object
    ## Mirrors `chat_turn` in llama_chat_shim.c.
    role: cstring
    content: cstring

  ChatError* = object of CatchableError

  ChatMessage* = object
    role*: string     ## "system", "user" or "assistant"
    content*: string

  ChatOptions* = object
    modelPath*: string       ## GGUF file to load
    systemPrompt*: string    ## prepended as a system turn when not empty
    nCtx*: int               ## KV cache size in tokens, 0 = model default
    nBatch*: int             ## prompt tokens decoded per step
    nThreads*: int           ## compute threads, 0 = let llama.cpp decide
    gpuLayers*: int          ## -1 = all layers on the GPU, 0 = CPU only
    maxTokens*: int          ## generation budget per answer
    temperature*: float32
    topK*: int               ## 0 = disabled
    topP*: float32           ## 1.0 = disabled
    minP*: float32           ## 0.0 = disabled
    seed*: uint32
    greedy*: bool            ## ignore the sampling knobs, always take the top token
    thinking*: bool          ## let the model emit a `<think>` block first
    quiet*: bool             ## silence llama.cpp's load/decode logging

  ChatStats* = object
    promptTokens*: int       ## tokens of the last prompt (history included)
    genTokens*: int          ## tokens generated for the last answer

  TokenCallback* = proc(token: string) {.closure.}

  ChatModel* = ref object
    model: Model
    ctx: Context
    sampler: Sampler
    opts: ChatOptions
    history: seq[ChatMessage]
    chatTemplate: string     ## the model's chat template, "" when it has none
    stats*: ChatStats

# --- the C shim (src/llama_chat_shim.c) -------------------------------------

proc cBackendInit(quiet: cint) {.importc: "chat_backend_init", cdecl.}
proc cLoadModel(path: cstring, nGpuLayers: cint, err: cstring, errLen: csize_t): Model
  {.importc: "chat_load_model", cdecl.}
proc cFreeModel(model: Model) {.importc: "chat_free_model", cdecl.}
proc cNewContext(model: Model, nCtx, nBatch, nThreads: cint,
                 err: cstring, errLen: csize_t): Context
  {.importc: "chat_new_context", cdecl.}
proc cFreeContext(ctx: Context) {.importc: "chat_free_context", cdecl.}
proc cNCtx(ctx: Context): cint {.importc: "chat_n_ctx", cdecl.}
proc cClearKv(ctx: Context) {.importc: "chat_clear_kv", cdecl.}
proc cModelTemplate(model: Model): cstring {.importc: "chat_model_template", cdecl.}
proc cFormatPrompt(model: Model, turns: ptr ChatTurn, nTurns: cint, addAssistant: cint,
                   outBuf: cstring, cap: cint): cint
  {.importc: "chat_format_prompt", cdecl.}
proc cTokenize(model: Model, text: cstring, textLen: cint, outIds: ptr int32,
               cap: cint): cint
  {.importc: "chat_tokenize", cdecl.}
proc cDecodePrompt(ctx: Context, tokens: ptr int32, n, pos: cint,
                   err: cstring, errLen: csize_t): cint
  {.importc: "chat_decode_prompt", cdecl.}
proc cDecodeToken(ctx: Context, token: int32, pos: cint,
                  err: cstring, errLen: csize_t): cint
  {.importc: "chat_decode_token", cdecl.}
proc cSamplerNew(greedy: cint, temperature: cfloat, topK: cint, topP, minP: cfloat,
                 seed: cuint): Sampler
  {.importc: "chat_sampler_new", cdecl.}
proc cSamplerFree(sampler: Sampler) {.importc: "chat_sampler_free", cdecl.}
proc cSamplerReset(sampler: Sampler) {.importc: "chat_sampler_reset", cdecl.}
proc cSample(sampler: Sampler, ctx: Context): int32 {.importc: "chat_sample", cdecl.}
proc cTokenText(model: Model, token: int32, outBuf: cstring, cap: cint): cint
  {.importc: "chat_token_text", cdecl.}
proc cIsEog(model: Model, token: int32): cint {.importc: "chat_is_eog", cdecl.}

# --- helpers ----------------------------------------------------------------

proc defaultChatOptions*(): ChatOptions =
  ## Qwen3's recommended settings: 0.6/0.95/20 while thinking, 0.7/0.8/20 when
  ## the answer comes straight out.
  result = ChatOptions(
    modelPath: getEnv("QWEN_GGUF", getHomeDir() / "models" / "gguf" / "Qwen3-1.7B-Q4_K_M.gguf"),
    nCtx: DefaultCtx,
    nBatch: DefaultBatch,
    nThreads: 0,
    gpuLayers: -1,
    maxTokens: DefaultMaxTokens,
    temperature: 0.7,
    topK: DefaultTopK,
    topP: 0.8,
    minP: 0.0,
    seed: 0,
    greedy: false,
    thinking: false,
    quiet: true)

proc fail(what: string, err: cstring) {.noreturn.} =
  raise newException(ChatError, what & ": " & (if err == nil: "unknown error" else: $err))

proc check(rc: cint, err: cstring, what: string) =
  if rc < 0:
    fail(what, err)

proc modelTemplate(chat: ChatModel): string =
  let tmpl = cModelTemplate(chat.model)
  if tmpl == nil: "" else: $tmpl

proc formatPrompt(chat: ChatModel, addAssistant: bool): string =
  ## The whole conversation as the model's chat template renders it.
  if chat.chatTemplate.len == 0:
    raise newException(ChatError,
      "the GGUF file has no chat template; llama.cpp cannot build a prompt")

  var turns = newSeq[ChatTurn](chat.history.len)
  for i, message in chat.history:
    turns[i] = ChatTurn(role: message.role.cstring, content: message.content.cstring)
  let turnPtr = if turns.len > 0: addr turns[0] else: nil

  var cap = 0
  for message in chat.history:
    cap += message.role.len + message.content.len + 16
  cap = cap * 2 + 64

  result = newString(cap)
  let n = cFormatPrompt(chat.model, turnPtr, cint(turns.len), cint(addAssistant),
                        cstring(result), cint(cap))
  if n < 0:
    raise newException(ChatError,
      "llama.cpp does not recognise this model's chat template")
  result.setLen(n)

proc tokenize(chat: ChatModel, text: string): seq[int32] =
  if text.len == 0:
    return @[]
  # A token is never longer than a byte, plus a few specials: start there and
  # let llama.cpp ask for more on the rare miss.
  var cap = text.len + 8
  while true:
    result = newSeq[int32](cap)
    let n = cTokenize(chat.model, text.cstring, cint(text.len),
                      addr result[0], cint(cap))
    if n >= 0:
      result.setLen(n)
      return
    cap = -n

proc tokenText(chat: ChatModel, token: int32): string =
  var buf = newString(PieceLen)
  let n = cTokenText(chat.model, token, cstring(buf), cint(buf.len))
  if n >= 0:
    buf.setLen(n)
    return buf
  buf = newString(-n)
  let again = cTokenText(chat.model, token, cstring(buf), cint(buf.len))
  if again < 0:
    return ""
  buf.setLen(again)
  buf

proc stripThinking(text: string): string =
  ## Drop a leading reasoning block. Only the visible answer is kept in the
  ## history: Qwen3's template would re-format it as reasoning anyway, and the
  ## block is usually most of the token count.
  const closeTag = "</think>"
  let i = text.find(closeTag)
  if i < 0:
    return text
  result = text[i + closeTag.len .. ^1].strip()

# --- public API -------------------------------------------------------------

proc loadChat*(opts: ChatOptions): ChatModel =
  ## Load the GGUF and create the decode context. The sampling chain is created
  ## here too, so `ask` never has to rebuild it.
  if not fileExists(opts.modelPath):
    raise newException(ChatError, "model file not found: " & opts.modelPath)

  result = ChatModel(opts: opts)
  cBackendInit(cint(ord(opts.quiet)))

  var err = newString(ErrLen)
  var gpuLayers = opts.gpuLayers
  result.model = cLoadModel(opts.modelPath.cstring, cint(gpuLayers),
                            cstring(err), csize_t(err.len))
  if result.model == nil and gpuLayers != 0:
    # No VRAM / driver mismatch - a 1.7B Q4 model on the CPU is slow but works.
    stderr.writeLine "llama_chat: GPU load failed (" & $err & "), retrying on the CPU"
    gpuLayers = 0
    result.model = cLoadModel(opts.modelPath.cstring, cint(gpuLayers),
                              cstring(err), csize_t(err.len))
  if result.model == nil:
    fail("cannot load " & opts.modelPath, cstring(err))

  result.ctx = cNewContext(result.model, cint(opts.nCtx), cint(opts.nBatch),
                           cint(opts.nThreads), cstring(err), csize_t(err.len))
  if result.ctx == nil:
    cFreeModel(result.model)
    result.model = nil
    fail("cannot create the context", cstring(err))

  let seed = if opts.seed == 0: 0xFFFFFFFF'u32 else: opts.seed
  result.sampler = cSamplerNew(cint(ord(opts.greedy)), cfloat(opts.temperature),
                               cint(opts.topK), cfloat(opts.topP),
                               cfloat(opts.minP), cuint(seed))
  if result.sampler == nil:
    cFreeContext(result.ctx)
    cFreeModel(result.model)
    result.ctx = nil
    result.model = nil
    raise newException(ChatError, "cannot create the sampler chain")

  result.chatTemplate = result.modelTemplate
  if opts.systemPrompt.len > 0:
    result.history.add ChatMessage(role: "system", content: opts.systemPrompt)

proc closeChat*(chat: ChatModel) =
  ## Free the sampler, the context and the model. The llama.cpp backend itself
  ## stays initialised - the process is about to exit anyway.
  if chat == nil:
    return
  if chat.sampler != nil:
    cSamplerFree(chat.sampler)
    chat.sampler = nil
  if chat.ctx != nil:
    cFreeContext(chat.ctx)
    chat.ctx = nil
  if chat.model != nil:
    cFreeModel(chat.model)
    chat.model = nil

proc resetChat*(chat: ChatModel) =
  ## Forget the conversation, keeping the system prompt.
  chat.history.setLen(0)
  if chat.opts.systemPrompt.len > 0:
    chat.history.add ChatMessage(role: "system", content: chat.opts.systemPrompt)
  cClearKv(chat.ctx)
  cSamplerReset(chat.sampler)

proc history*(chat: ChatModel): seq[ChatMessage] =
  ## The turns the next prompt will be built from.
  chat.history

proc templateText*(chat: ChatModel): string =
  ## The chat template exactly as stored in the GGUF metadata.
  chat.chatTemplate

proc ask*(chat: ChatModel, prompt: string, onToken: TokenCallback = nil): string =
  ## One conversational turn: append `prompt`, re-decode the whole conversation
  ## and generate up to `maxTokens` tokens, streaming each one to `onToken`
  ## (thinking included). Returns the full answer and appends it to the history.
  if chat.model == nil:
    raise newException(ChatError, "the model is closed")

  chat.history.add ChatMessage(role: "user", content: prompt)

  var text = chat.formatPrompt(addAssistant = true)
  if not chat.opts.thinking and "enable_thinking" in chat.chatTemplate:
    # Qwen3-style template: an already closed `<think>` block means "no reasoning".
    text.add ThinkOff

  let tokens = chat.tokenize(text)
  if tokens.len == 0:
    raise newException(ChatError, "the prompt tokenized to nothing")

  let nCtx = int(cNCtx(chat.ctx))
  if tokens.len >= nCtx:
    raise newException(ChatError,
      "the conversation does not fit the context (" & $tokens.len & " of " & $nCtx &
      " tokens); raise nCtx or reset the chat")

  var err = newString(ErrLen)

  # The KV cache is cleared and the full conversation re-decoded every turn,
  # so nothing has to be tracked between turns. Prompt caching (keeping the
  # common token prefix) is the obvious optimisation for long chats.
  cClearKv(chat.ctx)
  check(cDecodePrompt(chat.ctx, addr tokens[0], cint(tokens.len), 0,
                      cstring(err), csize_t(err.len)), cstring(err), "prompt decode")

  chat.stats = ChatStats(promptTokens: tokens.len, genTokens: 0)

  var pos = tokens.len
  while chat.stats.genTokens < chat.opts.maxTokens and pos < nCtx:
    let token = cSample(chat.sampler, chat.ctx)
    if cIsEog(chat.model, token) != 0:
      break

    let piece = chat.tokenText(token)
    inc chat.stats.genTokens
    result.add piece
    if onToken != nil and piece.len > 0:
      onToken(piece)

    if chat.stats.genTokens >= chat.opts.maxTokens:
      break
    check(cDecodeToken(chat.ctx, token, cint(pos), cstring(err), csize_t(err.len)),
          cstring(err), "token decode")
    inc pos

  chat.history.add ChatMessage(role: "assistant", content: stripThinking(result))
