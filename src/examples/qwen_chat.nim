## Chat with a local GGUF model - a minimal REPL on top of `llama_chat.nim`.
##
##   nim c -d:release src/examples/qwen_chat.nim
##   ./src/examples/qwen_chat
##
## Defaults to ~/models/gguf/Qwen3-1.7B-Q4_K_M.gguf; pass `--model` or set
## QWEN_GGUF to use another file. Type a message and press enter, `/reset`
## forgets the conversation, `/exit` (or Ctrl-D) quits. Passing `-p "..."` sends
## a single prompt and exits, which is handy for scripting.
##
## `-a text1.txt -b text2.txt` builds a merge prompt from the two files, sends it
## once and prints only the merged text on stdout (`-a -` reads stdin):
##
##   qwen_chat -a part1.md -b part2.md --temp 0.3 > merged.md

import std/[monotimes, os, parseopt, strformat, strutils, times]

import ../llama_chat

proc usage() =
  echo """
qwen_chat - talk to a local GGUF model

  -m, --model PATH    GGUF file (default $QWEN_GGUF or ~/models/gguf/Qwen3-1.7B-Q4_K_M.gguf)
  -p, --prompt TEXT   answer one prompt and exit
  -a, --text-a PATH   first text to merge ('-' reads stdin)
  -b, --text-b PATH   second text; together with -a the merge prompt is built,
                      sent once, and only the result is printed on stdout
      --task TEXT     instruction used instead of "merge the two texts"
  -s, --system TEXT   system prompt
      --ctx N         context size in tokens (default 4096)
      --max-tokens N  generation budget per answer (default 512)
      --temp F        temperature (default 0.7, or 0.6 with --think)
      --top-p F       nucleus sampling (default 0.8, or 0.95 with --think)
      --greedy        always take the most likely token
      --think         let the model reason in a <think> block first
      --no-gpu        run on the CPU only
      --verbose       show llama.cpp's load/decode logging
  -h, --help          this text"""

const valueOptions = ["m", "model", "p", "prompt", "s", "system", "ctx",
                      "max-tokens", "temp", "top-p",
                      "a", "text-a", "b", "text-b", "task"]

const MergeInstruction = """Fasse die folgenden zwei Texte zu einem einzigen, flüssigen Text zusammen.

Regeln:
- Alle Fakten aus beiden Texten bleiben erhalten, nichts wird erfunden.
- Doppelungen zusammenführen, Stil und Zeitform vereinheitlichen.
- Länge: etwa wie der längere der beiden Texte.
- Antworte nur mit dem zusammengeführten Text, ohne Vorrede."""
  ## The prompt `-a`/`-b` sends; `--task` replaces it wholesale.

type
  Args = object
    opts: ChatOptions
    prompt: string       ## -p: one turn from the command line
    textA, textB: string ## -a/-b: the two texts to merge (paths, '-' = stdin)
    task: string         ## --task: instruction instead of MergeInstruction
    help: bool

proc readInput(path: string): string =
  ## `-` means stdin, everything else a file path.
  if path == "-":
    return stdin.readAll()
  if not fileExists(path):
    raise newException(IOError, "file not found: " & path)
  readFile(path)

proc mergePrompt(first, second, instruction: string): string =
  ## Both texts are fenced, so the model cannot mistake content for instruction.
  result = instruction.strip & "\n\nTEXT A:\n\"\"\"\n" & first.strip &
           "\n\"\"\"\n\nTEXT B:\n\"\"\"\n" & second.strip & "\n\"\"\""

proc parseArgs(): Args =
  result.opts = defaultChatOptions()
  result.opts.systemPrompt = "You are a concise, friendly assistant."
  var thinking = false

  # Nim's parseopt only fills `val` for `--key=value`, so the args are collected
  # first and `--ctx 8192` is stitched back together here.
  var items: seq[tuple[kind: CmdLineKind, key: string, val: string]]
  var p = initOptParser()
  for kind, key, val in p.getopt():
    items.add (kind, key, val)

  var i = 0
  while i < items.len:
    let (kind, key, val) = items[i]
    inc i
    case kind
    of cmdArgument:
      if result.prompt.len == 0: result.prompt = key
    of cmdShortOption, cmdLongOption:
      if key.len == 0:
        # A lone "-" arrives from parseopt as an option with an empty key.
        if result.prompt.len == 0: result.prompt = "-"
        continue

      var value = val
      if value.len == 0 and key in valueOptions and i < items.len:
        let next = items[i]
        if next.kind == cmdArgument:
          value = next.key
          inc i
        elif next.kind in {cmdShortOption, cmdLongOption} and next.key.len == 0:
          value = "-"   # `-a -` / `-b -`: read this text from stdin
          inc i

      case key
      of "m", "model": result.opts.modelPath = value
      of "p", "prompt": result.prompt = value
      of "a", "text-a": result.textA = value
      of "b", "text-b": result.textB = value
      of "task": result.task = value
      of "s", "system": result.opts.systemPrompt = value
      of "ctx": result.opts.nCtx = parseInt(value)
      of "max-tokens": result.opts.maxTokens = parseInt(value)
      of "temp": result.opts.temperature = parseFloat(value).float32
      of "top-p": result.opts.topP = parseFloat(value).float32
      of "greedy": result.opts.greedy = true
      of "think": thinking = true
      of "no-gpu": result.opts.gpuLayers = 0
      of "verbose": result.opts.quiet = false
      of "h", "help": result.help = true
      else:
        stderr.writeLine "unknown option: " & key
        result.help = true
    of cmdEnd:
      discard

  result.opts.thinking = thinking
  if thinking and not result.opts.greedy:
    # Qwen3's recommendation changes with the mode; these are the "thinking"
    # values, the no-think ones are already the defaults in ChatOptions.
    result.opts.temperature = 0.6
    result.opts.topP = 0.95

proc describeSpeed(stats: ChatStats, elapsed: Duration): string =
  let seconds = elapsed.inMicroseconds.float / 1_000_000.0
  var line = &"{stats.genTokens} tokens"
  if seconds > 0:
    line.add &" in {seconds:.2f}s ({stats.genTokens.float / seconds:.1f} tok/s)"
  line.add &", prompt {stats.promptTokens} tokens"
  line

proc answer(chat: ChatModel, question: string, prefix = true) =
  if prefix:
    stdout.write "bot> "
    stdout.flushFile()

  let start = getMonoTime()
  discard chat.ask(question, proc (piece: string) =
    stdout.write piece
    stdout.flushFile())
  let elapsed = getMonoTime() - start

  echo ""
  stderr.writeLine "     [" & describeSpeed(chat.stats, elapsed) & "]"

proc main() =
  let args = parseArgs()
  if args.help:
    usage()
    return

  # Build the one-shot prompt before the model is loaded, so a wrong path or a
  # bad option costs nothing.
  var prompt = args.prompt
  var batch = false
  if args.textA.len > 0 or args.textB.len > 0:
    if args.textA.len == 0 or args.textB.len == 0:
      stderr.writeLine "-a and -b belong together: one input file each"
      quit(QuitFailure)
    if args.textA == "-" and args.textB == "-":
      stderr.writeLine "only one of -a/-b can read stdin"
      quit(QuitFailure)
    let instruction = if args.task.len > 0: args.task else: MergeInstruction
    try:
      prompt = mergePrompt(readInput(args.textA), readInput(args.textB), instruction)
    except IOError as error:
      stderr.writeLine "cannot read the input: " & error.msg
      quit(QuitFailure)
    batch = true

  stderr.writeLine &"loading {args.opts.modelPath.extractFilename} ..."
  let loadStart = getMonoTime()
  let chat = loadChat(args.opts)
  defer: chat.closeChat()
  stderr.writeLine &"ready in {(getMonoTime() - loadStart).inMilliseconds} ms" &
    (if args.opts.thinking: " (thinking on)" else: "")

  if prompt.len > 0:
    # `-a`/`-b` is batch mode: stdout carries the result alone, so
    # `qwen_chat -a x -b y > merged.md` produces a clean file.
    answer(chat, prompt, prefix = not batch)
    return

  stderr.writeLine "type a message, /reset to start over, /exit to quit"
  while true:
    stdout.write "you> "
    stdout.flushFile()
    let line = stdin.readLine()
    if line.len == 0:
      # readLine() returns "" on EOF, and on a bare enter we just ask again.
      if endOfFile(stdin): break
      continue

    case line.strip()
    of "/exit", "/quit": break
    of "/reset":
      chat.resetChat()
      echo "(conversation cleared)"
    else:
      answer(chat, line)

when isMainModule:
  main()
