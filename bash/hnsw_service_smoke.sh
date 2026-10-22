#!/usr/bin/env bash
#
# Smoke test for src/hnsw_service.nim: build it, start it on a throwaway index,
# embed a few articles, search for them, and clean up again.
#
# Needs YottaDB's environment (ydb_dist / ydb_gbldir, as any tool in this repo)
# and the venv; the script finds hnsw_env/ next to itself.
#
#   bash/hnsw_service_smoke.sh
#   PORT=8099 bash/hnsw_service_smoke.sh
#
set -euo pipefail

repo=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
bin="$repo/src/hnsw_service"
index="^HNSWSvcSmoke"
port="${PORT:-8099}"
venv="$repo/hnsw_env/lib/python3.14/site-packages"
base="http://127.0.0.1:$port"

cd "$repo"
nim c --threads:on --mm:orc -o:"$bin" src/hnsw_service.nim

PYTHONPATH="$venv" "$bin" --index="$index" --reset >/dev/null

PYTHONPATH="$venv" "$bin" --index="$index" --port="$port" &
server=$!
cleanup() {
  kill "$server" 2>/dev/null || true
  wait "$server" 2>/dev/null || true
  # give the first process a moment to release the port before the reset run
  sleep 1
  PYTHONPATH="$venv" "$bin" --index="$index" --reset >/dev/null
}
trap cleanup EXIT

echo "waiting for the service (model load) ..."
for _ in $(seq 120); do
  if curl -sf "$base/" >/dev/null; then break; fi
  sleep 1
done
curl -sf "$base/" >/dev/null || { echo "service did not come up on $base"; exit 1; }

post() { curl -sS -X POST "$base$1" -H 'Content-Type: application/json' -d "$2"; }
expect() { # expect <description> <needle> <haystack>
  if [[ "$3" != *"$2"* ]]; then
    echo "FAILED: $1 (expected '$2' in '$3')"
    exit 1
  fi
  echo "ok: $1"
}

expect "embed new article" '"created":true' \
  "$(post /embed '{"idxref":"smoke-1","title":"YottaDB vector search","description":"The M database gets an HNSW index written in Nim."}')"
expect "embed second article" '"created":true' \
  "$(post /embed '{"idxref":"smoke-2","title":"Baking sourdough","description":"A long cold fermentation makes the bread."}')"

expect "nearest hit is the database article" '"idxref":"smoke-1"' \
  "$(curl -sS "$base/search?q=vector%20database&k=2")"
expect "nearest hit is the bread article" '"idxref":"smoke-2"' \
  "$(curl -sS --get --data-urlencode 'q=bread recipe' --data 'k=2' "$base/search")"
expect "POST search works" '"count":2' \
  "$(post /search '{"query":"database","k":2}')"

expect "upsert replaces the article" '"created":false' \
  "$(post /embed '{"idxref":"smoke-1","title":"HNSW in YottaDB","description":"Replacement text."}')"
expect "replacement is searchable" '"title":"HNSW in YottaDB"' \
  "$(curl -sS "$base/search?q=hnsw&k=1")"

expect "missing field is a 400" "missing field 'description'" \
  "$(post /embed '{"idxref":"x","title":"t"}')"
expect "bad JSON is a 400" "invalid JSON body" \
  "$(post /embed '{oops')"

echo "all smoke tests passed"
