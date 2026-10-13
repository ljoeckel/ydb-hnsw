#!/usr/bin/env bash
#
# install_callm.sh
# ----------------
# Install the YottaDB call-in table (callm.ci) and M routine (callm.m) used by
# nim-yottadb's `updateDBStats()`, then zcompile the routine.
#
# Background
# ==========
# nim-yottadb's `updateDBStats` runs `ydb_ci: "xzshow"`, i.e. it calls the
# YottaDB C call-in interface with the call-in name "xzshow". YottaDB resolves
# that name through the call-in table file referenced by the environment
# variable $ydb_ci, e.g.:
#
#     export ydb_ci=$HOME/.yottadb/r2.06_x86_64/r/callm.ci
#
# The table maps "xzshow" -> the label xzshow^callm inside the M routine callm.m.
# The routine must be compiled (zcompile) so YottaDB can load its object from the
# routine object directory (the first entry of $ydb_routines, e.g.
# .../o/utf8*(...)).
#
# This script:
#   1. installs callm.ci and callm.m into the directory $ydb_ci points to,
#   2. makes sure the routine object directory exists,
#   3. runs `zcompile` on callm.m so its object lands in the routine object
#      directory (e.g. .../o/utf8/callm.o) where YottaDB loads it from.
#
# Usage
# =====
#   ./install_callm.sh                 # use $ydb_ci + embedded routines
#   ./install_callm.sh --src ../../m   # copy callm.ci/callm.m from a directory
#   ./install_callm.sh --ydb-ci /path/to/callm.ci
#   ./install_callm.sh -h | --help
#
set -euo pipefail

prog="$(basename "$0")"
src_dir=""
ci_path="${ydb_ci:-}"

usage() {
    cat <<EOF
Usage: $prog [options]

Installs callm.ci and callm.m into the directory that \$ydb_ci points to,
then runs 'zcompile' on callm.m.

Options:
  --src DIR        Copy callm.ci and callm.m from DIR instead of writing the
                   embedded copies (DIR must contain both files).
  --ydb-ci PATH    Use PATH as the call-in table location instead of \$ydb_ci.
  -h, --help       Show this help.

Environment:
  ydb_ci           Target call-in table path (e.g. \$HOME/.yottadb/r2.06_x86_64/r/callm.ci).
                   If unset, defaults to \$ydb_dir/\$ydb_rel/r/callm.ci and a
                   reminder to export it is printed.
  ydb_dir, ydb_rel, ydb_routines, ydb_dist
                   YottaDB environment (set up via ydb_env_set).
EOF
}

die() { echo "$prog: error: $*" >&2; exit 1; }
warn() { echo "$prog: warning: $*" >&2; }

# ---------------------------------------------------------------------------
# Parse arguments
# ---------------------------------------------------------------------------
while [ $# -gt 0 ]; do
    case "$1" in
        --src)      [ $# -ge 2 ] || die "--src needs a directory"; src_dir="$2"; shift 2 ;;
        --src=*)    src_dir="${1#*=}"; shift ;;
        --ydb-ci)   [ $# -ge 2 ] || die "--ydb-ci needs a path"; ci_path="$2"; shift 2 ;;
        --ydb-ci=*) ci_path="${1#*=}"; shift ;;
        -h|--help)  usage; exit 0 ;;
        --)         shift; break ;;
        *)          die "unknown option '$1' (try --help)" ;;
    esac
done

# ---------------------------------------------------------------------------
# Make sure the YottaDB environment is available (mumps on PATH)
# ---------------------------------------------------------------------------
if ! command -v mumps >/dev/null 2>&1; then
    for envset in "${ydb_dist:-}/ydb_env_set" \
                  /usr/local/lib/yottadb/*/ydb_env_set \
                  /usr/local/etc/ydb_env_set; do
        if [ -f "$envset" ]; then
            # ydb_env_set references some variables unguarded, so relax the
            # strict shell options while sourcing it; restore them afterwards.
            set +eu
            # shellcheck disable=SC1090
            . "$envset" >/dev/null 2>&1
            set -eu
            command -v mumps >/dev/null 2>&1 && break
        fi
    done
fi

YDB_MUMPS="$(command -v mumps 2>/dev/null || true)"
if [ -z "$YDB_MUMPS" ] && [ -x "${ydb_dist:-}/mumps" ]; then
    YDB_MUMPS="${ydb_dist}/mumps"
fi
[ -n "$YDB_MUMPS" ] || die "cannot find the YottaDB 'mumps' executable. Source your YottaDB environment (ydb_env_set) first."

# ---------------------------------------------------------------------------
# Resolve the target call-in table path ($ydb_ci)
# ---------------------------------------------------------------------------
if [ -z "$ci_path" ]; then
    if [ -n "${ydb_dir:-}" ] && [ -n "${ydb_rel:-}" ]; then
        ci_path="${ydb_dir}/${ydb_rel}/r/callm.ci"
        warn "\$ydb_ci is not set; defaulting to $ci_path"
        warn "add this to your shell profile:  export ydb_ci=$ci_path"
    else
        die "\$ydb_ci is not set and \$ydb_dir/\$ydb_rel could not be determined."
    fi
fi

# Expand a leading ~/ (bash does not expand it inside a variable value).
case "$ci_path" in
    "~/"*) ci_path="$HOME/${ci_path#~/}" ;;
    "~")   ci_path="$HOME" ;;
esac

ci_dir="$(dirname "$ci_path")"
rtn_name="callm"                 # ^callm, as referenced by the embedded table
m_file="$ci_dir/${rtn_name}.m"

mkdir -p "$ci_dir"

# ---------------------------------------------------------------------------
# Install callm.ci and callm.m
# ---------------------------------------------------------------------------
if [ -n "$src_dir" ]; then
    [ -d "$src_dir" ]          || die "--src '$src_dir' is not a directory"
    [ -f "$src_dir/callm.m" ]  || die "missing '$src_dir/callm.m'"
    [ -f "$src_dir/callm.ci" ] || die "missing '$src_dir/callm.ci'"
    install -m 0644 "$src_dir/callm.m"  "$m_file"
    install -m 0644 "$src_dir/callm.ci" "$ci_path"
    echo "Copied callm.m  -> $m_file"
    echo "Copied callm.ci -> $ci_path"
else
    cat > "$m_file" <<'M_EOF'
callm   ;
        quit

xzshow  ; Entry point used by nim-yottadb updateDBStats() via ydb_ci
        ZSHOW "G":RESULT
        quit

method1 ; Echo back CTX set by Nim program (CallM macro)
        set key="CTX"
        set i=0
        set RESULT="Traverse RESULT with QueryItr"
        for  set key=$query(@key) quit:key=""  do
        . set value=$get(@key)
        . set RESULT(i)=value
        . set i=i+1
        quit

method2 ; echo back some text with CTX (single argument)
        set RESULT="TheResultFrom YDB CTX="_CTX
        quit

method3 ; echo back some text with CTX (multiple arguments)
        set RESULT="From callin: CTX(1..4)="_CTX(1)_","_CTX(2)_","_CTX(3)_","_CTX(4)
        quit
M_EOF

    cat > "$ci_path" <<'CI_EOF'
method1  : void method1^callm()
method2  : void method2^callm()
method3  : void method3^callm()
xzshow   : void xzshow^callm()
CI_EOF

    chmod 0644 "$m_file" "$ci_path"
    echo "Wrote callm.m  -> $m_file"
    echo "Wrote callm.ci -> $ci_path"
fi

# ---------------------------------------------------------------------------
# Determine the routine object directory (first entry of $ydb_routines)
#   e.g. /home/u/.yottadb/r2.06_x86_64/o/utf8*(/home/u/.yottadb/r2.06_x86_64/r ...)
#   -> object dir is everything before the '*'
# ---------------------------------------------------------------------------
obj_dir=""
rt="${ydb_routines:-${gtmroutines:-}}"
if [ -n "$rt" ]; then
    case "$rt" in
        *\**) obj_dir="${rt%%\**}" ;;   # everything before the first '*'
        *)    obj_dir="${rt%% *}"  ;;   # else the first space-separated field
    esac
fi

# Fallback: derive from $ydb_dir/$ydb_rel.
if [ -z "$obj_dir" ] && [ -n "${ydb_dir:-}" ] && [ -n "${ydb_rel:-}" ]; then
    obj_dir="${ydb_dir}/${ydb_rel}/o"
    [ -d "$obj_dir/utf8" ] && obj_dir="$obj_dir/utf8"
fi

if [ -n "$obj_dir" ]; then
    mkdir -p "$obj_dir"
else
    warn "could not determine the object directory from \$ydb_routines; zcompile will place the object wherever YottaDB decides."
fi

# ---------------------------------------------------------------------------
# zcompile the routine.
#   * pass the ABSOLUTE source path, and
#   * run with cwd = the object directory, because zcompile writes <base>.o into
#     the *current directory* (not next to the source). Running it from the
#     object directory makes the object land in e.g. .../o/utf8/callm.o, which
#     is where YottaDB looks for the compiled routine.
# ---------------------------------------------------------------------------
echo "Compiling $m_file ..."
if [ -n "$obj_dir" ]; then
    ( cd "$obj_dir" && "$YDB_MUMPS" -run %XCMD "zcompile \"$m_file\"" )
else
    "$YDB_MUMPS" -run %XCMD "zcompile \"$m_file\""
fi

# ---------------------------------------------------------------------------
# Verify
# ---------------------------------------------------------------------------
obj_file=""
for candidate in "$obj_dir/${rtn_name}.o" "$ci_dir/${rtn_name}.o"; do
    if [ -n "$candidate" ] && [ -f "$candidate" ]; then
        obj_file="$candidate"
        break
    fi
done

if [ -n "$obj_file" ]; then
    echo "OK: compiled object -> $obj_file"
else
    die "zcompile reported success but no ${rtn_name}.o could be found (looked in '${obj_dir:-?}' and '$ci_dir')."
fi

cat <<EOF

Done.
  call-in table : $ci_path
  M routine     : $m_file
  object file   : $obj_file

Make sure the call-in table is exported before running programs that use it:
  export ydb_ci=$ci_path
and restart any already-running processes so they pick up the new routine.
EOF
