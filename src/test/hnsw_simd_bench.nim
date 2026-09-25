## Microbenchmark for the three kernels `hnsw.nim` spends its time in:
## the float32 dot product (`dotStored` in vqNone mode), the int8 dot product
## (`dotInt8` in the quantized modes) and `normalize`.
##
## The point is a two-way comparison in one binary - the scalar reference and the
## intrinsics - so the gain can be attributed:
##
##   nim c -r -d:release src/test/hnsw_simd_bench.nim                 # both
##   nim c -r -d:release -d:hnswNoSimd src/test/hnsw_simd_bench.nim   # scalar only
##
## `-mavx` comes from the project's config.nims whenever `hnswSimd` is defined.
## `-march=native` on its own does *not* speed these loops up: they are
## reductions, and gcc will not reassociate them without `-ffast-math`.

import std/[math, random, strformat, times]

const
  Dim = 384                 # the article index's dimension
  Reps = 20_000             # dot products per timing round
  Rounds = 5

when defined(hnswSimd):
  import nimsimd/avx      # pulls in sse42 -> sse41 -> sse3 -> sse2 as well

# ---------------------------------------------------------------------------
# reference (current) implementations
# ---------------------------------------------------------------------------

proc dotScalar(a, b: openArray[float32]): float32 =
  for i in 0 ..< a.len:
    result += a[i] * b[i]

proc dotInt8Scalar(a, b: openArray[int8]): int32 =
  var acc = 0'i32
  for i in 0 ..< a.len:
    acc += int32(a[i]) * int32(b[i])
  acc

proc normScalar(v: var seq[float32]) =
  var sum = 0.0'f32
  for x in v:
    sum += x * x
  let n = sqrt(sum)
  if n > 0:
    for i in 0 ..< v.len:
      v[i] = v[i] / n

# ---------------------------------------------------------------------------
# AVX (256-bit float) / SSE4.1 (128-bit int8) implementations
# ---------------------------------------------------------------------------

when defined(hnswSimd):
  proc dotAvx(a, b: openArray[float32]): float32 =
    ## Four independent accumulators: a single chain of `mul; add` is bound by
    ## the add's latency (~4 cycles) and would not reach one multiply per cycle.
    let n = a.len
    var i = 0
    var acc0 = mm256_setzero_ps()
    var acc1 = mm256_setzero_ps()
    var acc2 = mm256_setzero_ps()
    var acc3 = mm256_setzero_ps()
    while i + 32 <= n:
      acc0 = mm256_add_ps(acc0, mm256_mul_ps(mm256_loadu_ps(a[i].unsafeAddr),
                                            mm256_loadu_ps(b[i].unsafeAddr)))
      acc1 = mm256_add_ps(acc1, mm256_mul_ps(mm256_loadu_ps(a[i + 8].unsafeAddr),
                                            mm256_loadu_ps(b[i + 8].unsafeAddr)))
      acc2 = mm256_add_ps(acc2, mm256_mul_ps(mm256_loadu_ps(a[i + 16].unsafeAddr),
                                            mm256_loadu_ps(b[i + 16].unsafeAddr)))
      acc3 = mm256_add_ps(acc3, mm256_mul_ps(mm256_loadu_ps(a[i + 24].unsafeAddr),
                                            mm256_loadu_ps(b[i + 24].unsafeAddr)))
      i += 32
    while i + 8 <= n:
      acc0 = mm256_add_ps(acc0, mm256_mul_ps(mm256_loadu_ps(a[i].unsafeAddr),
                                            mm256_loadu_ps(b[i].unsafeAddr)))
      i += 8
    var acc = mm256_add_ps(mm256_add_ps(acc0, acc1), mm256_add_ps(acc2, acc3))
    var s = mm_add_ps(mm256_castps256_ps128(acc),
                      mm256_extractf128_ps(acc, 1))
    s = mm_hadd_ps(s, s)
    s = mm_hadd_ps(s, s)
    var buf: array[4, float32]
    mm_storeu_ps(buf[0].addr, s)
    result = buf[0]
    while i < n:
      result += a[i] * b[i]
      inc i

  proc dotInt8Avx(a, b: openArray[int8]): int32 =
    ## `pmaddwd` multiplies int16 pairs and adds them into int32, so an int8
    ## operand has to be widened first: 16 bytes are loaded, the low and high
    ## halves sign-extended to int16 (`pmovsxbw`) and each half folded by one
    ## `pmaddwd`. The accumulators stay exact - 127*127*4096 is far below 2^31.
    let n = a.len
    var i = 0
    var acc0 = mm_setzero_si128()
    var acc1 = mm_setzero_si128()
    while i + 32 <= n:
      let a0 = mm_loadu_si128(a[i].unsafeAddr)
      let b0 = mm_loadu_si128(b[i].unsafeAddr)
      acc0 = mm_add_epi32(acc0, mm_madd_epi16(mm_cvtepi8_epi16(a0), mm_cvtepi8_epi16(b0)))
      acc0 = mm_add_epi32(acc0, mm_madd_epi16(mm_cvtepi8_epi16(mm_srli_si128(a0, 8)),
                                             mm_cvtepi8_epi16(mm_srli_si128(b0, 8))))
      let a1 = mm_loadu_si128(a[i + 16].unsafeAddr)
      let b1 = mm_loadu_si128(b[i + 16].unsafeAddr)
      acc1 = mm_add_epi32(acc1, mm_madd_epi16(mm_cvtepi8_epi16(a1), mm_cvtepi8_epi16(b1)))
      acc1 = mm_add_epi32(acc1, mm_madd_epi16(mm_cvtepi8_epi16(mm_srli_si128(a1, 8)),
                                             mm_cvtepi8_epi16(mm_srli_si128(b1, 8))))
      i += 32
    var acc = mm_add_epi32(acc0, acc1)
    var t: array[4, int32]
    mm_storeu_si128(t[0].addr, acc)
    result = t[0] + t[1] + t[2] + t[3]
    while i + 16 <= n:
      let a0 = mm_loadu_si128(a[i].unsafeAddr)
      let b0 = mm_loadu_si128(b[i].unsafeAddr)
      let s = mm_add_epi32(mm_madd_epi16(mm_cvtepi8_epi16(a0), mm_cvtepi8_epi16(b0)),
                           mm_madd_epi16(mm_cvtepi8_epi16(mm_srli_si128(a0, 8)),
                                         mm_cvtepi8_epi16(mm_srli_si128(b0, 8))))
      mm_storeu_si128(t[0].addr, s)
      result += t[0] + t[1] + t[2] + t[3]
      i += 16
    while i < n:
      result += int32(a[i]) * int32(b[i])
      inc i

  proc normAvx(v: var seq[float32]) =
    let n = v.len
    var i = 0
    var acc0 = mm256_setzero_ps()
    var acc1 = mm256_setzero_ps()
    while i + 16 <= n:
      let x0 = mm256_loadu_ps(v[i].addr)
      acc0 = mm256_add_ps(acc0, mm256_mul_ps(x0, x0))
      let x1 = mm256_loadu_ps(v[i + 8].addr)
      acc1 = mm256_add_ps(acc1, mm256_mul_ps(x1, x1))
      i += 16
    var acc = mm256_add_ps(acc0, acc1)
    var s = mm_add_ps(mm256_castps256_ps128(acc), mm256_extractf128_ps(acc, 1))
    s = mm_hadd_ps(s, s)
    s = mm_hadd_ps(s, s)
    var buf: array[4, float32]
    mm_storeu_ps(buf[0].addr, s)
    var sum = buf[0]
    while i < n:
      sum += v[i] * v[i]
      inc i
    let nm = sqrt(sum)
    if nm > 0:
      let inv = mm256_set1_ps(1.0'f32 / nm)
      i = 0
      while i + 8 <= n:
        mm256_storeu_ps(v[i].addr, mm256_mul_ps(mm256_loadu_ps(v[i].addr), inv))
        i += 8
      while i < n:
        v[i] = v[i] / nm
        inc i

# ---------------------------------------------------------------------------
# harness
# ---------------------------------------------------------------------------

var sink = 0.0'f64   # keeps the optimizer from deleting the measured loops

proc bench(tag: string, bytesPerCall: int, body: proc()): float64 =
  ## Fastest of `Rounds` runs, in nanoseconds per call.
  var best = Inf
  for _ in 0 ..< Rounds:
    let t0 = epochTime()
    body()
    let dt = (epochTime() - t0) * 1e9 / float64(Reps)
    if dt < best:
      best = dt
  echo &"  {tag:<22} {best:8.1f} ns/op  " &
       &"{float64(bytesPerCall) / (best * 1e-9) / 1e9:7.2f} GB/s  " &
       &"({best * float64(Reps) / 1e6:7.1f} ms/round)"
  best

proc main() =
  var rng = initRand(42)
  var fa = newSeq[float32](Dim)
  var fb = newSeq[float32](Dim)
  for i in 0 ..< Dim:
    fa[i] = float32(2.0 * rand(rng, 1.0) - 1.0)
    fb[i] = float32(2.0 * rand(rng, 1.0) - 1.0)
  var qa = newSeq[int8](Dim)
  var qb = newSeq[int8](Dim)
  for i in 0 ..< Dim:
    qa[i] = int8(rand(rng, 254) - 127)
    qb[i] = int8(rand(rng, 254) - 127)

  echo &"dim {Dim}, {Reps} calls/round, best of {Rounds}"
  when defined(hnswSimd):
    echo "kernels: scalar + nimsimd intrinsics"
  else:
    echo "kernels: scalar only"

  # correctness first: a fast kernel that disagrees is worthless
  when defined(hnswSimd):
    doAssert abs(dotAvx(fa, fb) - dotScalar(fa, fb)) < 1e-3, "dotAvx mismatch"
    doAssert dotInt8Avx(qa, qb) == dotInt8Scalar(qa, qb), "dotInt8Avx mismatch"
    var v1 = fa
    var v2 = fa
    normScalar(v1)
    normAvx(v2)
    for i in 0 ..< Dim:
      doAssert abs(v1[i] - v2[i]) < 1e-5, "normAvx mismatch"
    echo "  (intrinsics agree with the scalar reference)"

  let fbytes = 2 * Dim * sizeof(float32)
  let ibytes = 2 * Dim
  echo "\ndot float32 (vqNone):"
  discard bench("scalar", fbytes, proc() =
    var acc = 0.0'f32
    for _ in 0 ..< Reps: acc += dotScalar(fa, fb)
    sink += acc)
  when defined(hnswSimd):
    discard bench("nimsimd avx", fbytes, proc() =
      var acc = 0.0'f32
      for _ in 0 ..< Reps: acc += dotAvx(fa, fb)
      sink += acc)

  echo "\ndot int8 (vqInt8 / vqInt8Fixed):"
  discard bench("scalar", ibytes, proc() =
    var acc = 0'i32
    for _ in 0 ..< Reps: acc += dotInt8Scalar(qa, qb)
    sink += acc.float64)
  when defined(hnswSimd):
    discard bench("nimsimd sse4.1", ibytes, proc() =
      var acc = 0'i32
      for _ in 0 ..< Reps: acc += dotInt8Avx(qa, qb)
      sink += acc.float64)

  echo "\nnormalize:"
  let nbytes = 2 * Dim * sizeof(float32)
  discard bench("scalar", nbytes, proc() =
    var v = fa
    for _ in 0 ..< Reps:
      normScalar(v)
    sink += v[0])
  when defined(hnswSimd):
    discard bench("nimsimd avx", nbytes, proc() =
      var v = fa
      for _ in 0 ..< Reps:
        normAvx(v)
      sink += v[0])

main()
