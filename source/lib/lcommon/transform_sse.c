
/*!
 ***************************************************************************
 * \file transform_sse.c
 *
 * \brief
 *    SSE2 / SSSE3 implementations of the H.264 inverse 4x4 and
 *    inverse 8x8 integer transforms.
 *
 *    Each function here is a drop-in replacement for the corresponding
 *    scalar version in transform.c. Selected at runtime via the
 *    jm_simd dispatch table (see app/ldecod/jm_simd.c).
 *
 *    These are IMGTYPE-agnostic: they operate on int** coefficient and
 *    residual arrays, not on pixels. The transform output range fits
 *    int32 for all H.264 bit depths.
 ***************************************************************************
 */

#include <emmintrin.h> /* SSE2 */

/*!
 ***************************************************************************
* inverse4x4
 ***************************************************************************
 */

/* 4x4 transpose of 32-bit integer lanes.
   Before: r[i].lane[j] = M[i][j]
   After:  r[i].lane[j] = M[j][i] */
#define TRANSPOSE4_EPI32_SSE(r0, r1, r2, r3)                                   \
  do {                                                                         \
    __m128i _a = _mm_unpacklo_epi32(r0, r1);                                   \
    __m128i _b = _mm_unpackhi_epi32(r0, r1);                                   \
    __m128i _c = _mm_unpacklo_epi32(r2, r3);                                   \
    __m128i _d = _mm_unpackhi_epi32(r2, r3);                                   \
    r0 = _mm_unpacklo_epi64(_a, _c);                                           \
    r1 = _mm_unpackhi_epi64(_a, _c);                                           \
    r2 = _mm_unpacklo_epi64(_b, _d);                                           \
    r3 = _mm_unpackhi_epi64(_b, _d);                                           \
  } while (0)

/* H.264 inverse 4x4 1D butterfly applied lane-wise across 4 registers.
   This mirrors the scalar inner block exactly:
       e0 =  t0 + t2;
       e1 =  t0 - t2;
       e2 = (t1 >> 1) - t3;
       e3 =  t1 + (t3 >> 1);
       out0 = e0 + e3;
       out1 = e1 + e2;
       out2 = e1 - e2;
       out3 = e0 - e3;
   _mm_srai_epi32 is arithmetic right shift, matching scalar `>>` on signed int.
 */
#define IDCT4_BUTTERFLY_SSE(t0, t1, t2, t3)                                    \
  do {                                                                         \
    __m128i _e0 = _mm_add_epi32(t0, t2);                                       \
    __m128i _e1 = _mm_sub_epi32(t0, t2);                                       \
    __m128i _e2 = _mm_sub_epi32(_mm_srai_epi32(t1, 1), t3);                    \
    __m128i _e3 = _mm_add_epi32(t1, _mm_srai_epi32(t3, 1));                    \
    t0 = _mm_add_epi32(_e0, _e3);                                              \
    t1 = _mm_add_epi32(_e1, _e2);                                              \
    t2 = _mm_sub_epi32(_e1, _e2);                                              \
    t3 = _mm_sub_epi32(_e0, _e3);                                              \
  } while (0)

void inverse4x4_sse(int **tblock, int **block, int pos_y, int pos_x) {
  /* Load 4 rows. Each row is 4 contiguous int32 values starting at
     tblock[pos_y+k][pos_x]. The row stride only affects the row-pointer
     table, not the within-row layout, so loadu of 16 bytes is correct
     regardless of how tblock was allocated. */
  __m128i r0 = _mm_loadu_si128((const __m128i *)&tblock[pos_y][pos_x]);
  __m128i r1 = _mm_loadu_si128((const __m128i *)&tblock[pos_y + 1][pos_x]);
  __m128i r2 = _mm_loadu_si128((const __m128i *)&tblock[pos_y + 2][pos_x]);
  __m128i r3 = _mm_loadu_si128((const __m128i *)&tblock[pos_y + 3][pos_x]);

  /* === Pass 1: horizontal (matches scalar's first loop) ===
     Scalar applies butterfly to each row's 4 columns. To do that lane-wise
     across registers, transpose first so each register holds one column. */
  TRANSPOSE4_EPI32_SSE(r0, r1, r2, r3);
  IDCT4_BUTTERFLY_SSE(r0, r1, r2, r3);
  /* Result: r[i].lane[j] = H_out_i for row j. */

  /* === Pass 2: vertical (matches scalar's second loop) ===
     Scalar reads H output column-by-column with stride 4. Our transpose
     below puts the H output back into row-major form so the next butterfly
     operates on columns of H, matching scalar exactly. */
  TRANSPOSE4_EPI32_SSE(r0, r1, r2, r3);
  IDCT4_BUTTERFLY_SSE(r0, r1, r2, r3);
  /* Result: r[i].lane[j] = final[i][j], standard row-major. */

  _mm_storeu_si128((__m128i *)&block[pos_y][pos_x], r0);
  _mm_storeu_si128((__m128i *)&block[pos_y + 1][pos_x], r1);
  _mm_storeu_si128((__m128i *)&block[pos_y + 2][pos_x], r2);
  _mm_storeu_si128((__m128i *)&block[pos_y + 3][pos_x], r3);
}

/*!
 ***************************************************************************
 * inverse8x8
 ***************************************************************************
 */

/* In-place 4x4 transpose of 32-bit integer lanes. */
void transpose4_sse2(__m128i out[4], const __m128i in[4]) {
  __m128i a = _mm_unpacklo_epi32(in[0], in[1]);
  __m128i b = _mm_unpackhi_epi32(in[0], in[1]);
  __m128i c = _mm_unpacklo_epi32(in[2], in[3]);
  __m128i d = _mm_unpackhi_epi32(in[2], in[3]);
  out[0] = _mm_unpacklo_epi64(a, c);
  out[1] = _mm_unpackhi_epi64(a, c);
  out[2] = _mm_unpacklo_epi64(b, d);
  out[3] = _mm_unpackhi_epi64(b, d);
}

/* 8x8 transpose, data stored as halves[0][i] = cols 0..3 of row i,
   halves[1][i] = cols 4..7 of row i.  Transposing the 8x8 means
   transposing each 4x4 sub-tile AND swapping the TR <-> BL positions. */
void transpose8x8_sse2(__m128i halves[2][8]) {
  __m128i tile[4][4];
  transpose4_sse2(tile[0], &halves[0][0]); /* TL: rows 0-3, cols 0-3 */
  transpose4_sse2(tile[1], &halves[0][4]); /* BL: rows 4-7, cols 0-3 */
  transpose4_sse2(tile[2], &halves[1][0]); /* TR: rows 0-3, cols 4-7 */
  transpose4_sse2(tile[3], &halves[1][4]); /* BR: rows 4-7, cols 4-7 */

  for (int k = 0; k < 4; k++) {
    halves[0][k] = tile[0][k];     /* TL stays */
    halves[1][k] = tile[1][k];     /* old BL goes to TR position */
    halves[0][k + 4] = tile[2][k]; /* old TR goes to BL position */
    halves[1][k + 4] = tile[3][k]; /* BR stays */
  }
}

/* H.264 8-point inverse 1D butterfly, lane-wise across p[0..7].
   This is the part that genuinely can't be a loop — fixed permutation. */
void sse2_idct8_butterfly(__m128i p[8]) {
  __m128i p0 = p[0], p1 = p[1], p2 = p[2], p3 = p[3];
  __m128i p4 = p[4], p5 = p[5], p6 = p[6], p7 = p[7];

  __m128i a0 = _mm_add_epi32(p0, p4);
  __m128i a1 = _mm_sub_epi32(p0, p4);
  __m128i a2 = _mm_sub_epi32(p6, _mm_srai_epi32(p2, 1));
  __m128i a3 = _mm_add_epi32(p2, _mm_srai_epi32(p6, 1));

  __m128i b0 = _mm_add_epi32(a0, a3);
  __m128i b2 = _mm_sub_epi32(a1, a2);
  __m128i b4 = _mm_add_epi32(a1, a2);
  __m128i b6 = _mm_sub_epi32(a0, a3);

  __m128i c0 = _mm_sub_epi32(_mm_sub_epi32(_mm_sub_epi32(p5, p3), p7),
                             _mm_srai_epi32(p7, 1));
  __m128i c1 = _mm_sub_epi32(_mm_sub_epi32(_mm_add_epi32(p1, p7), p3),
                             _mm_srai_epi32(p3, 1));
  __m128i c2 = _mm_add_epi32(_mm_add_epi32(_mm_sub_epi32(p7, p1), p5),
                             _mm_srai_epi32(p5, 1));
  __m128i c3 = _mm_add_epi32(_mm_add_epi32(_mm_add_epi32(p3, p5), p1),
                             _mm_srai_epi32(p1, 1));

  __m128i b1 = _mm_add_epi32(c0, _mm_srai_epi32(c3, 2));
  __m128i b3 = _mm_add_epi32(c1, _mm_srai_epi32(c2, 2));
  __m128i b5 = _mm_sub_epi32(c2, _mm_srai_epi32(c1, 2));
  __m128i b7 = _mm_sub_epi32(c3, _mm_srai_epi32(c0, 2));

  p[0] = _mm_add_epi32(b0, b7);
  p[1] = _mm_sub_epi32(b2, b5);
  p[2] = _mm_add_epi32(b4, b3);
  p[3] = _mm_add_epi32(b6, b1);
  p[4] = _mm_sub_epi32(b6, b1);
  p[5] = _mm_sub_epi32(b4, b3);
  p[6] = _mm_add_epi32(b2, b5);
  p[7] = _mm_sub_epi32(b0, b7);
}

void inverse8x8_sse2(int **tblock, int **block, int pos_x) {
  /* halves[0][i] = cols 0..3 of row i, halves[1][i] = cols 4..7 of row i. */
  __m128i halves[2][8];

  /* Load */
  for (int i = 0; i < 8; i++) {
    halves[0][i] = _mm_loadu_si128((const __m128i *)&tblock[i][pos_x]);
    halves[1][i] = _mm_loadu_si128((const __m128i *)&tblock[i][pos_x + 4]);
  }

  /* Two passes: each is "transpose, then butterfly both halves".
     Pass 1 transposes -> butterfly does H of original (per row).
     Pass 2 transposes back -> butterfly does V on H-transformed (per column).
   */
  for (int pass = 0; pass < 2; pass++) {
    transpose8x8_sse2(halves);
    for (int h = 0; h < 2; h++)
      sse2_idct8_butterfly(halves[h]);
  }

  /* Store */
  for (int i = 0; i < 8; i++) {
    _mm_storeu_si128((__m128i *)&block[i][pos_x], halves[0][i]);
    _mm_storeu_si128((__m128i *)&block[i][pos_x + 4], halves[1][i]);
  }
}
