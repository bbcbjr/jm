
/*!
 ***************************************************************************
 * \file transform_avx2.c
 *
 * \brief
 *    AVX2 implementations of the H.264 inverse 4x4 and
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

#include <immintrin.h>   /* AVX2: -mavx2 / /arch:AVX2 */

/* 8x8 transpose of 32-bit integer lanes. Each __m256i is one row of 8 ints.
   Pattern: unpack-32, unpack-64, then cross-128 permute. */
#define TRANSPOSE8_EPI32_AVX2(r0, r1, r2, r3, r4, r5, r6, r7) do {     \
    __m256i _t0 = _mm256_unpacklo_epi32(r0, r1);                       \
    __m256i _t1 = _mm256_unpackhi_epi32(r0, r1);                       \
    __m256i _t2 = _mm256_unpacklo_epi32(r2, r3);                       \
    __m256i _t3 = _mm256_unpackhi_epi32(r2, r3);                       \
    __m256i _t4 = _mm256_unpacklo_epi32(r4, r5);                       \
    __m256i _t5 = _mm256_unpackhi_epi32(r4, r5);                       \
    __m256i _t6 = _mm256_unpacklo_epi32(r6, r7);                       \
    __m256i _t7 = _mm256_unpackhi_epi32(r6, r7);                       \
                                                                       \
    __m256i _u0 = _mm256_unpacklo_epi64(_t0, _t2);                     \
    __m256i _u1 = _mm256_unpackhi_epi64(_t0, _t2);                     \
    __m256i _u2 = _mm256_unpacklo_epi64(_t1, _t3);                     \
    __m256i _u3 = _mm256_unpackhi_epi64(_t1, _t3);                     \
    __m256i _u4 = _mm256_unpacklo_epi64(_t4, _t6);                     \
    __m256i _u5 = _mm256_unpackhi_epi64(_t4, _t6);                     \
    __m256i _u6 = _mm256_unpacklo_epi64(_t5, _t7);                     \
    __m256i _u7 = _mm256_unpackhi_epi64(_t5, _t7);                     \
                                                                       \
    /* This step is what your SSE macro was missing entirely. */       \
    r0 = _mm256_permute2x128_si256(_u0, _u4, 0x20);                    \
    r1 = _mm256_permute2x128_si256(_u1, _u5, 0x20);                    \
    r2 = _mm256_permute2x128_si256(_u2, _u6, 0x20);                    \
    r3 = _mm256_permute2x128_si256(_u3, _u7, 0x20);                    \
    r4 = _mm256_permute2x128_si256(_u0, _u4, 0x31);                    \
    r5 = _mm256_permute2x128_si256(_u1, _u5, 0x31);                    \
    r6 = _mm256_permute2x128_si256(_u2, _u6, 0x31);                    \
    r7 = _mm256_permute2x128_si256(_u3, _u7, 0x31);                    \
} while (0)

/* H.264 8x8 inverse 1D butterfly, lane-wise across 8 registers.
   Mirrors the scalar block in lib/lcommon/transform.c:543-571 exactly.
   Note: writes results back to the PARAMETERS p0..p7. */
#define IDCT8_BUTTERFLY_AVX2(p0, p1, p2, p3, p4, p5, p6, p7) do {              \
    __m256i _a0 = _mm256_add_epi32(p0, p4);                                    \
    __m256i _a1 = _mm256_sub_epi32(p0, p4);                                    \
    __m256i _a2 = _mm256_sub_epi32(p6, _mm256_srai_epi32(p2, 1));              \
    __m256i _a3 = _mm256_add_epi32(p2, _mm256_srai_epi32(p6, 1));              \
                                                                               \
    __m256i _b0 = _mm256_add_epi32(_a0, _a3);                                  \
    __m256i _b2 = _mm256_sub_epi32(_a1, _a2);                                  \
    __m256i _b4 = _mm256_add_epi32(_a1, _a2);                                  \
    __m256i _b6 = _mm256_sub_epi32(_a0, _a3);                                  \
                                                                               \
    /* a0 = -p3 + p5 - p7 - (p7 >> 1) == (p5 - p3) - p7 - (p7 >> 1)   */       \
    __m256i _c0 = _mm256_sub_epi32(                                            \
                    _mm256_sub_epi32(_mm256_sub_epi32(p5, p3), p7),            \
                    _mm256_srai_epi32(p7, 1));                                 \
    /* a1 =  p1 + p7 - p3 - (p3 >> 1) */                                       \
    __m256i _c1 = _mm256_sub_epi32(                                            \
                    _mm256_sub_epi32(_mm256_add_epi32(p1, p7), p3),            \
                    _mm256_srai_epi32(p3, 1));                                 \
    /* a2 = -p1 + p7 + p5 + (p5 >> 1) == (p7 - p1) + p5 + (p5 >> 1) */         \
    __m256i _c2 = _mm256_add_epi32(                                            \
                    _mm256_add_epi32(_mm256_sub_epi32(p7, p1), p5),            \
                    _mm256_srai_epi32(p5, 1));                                 \
    /* a3 =  p3 + p5 + p1 + (p1 >> 1) */                                       \
    __m256i _c3 = _mm256_add_epi32(                                            \
                    _mm256_add_epi32(_mm256_add_epi32(p3, p5), p1),            \
                    _mm256_srai_epi32(p1, 1));                                 \
                                                                               \
    __m256i _b1 = _mm256_add_epi32(_c0, _mm256_srai_epi32(_c3, 2));            \
    __m256i _b3 = _mm256_add_epi32(_c1, _mm256_srai_epi32(_c2, 2));            \
    __m256i _b5 = _mm256_sub_epi32(_c2, _mm256_srai_epi32(_c1, 2));            \
    __m256i _b7 = _mm256_sub_epi32(_c3, _mm256_srai_epi32(_c0, 2));            \
                                                                               \
    p0 = _mm256_add_epi32(_b0, _b7);                                           \
    p1 = _mm256_sub_epi32(_b2, _b5);                                           \
    p2 = _mm256_add_epi32(_b4, _b3);                                           \
    p3 = _mm256_add_epi32(_b6, _b1);                                           \
    p4 = _mm256_sub_epi32(_b6, _b1);                                           \
    p5 = _mm256_sub_epi32(_b4, _b3);                                           \
    p6 = _mm256_add_epi32(_b2, _b5);                                           \
    p7 = _mm256_sub_epi32(_b0, _b7);                                           \
} while (0)   /* <- no trailing backslash */

void inverse8x8_avx2(int **tblock, int **block, int pos_x) {
    __m256i r0 = _mm256_loadu_si256((const __m256i*)&tblock[0][pos_x]);
    __m256i r1 = _mm256_loadu_si256((const __m256i*)&tblock[1][pos_x]);
    __m256i r2 = _mm256_loadu_si256((const __m256i*)&tblock[2][pos_x]);
    __m256i r3 = _mm256_loadu_si256((const __m256i*)&tblock[3][pos_x]);
    __m256i r4 = _mm256_loadu_si256((const __m256i*)&tblock[4][pos_x]);
    __m256i r5 = _mm256_loadu_si256((const __m256i*)&tblock[5][pos_x]);
    __m256i r6 = _mm256_loadu_si256((const __m256i*)&tblock[6][pos_x]);
    __m256i r7 = _mm256_loadu_si256((const __m256i*)&tblock[7][pos_x]);

    /* Match scalar order (H then V): transpose, butterfly, transpose, butterfly */
    TRANSPOSE8_EPI32_AVX2(r0, r1, r2, r3, r4, r5, r6, r7);
    IDCT8_BUTTERFLY_AVX2 (r0, r1, r2, r3, r4, r5, r6, r7);
    TRANSPOSE8_EPI32_AVX2(r0, r1, r2, r3, r4, r5, r6, r7);
    IDCT8_BUTTERFLY_AVX2 (r0, r1, r2, r3, r4, r5, r6, r7);

    _mm256_storeu_si256((__m256i*)&block[0][pos_x], r0);
    _mm256_storeu_si256((__m256i*)&block[1][pos_x], r1);
    _mm256_storeu_si256((__m256i*)&block[2][pos_x], r2);
    _mm256_storeu_si256((__m256i*)&block[3][pos_x], r3);
    _mm256_storeu_si256((__m256i*)&block[4][pos_x], r4);
    _mm256_storeu_si256((__m256i*)&block[5][pos_x], r5);
    _mm256_storeu_si256((__m256i*)&block[6][pos_x], r6);
    _mm256_storeu_si256((__m256i*)&block[7][pos_x], r7);
}
