
/*!
 ***************************************************************************
 * \file
 *    mc_kernels_sse.c
 *
 * \brief
 *    SSE2/SSSE3 implementations of H.264 motion compensation kernels.
 *    Each function is the SIMD equivalent of a corresponding scalar
 *    helper in mc_prediction.c. Selected at runtime by jm_simd_init()
 *    based on detected CPU features.
 *
 *    Requires SSSE3 (uses _mm_alignr_epi8). Falls back to scalar on
 *    pre-SSSE3 CPUs (no longer in production circa ~2008).
 *
 *    H.264 luma half-pel filter coefficients: [1, -5, 20, 20, -5, 1] / 32
 *      output[i] = clip(0, max,
 *                       ((src[i+0] + src[i+5])
 *                       - 5  * (src[i+1] + src[i+4])
 *                       + 20 * (src[i+2] + src[i+3])
 *                       + 16) >> 5)
 *
 *    The source region is always at least width+5 bytes wide (callers
 *    pass &cur_imgY[j][x_pos - 2], which is inside the reference
 *    picture's pad region of >= 32 bytes on each side). So loading 16
 *    bytes is always safe even when only 4 or 8 source bytes are
 *    semantically needed.
 ***************************************************************************
 */

#include "global.h"
#include "mc_prediction.h"

/* ===================================================================== */
/*   IMGTYPE compatibility gate                                           */
/* ===================================================================== */
/*  The SIMD kernels below assume sizeof(imgpel) == 1 (IMGTYPE == 0):     */
/*    - _mm_unpacklo_epi8 / _mm_packus_epi16 are byte-vs-short routines   */
/*    - _mm_alignr_epi8 offsets are in bytes (= 1 pixel for 8-bit only)   */
/*    - int16 intermediates fit only for 8-bit input (max product ~10200) */
/*  For IMGTYPE == 1 (10/12-bit imgpel = uint16) we don't compile the     */
/*  kernels at all, and jm_simd_init keeps the scalar pointers.           */
/*  A proper 16-bit SIMD path needs int32 intermediates and _mm_*_epi16   */
/*  loads/stores -- left as future work.                                  */
/* ===================================================================== */
#if IMGTYPE == 0

#include <assert.h>      /* runtime sanity asserts (NDEBUG-stripped in release) */
#include <string.h>      /* memcpy for 4-byte store */
#include <emmintrin.h>   /* SSE2 */
#include <tmmintrin.h>   /* SSSE3: _mm_alignr_epi8 */

/* ===================================================================== */
/*   Static assertion: sizeof(imgpel) MUST be 1 for these kernels.        */
/*   Catches the case where IMGTYPE == 0 in defines.h but someone tweaks  */
/*   typedefs.h or otherwise breaks the byte-pixel assumption. Portable   */
/*   to pre-C11 compilers via the array-size trick: negative size is an   */
/*   error.                                                               */
/* ===================================================================== */
#define JM_STATIC_ASSERT(cond, tag) \
  typedef char jm_static_assert_##tag[(cond) ? 1 : -1]

JM_STATIC_ASSERT(sizeof(imgpel) == 1,
                 mc_kernels_sse_requires_byte_imgpel_when_IMGTYPE_is_0);

/*!
 ************************************************************************
 * \brief Irreducible 6-tap filter kernel. Returns the RAW int16 result
 *        of the H.264 filter coefficients [1, -5, 20, 20, -5, 1]:
 *          out = (p0+p5) - 5*(p1+p4) + 20*(p2+p3)
 *
 *        No round, no shift, no clip -- those are output specifications
 *        that belong to the caller (see jm_hpel_finish_epi16 for the
 *        half-pel finisher; 2D positions 22/21/23/12/32 store this raw
 *        result into int32 tmp_res between passes).
 *
 *        Range for byte-pel inputs:
 *          max sum05 = 510, max 5*sum14 = 2550, max 20*sum23 = 10200,
 *          result range roughly [-2550, +10710] -- well inside int16.
 ************************************************************************
 */
static __inline __m128i jm_filter6tap_epi16(__m128i p0, __m128i p1, __m128i p2,
                                            __m128i p3, __m128i p4, __m128i p5,
                                            __m128i k5, __m128i k20)
{
  __m128i sum05  = _mm_add_epi16(p0, p5);
  __m128i sum14  = _mm_add_epi16(p1, p4);
  __m128i sum23  = _mm_add_epi16(p2, p3);
  __m128i term14 = _mm_mullo_epi16(sum14, k5);
  __m128i term23 = _mm_mullo_epi16(sum23, k20);
  __m128i res    = _mm_sub_epi16(sum05, term14);
  return _mm_add_epi16(res, term23);
}

/*!
 ************************************************************************
 * \brief Half-pel output finisher: (raw + 16) >> 5, clipped to [0, vmax].
 *        Apply on the result of jm_filter6tap_epi16 (or jm_hfilter8_epi16
 *        / jm_vfilter8_epi16) when producing a spec half-pel sample --
 *        positions 20 / 02 directly, and the half-pel half of the qpel
 *        positions 10 / 30 / 01 / 03 / 21 / 23 / 12 / 32 before the
 *        bilinear average step.
 ************************************************************************
 */
static __inline __m128i jm_hpel_finish_epi16(__m128i raw,
                                             __m128i k16, __m128i vmax,
                                             __m128i zero)
{
  __m128i r = _mm_add_epi16(raw, k16);
  r = _mm_srai_epi16(r, 5);
  r = _mm_max_epi16(r, zero);
  return _mm_min_epi16(r, vmax);
}

/* ===================================================================== */
/*     get_luma_20: horizontal half-pel (dx=2, dy=0)                      */
/* ===================================================================== */

/*!
 ************************************************************************
 * \brief Compute 8 horizontal RAW 6-tap filter outputs into one 128-bit
 *        register. src points to scalar p0 (= cur_imgY[j][x_pos-2]).
 *        We load 16 source bytes; only src[0..12] are semantically used.
 *
 *        Returns raw int16 (no round/shift/clip). Used directly by 2D
 *        position pass 1 (22/21/23). For a spec half-pel output, wrap
 *        with jm_hpel_finish_epi16 (or use jm_hpel_h8_epi16).
 ************************************************************************
 */
static __inline __m128i jm_hfilter8_epi16(const imgpel *src,
                                          __m128i k5, __m128i k20, __m128i zero)
{
  __m128i s = _mm_loadu_si128((const __m128i *)src);
  __m128i s_lo = _mm_unpacklo_epi8(s, zero);   /* src[0..7]  as 8 shorts */
  __m128i s_hi = _mm_unpackhi_epi8(s, zero);   /* src[8..15] as 8 shorts */

  /* The six 8-wide windows, slid 0..5 elements across src. */
  __m128i p0 = s_lo;                                       /* src[0..7]  */
  __m128i p1 = _mm_alignr_epi8(s_hi, s_lo,  2);            /* src[1..8]  */
  __m128i p2 = _mm_alignr_epi8(s_hi, s_lo,  4);            /* src[2..9]  */
  __m128i p3 = _mm_alignr_epi8(s_hi, s_lo,  6);            /* src[3..10] */
  __m128i p4 = _mm_alignr_epi8(s_hi, s_lo,  8);            /* src[4..11] */
  __m128i p5 = _mm_alignr_epi8(s_hi, s_lo, 10);            /* src[5..12] */

  return jm_filter6tap_epi16(p0, p1, p2, p3, p4, p5, k5, k20);
}

/*!
 ************************************************************************
 * \brief Composed half-pel H filter: raw H 6-tap + half-pel finisher.
 *        Returns 8 clipped int16 lanes ready for _mm_packus_epi16 to
 *        bytes. Convenience wrapper for the single-pass half-pel
 *        kernels (20, 10, 30) so they don't churn at every call site.
 ************************************************************************
 */
static __inline __m128i jm_hpel_h8_epi16(const imgpel *src,
                                         __m128i k5, __m128i k20, __m128i k16,
                                         __m128i vmax, __m128i zero)
{
  return jm_hpel_finish_epi16(jm_hfilter8_epi16(src, k5, k20, zero),
                              k16, vmax, zero);
}

/* ===================================================================== */
/*     get_luma_02: vertical half-pel (dx=0, dy=2)                        */
/* ===================================================================== */

/*!
 ************************************************************************
 * \brief Compute 8 vertical RAW 6-tap filter outputs from 6 source row
 *        pointers, each pointing at the leftmost source column of its
 *        row. We load 8 bytes per row (only block_size_x bytes are
 *        semantically used; over-reads are safe due to lateral padding).
 *
 *        Returns raw int16 (no round/shift/clip). For a spec half-pel
 *        output, wrap with jm_hpel_finish_epi16 (or use jm_hpel_v8_epi16).
 ************************************************************************
 */
static __inline __m128i jm_vfilter8_epi16(const imgpel *r0, const imgpel *r1,
                                          const imgpel *r2, const imgpel *r3,
                                          const imgpel *r4, const imgpel *r5,
                                          __m128i k5, __m128i k20, __m128i zero)
{
  __m128i p0 = _mm_unpacklo_epi8(_mm_loadl_epi64((const __m128i *)r0), zero);
  __m128i p1 = _mm_unpacklo_epi8(_mm_loadl_epi64((const __m128i *)r1), zero);
  __m128i p2 = _mm_unpacklo_epi8(_mm_loadl_epi64((const __m128i *)r2), zero);
  __m128i p3 = _mm_unpacklo_epi8(_mm_loadl_epi64((const __m128i *)r3), zero);
  __m128i p4 = _mm_unpacklo_epi8(_mm_loadl_epi64((const __m128i *)r4), zero);
  __m128i p5 = _mm_unpacklo_epi8(_mm_loadl_epi64((const __m128i *)r5), zero);

  return jm_filter6tap_epi16(p0, p1, p2, p3, p4, p5, k5, k20);
}

/*!
 ************************************************************************
 * \brief Composed half-pel V filter: raw V 6-tap + half-pel finisher.
 *        Convenience wrapper for the single-pass half-pel kernels
 *        (02, 01, 03).
 ************************************************************************
 */
static __inline __m128i jm_hpel_v8_epi16(const imgpel *r0, const imgpel *r1,
                                         const imgpel *r2, const imgpel *r3,
                                         const imgpel *r4, const imgpel *r5,
                                         __m128i k5, __m128i k20, __m128i k16,
                                         __m128i vmax, __m128i zero)
{
  return jm_hpel_finish_epi16(jm_vfilter8_epi16(r0, r1, r2, r3, r4, r5, k5, k20, zero),
                              k16, vmax, zero);
}

void get_luma_20_sse(imgpel **block, imgpel **cur_imgY, int block_size_y,
                     int block_size_x, int x_pos, int max_imgpel_value)
{
  const __m128i k5   = _mm_set1_epi16(5);
  const __m128i k20  = _mm_set1_epi16(20);
  const __m128i k16  = _mm_set1_epi16(16);
  const __m128i zero = _mm_setzero_si128();
  const __m128i vmax = _mm_set1_epi16((short)max_imgpel_value);
  int j, i;

  /* Defensive: kernel assumes byte pixels. Stripped in release builds.
   * The companion static_assert at the top of this TU is the real defense;
   * this catches cross-TU mismatches in debug builds. */
  assert(sizeof(imgpel) == 1);

  if (block_size_x == 4)
  {
    /* width=4: compute 8 outputs (over-compute is harmless), store low 4 bytes. */
    for (j = 0; j < block_size_y; j++)
    {
      const imgpel *src = &cur_imgY[j][x_pos - 2];
      imgpel       *dst = block[j];
      __m128i res = jm_hpel_h8_epi16(src, k5, k20, k16, vmax, zero);
      __m128i pak = _mm_packus_epi16(res, res);
      int    tmp  = _mm_cvtsi128_si32(pak);
      memcpy(dst, &tmp, 4);   /* alignment-safe 4-byte store */
    }
  }
  else
  {
    /* width=8 or width=16: process 8 outputs per inner iteration. */
    for (j = 0; j < block_size_y; j++)
    {
      const imgpel *src_row = &cur_imgY[j][x_pos - 2];
      imgpel       *dst     = block[j];
      for (i = 0; i < block_size_x; i += 8)
      {
        __m128i res = jm_hpel_h8_epi16(src_row + i, k5, k20, k16, vmax, zero);
        __m128i pak = _mm_packus_epi16(res, res);
        _mm_storel_epi64((__m128i *)(dst + i), pak);
      }
    }
  }
}

/* shift_x parameter on the scalar 02/01/03 kernels is the source row
 * stride in bytes (`p1 = p0 + shift_x` advances to the next row's same
 * column). In our SIMD form we use cur_imgY[j+k] row pointers directly
 * since the row-pointer array contains exactly cur_imgY[k] = cur_imgY[0]
 * + k*shift_x for a padded reference. The scalar shift_x param is
 * therefore unused in our SIMD path -- the row-pointer indexing already
 * encodes the same row stepping. */

void get_luma_02_sse(imgpel **block, imgpel **cur_imgY, int block_size_y,
                     int block_size_x, int x_pos, int shift_x, int max_imgpel_value)
{
  const __m128i k5   = _mm_set1_epi16(5);
  const __m128i k20  = _mm_set1_epi16(20);
  const __m128i k16  = _mm_set1_epi16(16);
  const __m128i zero = _mm_setzero_si128();
  const __m128i vmax = _mm_set1_epi16((short)max_imgpel_value);
  int j, i;
  (void)shift_x;   /* see comment above */

  /* Defensive: kernel assumes byte pixels. Stripped in release builds. */
  assert(sizeof(imgpel) == 1);

  if (block_size_x == 4)
  {
    for (j = 0; j < block_size_y; j++)
    {
      const imgpel *r0 = &cur_imgY[j - 2][x_pos];
      const imgpel *r1 = &cur_imgY[j - 1][x_pos];
      const imgpel *r2 = &cur_imgY[j    ][x_pos];
      const imgpel *r3 = &cur_imgY[j + 1][x_pos];
      const imgpel *r4 = &cur_imgY[j + 2][x_pos];
      const imgpel *r5 = &cur_imgY[j + 3][x_pos];
      imgpel       *dst = block[j];
      __m128i res = jm_hpel_v8_epi16(r0, r1, r2, r3, r4, r5, k5, k20, k16, vmax, zero);
      __m128i pak = _mm_packus_epi16(res, res);
      int    tmp  = _mm_cvtsi128_si32(pak);
      memcpy(dst, &tmp, 4);
    }
  }
  else
  {
    for (j = 0; j < block_size_y; j++)
    {
      const imgpel *r0 = &cur_imgY[j - 2][x_pos];
      const imgpel *r1 = &cur_imgY[j - 1][x_pos];
      const imgpel *r2 = &cur_imgY[j    ][x_pos];
      const imgpel *r3 = &cur_imgY[j + 1][x_pos];
      const imgpel *r4 = &cur_imgY[j + 2][x_pos];
      const imgpel *r5 = &cur_imgY[j + 3][x_pos];
      imgpel       *dst = block[j];
      for (i = 0; i < block_size_x; i += 8)
      {
        __m128i res = jm_hpel_v8_epi16(r0+i, r1+i, r2+i, r3+i, r4+i, r5+i,
                                       k5, k20, k16, vmax, zero);
        __m128i pak = _mm_packus_epi16(res, res);
        _mm_storel_epi64((__m128i *)(dst + i), pak);
      }
    }
  }
}

/* ===================================================================== */
/*   get_luma_10 / get_luma_30: horizontal quarter-pel                    */
/* ===================================================================== */
/*  Both kernels compute the horizontal half-pel (same filter as          */
/*  get_luma_20) and then bilinear-average it with an integer pel:        */
/*     out = (half_pel + int_pel + 1) >> 1                                */
/*  get_luma_10: int pel at column x_pos + i  (left  side of half-pel)    */
/*  get_luma_30: int pel at column x_pos+1+i  (right side of half-pel)    */
/*  _mm_avg_epu8 is the exact byte-wise rounded average we need.          */
/* ===================================================================== */

/*!
 ************************************************************************
 * \brief Shared inner loop: compute hfilter for 8 outputs, pack to
 *        bytes, bilinear-average with 8 integer pels from intpel.
 ************************************************************************
 */
static __inline void jm_hqpel8_store(imgpel *dst, const imgpel *src_for_filter,
                                     const imgpel *intpel,
                                     __m128i k5, __m128i k20, __m128i k16,
                                     __m128i vmax, __m128i zero)
{
  __m128i half_short = jm_hpel_h8_epi16(src_for_filter, k5, k20, k16, vmax, zero);
  __m128i half_byte  = _mm_packus_epi16(half_short, half_short);     /* 8 bytes in low 64 */
  __m128i int_byte   = _mm_loadl_epi64((const __m128i *)intpel);     /* 8 int pels */
  __m128i avg        = _mm_avg_epu8(half_byte, int_byte);            /* (a+b+1) >> 1 */
  _mm_storel_epi64((__m128i *)dst, avg);
}

/*!
 ************************************************************************
 * \brief Width-4 variant of jm_hqpel8_store: compute 8, store low 4.
 ************************************************************************
 */
static __inline void jm_hqpel4_store(imgpel *dst, const imgpel *src_for_filter,
                                     const imgpel *intpel,
                                     __m128i k5, __m128i k20, __m128i k16,
                                     __m128i vmax, __m128i zero)
{
  __m128i half_short = jm_hpel_h8_epi16(src_for_filter, k5, k20, k16, vmax, zero);
  __m128i half_byte  = _mm_packus_epi16(half_short, half_short);
  __m128i int_byte   = _mm_loadl_epi64((const __m128i *)intpel);     /* 8 bytes; low 4 valid */
  __m128i avg        = _mm_avg_epu8(half_byte, int_byte);
  int tmp = _mm_cvtsi128_si32(avg);
  memcpy(dst, &tmp, 4);
}

/* Common body for get_luma_10/30. int_pel_offset is the column offset
 * from x_pos to the integer pel position (0 for _10, 1 for _30). */
static __inline void jm_qpel_horiz_common(imgpel **block, imgpel **cur_imgY,
                                          int block_size_y, int block_size_x,
                                          int x_pos, int max_imgpel_value,
                                          int int_pel_offset)
{
  const __m128i k5   = _mm_set1_epi16(5);
  const __m128i k20  = _mm_set1_epi16(20);
  const __m128i k16  = _mm_set1_epi16(16);
  const __m128i zero = _mm_setzero_si128();
  const __m128i vmax = _mm_set1_epi16((short)max_imgpel_value);
  int j, i;

  assert(sizeof(imgpel) == 1);

  if (block_size_x == 4)
  {
    for (j = 0; j < block_size_y; j++)
    {
      jm_hqpel4_store(block[j],
                      &cur_imgY[j][x_pos - 2],
                      &cur_imgY[j][x_pos + int_pel_offset],
                      k5, k20, k16, vmax, zero);
    }
  }
  else
  {
    for (j = 0; j < block_size_y; j++)
    {
      const imgpel *src_row = &cur_imgY[j][x_pos - 2];
      const imgpel *int_row = &cur_imgY[j][x_pos + int_pel_offset];
      imgpel       *dst     = block[j];
      for (i = 0; i < block_size_x; i += 8)
      {
        jm_hqpel8_store(dst + i, src_row + i, int_row + i,
                        k5, k20, k16, vmax, zero);
      }
    }
  }
}

void get_luma_10_sse(imgpel **block, imgpel **cur_imgY, int block_size_y,
                     int block_size_x, int x_pos, int max_imgpel_value)
{
  /* (dx=1, dy=0): integer pel is at x_pos + i (left of half-pel midpoint) */
  jm_qpel_horiz_common(block, cur_imgY, block_size_y, block_size_x,
                       x_pos, max_imgpel_value, 0);
}

void get_luma_30_sse(imgpel **block, imgpel **cur_imgY, int block_size_y,
                     int block_size_x, int x_pos, int max_imgpel_value)
{
  /* (dx=3, dy=0): integer pel is at x_pos + 1 + i (right of half-pel midpoint) */
  jm_qpel_horiz_common(block, cur_imgY, block_size_y, block_size_x,
                       x_pos, max_imgpel_value, 1);
}

/* ===================================================================== */
/*   get_luma_01 / get_luma_03: vertical quarter-pel                      */
/* ===================================================================== */
/*  Vertical analog of 10/30:                                             */
/*     out = (vert_half_pel + int_pel + 1) >> 1                           */
/*  get_luma_01: int pel from cur_imgY[j  ] (row above half-pel midpoint) */
/*  get_luma_03: int pel from cur_imgY[j+1] (row below half-pel midpoint) */
/*  scalar's `shift_x` is the row stride; in SIMD we use cur_imgY[k]      */
/*  row-pointer indexing instead and ignore the param.                    */
/* ===================================================================== */

/*!
 ************************************************************************
 * \brief Width-8+ inner work: vfilter for 8 outputs, pack to bytes,
 *        average with 8 integer pels from intpel.
 ************************************************************************
 */
static __inline void jm_vqpel8_store(imgpel *dst, const imgpel *r0, const imgpel *r1,
                                     const imgpel *r2, const imgpel *r3,
                                     const imgpel *r4, const imgpel *r5,
                                     const imgpel *intpel,
                                     __m128i k5, __m128i k20, __m128i k16,
                                     __m128i vmax, __m128i zero)
{
  __m128i half_short = jm_hpel_v8_epi16(r0, r1, r2, r3, r4, r5,
                                        k5, k20, k16, vmax, zero);
  __m128i half_byte  = _mm_packus_epi16(half_short, half_short);
  __m128i int_byte   = _mm_loadl_epi64((const __m128i *)intpel);
  __m128i avg        = _mm_avg_epu8(half_byte, int_byte);
  _mm_storel_epi64((__m128i *)dst, avg);
}

/* Common body for get_luma_01/03. int_row_offset is the row offset from
 * j (0 for _01: averaging with the row at j, 1 for _03: row at j+1). */
static __inline void jm_qpel_vert_common(imgpel **block, imgpel **cur_imgY,
                                         int block_size_y, int block_size_x,
                                         int x_pos, int max_imgpel_value,
                                         int int_row_offset)
{
  const __m128i k5   = _mm_set1_epi16(5);
  const __m128i k20  = _mm_set1_epi16(20);
  const __m128i k16  = _mm_set1_epi16(16);
  const __m128i zero = _mm_setzero_si128();
  const __m128i vmax = _mm_set1_epi16((short)max_imgpel_value);
  int j, i;

  assert(sizeof(imgpel) == 1);

  if (block_size_x == 4)
  {
    for (j = 0; j < block_size_y; j++)
    {
      const imgpel *r0 = &cur_imgY[j - 2][x_pos];
      const imgpel *r1 = &cur_imgY[j - 1][x_pos];
      const imgpel *r2 = &cur_imgY[j    ][x_pos];
      const imgpel *r3 = &cur_imgY[j + 1][x_pos];
      const imgpel *r4 = &cur_imgY[j + 2][x_pos];
      const imgpel *r5 = &cur_imgY[j + 3][x_pos];
      const imgpel *intp = &cur_imgY[j + int_row_offset][x_pos];
      imgpel       *dst = block[j];
      /* Width-4 inline path: compute 8 outputs, store low 4 bytes */
      __m128i half_short = jm_hpel_v8_epi16(r0, r1, r2, r3, r4, r5,
                                            k5, k20, k16, vmax, zero);
      __m128i half_byte  = _mm_packus_epi16(half_short, half_short);
      __m128i int_byte   = _mm_loadl_epi64((const __m128i *)intp);
      __m128i avg        = _mm_avg_epu8(half_byte, int_byte);
      int tmp = _mm_cvtsi128_si32(avg);
      memcpy(dst, &tmp, 4);
    }
  }
  else
  {
    for (j = 0; j < block_size_y; j++)
    {
      const imgpel *r0 = &cur_imgY[j - 2][x_pos];
      const imgpel *r1 = &cur_imgY[j - 1][x_pos];
      const imgpel *r2 = &cur_imgY[j    ][x_pos];
      const imgpel *r3 = &cur_imgY[j + 1][x_pos];
      const imgpel *r4 = &cur_imgY[j + 2][x_pos];
      const imgpel *r5 = &cur_imgY[j + 3][x_pos];
      const imgpel *intp = &cur_imgY[j + int_row_offset][x_pos];
      imgpel       *dst = block[j];
      for (i = 0; i < block_size_x; i += 8)
      {
        jm_vqpel8_store(dst + i, r0+i, r1+i, r2+i, r3+i, r4+i, r5+i, intp+i,
                        k5, k20, k16, vmax, zero);
      }
    }
  }
}

void get_luma_01_sse(imgpel **block, imgpel **cur_imgY, int block_size_y,
                     int block_size_x, int x_pos, int shift_x, int max_imgpel_value)
{
  /* (dx=0, dy=1): integer pel is from row j (above the half-pel midpoint) */
  (void)shift_x;
  jm_qpel_vert_common(block, cur_imgY, block_size_y, block_size_x,
                      x_pos, max_imgpel_value, 0);
}

void get_luma_03_sse(imgpel **block, imgpel **cur_imgY, int block_size_y,
                     int block_size_x, int x_pos, int shift_x, int max_imgpel_value)
{
  /* (dx=0, dy=3): integer pel is from row j+1 (below the half-pel midpoint) */
  (void)shift_x;
  jm_qpel_vert_common(block, cur_imgY, block_size_y, block_size_x,
                      x_pos, max_imgpel_value, 1);
}

/* ===================================================================== */
/*   get_luma_22: diagonal half-pel (dx=2, dy=2)                          */
/* ===================================================================== */
/*  True 2D 6-tap filter, separable as horizontal-then-vertical with an   */
/*  int32 intermediate buffer (tmp_res, sized [16+5][16] in the caller).  */
/*                                                                        */
/*  Pass 1 (horizontal raw 6-tap, NO round/shift/clip):                   */
/*    for j in 0..block_size_y+4:                                         */
/*      tmp_res[j][i] = (src[i+0]+src[i+5]) - 5*(src[i+1]+src[i+4])       */
/*                     + 20*(src[i+2]+src[i+3])                           */
/*    Result range is roughly [-2550, +10710] -- still fits in int16, so  */
/*    we compute in int16 vectors then sign-extend to int32 for storage.  */
/*                                                                        */
/*  Pass 2 (vertical 6-tap on int32, round+shift+clip+pack):              */
/*    for j in 0..block_size_y-1:                                         */
/*      r = (t[j]+t[j+5]) - 5*(t[j+1]+t[j+4]) + 20*(t[j+2]+t[j+3])        */
/*      block[j][i] = clip(0, max_imgpel_value, (r + 512) >> 10)          */
/*    int32 arithmetic is required: per-lane magnitudes can reach         */
/*    ~10710 * 42 ~= 450k, comfortably inside int32.                      */
/* ===================================================================== */

/* (Raw horizontal 6-tap helper now shared with the single-pass kernels --
 * see jm_hfilter8_epi16 at the top of the file. Both this kernel's pass 1
 * and 20/02/10/30/01/03 invoke it; the half-pel kernels additionally apply
 * jm_hpel_finish_epi16, while we feed the raw int16 into tmp_res.) */

/* Multiply each int32 lane by 5 via shift+add: x*5 = (x<<2) + x.
 * Avoids SSE4.1's _mm_mullo_epi32 so we stay SSSE3-only. */
static __inline __m128i jm_mul5_epi32(__m128i x)
{
  return _mm_add_epi32(_mm_slli_epi32(x, 2), x);
}

/* Multiply each int32 lane by 20: x*20 = (x<<4) + (x<<2). */
static __inline __m128i jm_mul20_epi32(__m128i x)
{
  return _mm_add_epi32(_mm_slli_epi32(x, 4), _mm_slli_epi32(x, 2));
}

/*!
 ************************************************************************
 * \brief Irreducible 6-tap filter kernel for int32 lanes -- the exact
 *        mirror of jm_filter6tap_epi16, just widened. Returns the RAW
 *        int32 result (no round, no shift, no clip):
 *          out = (v0+v5) - 5*(v1+v4) + 20*(v2+v3)
 *
 *        Used by the pass-2 vertical filter on tmp_res int32 values
 *        for the 2D positions 22, 21, 23, 12, 32. Their finishers
 *        differ -- 22 uses jm_dhpel_finish_epi32 (+512, >>10); the
 *        others combine that with bilinear averaging at the call site.
 *
 *        Shift+add (jm_mul5/20_epi32) keeps us SSSE3-only by avoiding
 *        SSE4.1's _mm_mullo_epi32.
 ************************************************************************
 */
static __inline __m128i jm_filter6tap_epi32(__m128i v0, __m128i v1, __m128i v2,
                                            __m128i v3, __m128i v4, __m128i v5)
{
  __m128i sum05 = _mm_add_epi32(v0, v5);
  __m128i sum14 = _mm_add_epi32(v1, v4);
  __m128i sum23 = _mm_add_epi32(v2, v3);
  __m128i t14   = jm_mul5_epi32(sum14);
  __m128i t23   = jm_mul20_epi32(sum23);
  __m128i res   = _mm_sub_epi32(sum05, t14);
  return _mm_add_epi32(res, t23);
}

/*!
 ************************************************************************
 * \brief Diagonal-half-pel int32 finisher: (raw + 512) >> 10. Returns
 *        int32 lanes. The int32->int16 saturated pack, the clip vs
 *        vmax, and the int16->byte pack stay at call sites because the
 *        packing pattern depends on width (4 / 8 / 16).
 ************************************************************************
 */
static __inline __m128i jm_dhpel_finish_epi32(__m128i raw, __m128i k512)
{
  return _mm_srai_epi32(_mm_add_epi32(raw, k512), 10);
}

/* Pass-2 vertical 6-tap over 4 int32 lanes, +512 round, >>10 arithmetic
 * shift. Returns int32x4 with each lane in roughly [-2, +max+2] pre-clip.
 * Composition: load + jm_filter6tap_epi32 + jm_dhpel_finish_epi32. */
static __inline __m128i jm_vfilter4_pass2_int32(const int *x0, const int *x1,
                                                const int *x2, const int *x3,
                                                const int *x4, const int *x5,
                                                __m128i k512)
{
  __m128i v0  = _mm_loadu_si128((const __m128i *)x0);
  __m128i v1  = _mm_loadu_si128((const __m128i *)x1);
  __m128i v2  = _mm_loadu_si128((const __m128i *)x2);
  __m128i v3  = _mm_loadu_si128((const __m128i *)x3);
  __m128i v4  = _mm_loadu_si128((const __m128i *)x4);
  __m128i v5  = _mm_loadu_si128((const __m128i *)x5);
  __m128i raw = jm_filter6tap_epi32(v0, v1, v2, v3, v4, v5);
  return jm_dhpel_finish_epi32(raw, k512);
}

/*!
 ************************************************************************
 * \brief Pass-1 helper shared by the 2D positions 22, 21, 23: horizontal
 *        raw 6-tap filter over (block_size_y + 5) source rows, with the
 *        int16 result sign-extended to int32 and stored into tmp_res.
 *
 *        block_size_x must be 4, 8, or 16 -- the three sizes H.264 MC
 *        produces. The defensive assert catches anything else (which
 *        would indicate a caller bug, not bad input data).
 ************************************************************************
 */
static __inline void jm_x2_pass1_hraw_to_tmpres(imgpel **cur_imgY, int **tmp_res,
                                                int block_size_y, int block_size_x,
                                                int x_pos,
                                                __m128i k5, __m128i k20,
                                                __m128i zero)
{
  int jj = -2;
  int j, i;

  assert((block_size_x == 4 || block_size_x == 8 || block_size_x == 16) &&
         "MC pass-1 expects block_size_x in {4, 8, 16}");

  if (block_size_x == 4)
  {
    for (j = 0; j < block_size_y + 5; j++)
    {
      const imgpel *src  = &cur_imgY[jj++][x_pos - 2];
      __m128i s16  = jm_hfilter8_epi16(src, k5, k20, zero);
      __m128i sign = _mm_srai_epi16(s16, 15);          /* sign-replication */
      __m128i lo32 = _mm_unpacklo_epi16(s16, sign);    /* 4 int32s         */
      _mm_storeu_si128((__m128i *)tmp_res[j], lo32);
    }
  }
  else if (block_size_x == 8)
  {
    for (j = 0; j < block_size_y + 5; j++)
    {
      const imgpel *src  = &cur_imgY[jj++][x_pos - 2];
      __m128i s16  = jm_hfilter8_epi16(src, k5, k20, zero);
      __m128i sign = _mm_srai_epi16(s16, 15);
      __m128i lo32 = _mm_unpacklo_epi16(s16, sign);
      __m128i hi32 = _mm_unpackhi_epi16(s16, sign);
      _mm_storeu_si128((__m128i *)(tmp_res[j] + 0), lo32);
      _mm_storeu_si128((__m128i *)(tmp_res[j] + 4), hi32);
    }
  }
  else  /* block_size_x == 16 (guaranteed by assert above) */
  {
    for (j = 0; j < block_size_y + 5; j++)
    {
      const imgpel *src = &cur_imgY[jj++][x_pos - 2];
      int          *dst = tmp_res[j];
      for (i = 0; i < block_size_x; i += 8)
      {
        __m128i s16  = jm_hfilter8_epi16(src + i, k5, k20, zero);
        __m128i sign = _mm_srai_epi16(s16, 15);
        __m128i lo32 = _mm_unpacklo_epi16(s16, sign);
        __m128i hi32 = _mm_unpackhi_epi16(s16, sign);
        _mm_storeu_si128((__m128i *)(dst + i + 0), lo32);
        _mm_storeu_si128((__m128i *)(dst + i + 4), hi32);
      }
    }
  }
}

void get_luma_22_sse(imgpel **block, imgpel **cur_imgY, int **tmp_res,
                     int block_size_y, int block_size_x, int x_pos,
                     int max_imgpel_value)
{
  const __m128i k5   = _mm_set1_epi16(5);
  const __m128i k20  = _mm_set1_epi16(20);
  const __m128i k512 = _mm_set1_epi32(512);
  const __m128i zero = _mm_setzero_si128();
  const __m128i vmax = _mm_set1_epi16((short)max_imgpel_value);
  int j;

  /* Defensive: kernel assumes byte pixels. Stripped in release. */
  assert(sizeof(imgpel) == 1);

  /* ---------- Pass 1: shared with get_luma_21 / get_luma_23 ---------- */
  jm_x2_pass1_hraw_to_tmpres(cur_imgY, tmp_res, block_size_y, block_size_x,
                             x_pos, k5, k20, zero);

  /* ---------- Pass 2: vertical int32 6-tap -> round/shift/clip/pack ---------- */
  /* _mm_packus_epi16 saturates int16 to [0, 255] already; we still apply
   * the explicit max/min vs vmax because max_imgpel_value can be < 255
   * (custom profiles / depth maps) and we need to honor it. */
  if (block_size_x == 4)
  {
    for (j = 0; j < block_size_y; j++)
    {
      const int *x0 = tmp_res[j];     const int *x1 = tmp_res[j + 1];
      const int *x2 = tmp_res[j + 2]; const int *x3 = tmp_res[j + 3];
      const int *x4 = tmp_res[j + 4]; const int *x5 = tmp_res[j + 5];
      imgpel    *dst = block[j];
      __m128i r32 = jm_vfilter4_pass2_int32(x0, x1, x2, x3, x4, x5, k512);
      __m128i r16 = _mm_packs_epi32(r32, r32);                /* low 4 valid */
      __m128i r16c = _mm_min_epi16(_mm_max_epi16(r16, zero), vmax);
      __m128i r8  = _mm_packus_epi16(r16c, r16c);
      int tmp = _mm_cvtsi128_si32(r8);
      memcpy(dst, &tmp, 4);
    }
  }
  else if (block_size_x == 8)
  {
    for (j = 0; j < block_size_y; j++)
    {
      const int *x0 = tmp_res[j];     const int *x1 = tmp_res[j + 1];
      const int *x2 = tmp_res[j + 2]; const int *x3 = tmp_res[j + 3];
      const int *x4 = tmp_res[j + 4]; const int *x5 = tmp_res[j + 5];
      imgpel    *dst = block[j];
      __m128i r32a = jm_vfilter4_pass2_int32(x0+0, x1+0, x2+0, x3+0, x4+0, x5+0, k512);
      __m128i r32b = jm_vfilter4_pass2_int32(x0+4, x1+4, x2+4, x3+4, x4+4, x5+4, k512);
      __m128i r16  = _mm_packs_epi32(r32a, r32b);
      __m128i r16c = _mm_min_epi16(_mm_max_epi16(r16, zero), vmax);
      __m128i r8   = _mm_packus_epi16(r16c, r16c);
      _mm_storel_epi64((__m128i *)dst, r8);
    }
  }
  else  /* block_size_x == 16 */
  {
    for (j = 0; j < block_size_y; j++)
    {
      const int *x0 = tmp_res[j];     const int *x1 = tmp_res[j + 1];
      const int *x2 = tmp_res[j + 2]; const int *x3 = tmp_res[j + 3];
      const int *x4 = tmp_res[j + 4]; const int *x5 = tmp_res[j + 5];
      imgpel    *dst = block[j];
      __m128i r32_0 = jm_vfilter4_pass2_int32(x0+0,  x1+0,  x2+0,  x3+0,  x4+0,  x5+0,  k512);
      __m128i r32_1 = jm_vfilter4_pass2_int32(x0+4,  x1+4,  x2+4,  x3+4,  x4+4,  x5+4,  k512);
      __m128i r32_2 = jm_vfilter4_pass2_int32(x0+8,  x1+8,  x2+8,  x3+8,  x4+8,  x5+8,  k512);
      __m128i r32_3 = jm_vfilter4_pass2_int32(x0+12, x1+12, x2+12, x3+12, x4+12, x5+12, k512);
      __m128i r16_lo  = _mm_packs_epi32(r32_0, r32_1);
      __m128i r16_hi  = _mm_packs_epi32(r32_2, r32_3);
      __m128i r16_loc = _mm_min_epi16(_mm_max_epi16(r16_lo, zero), vmax);
      __m128i r16_hic = _mm_min_epi16(_mm_max_epi16(r16_hi, zero), vmax);
      __m128i r8      = _mm_packus_epi16(r16_loc, r16_hic);
      _mm_storeu_si128((__m128i *)dst, r8);
    }
  }
}

/* ===================================================================== */
/*   get_luma_21 / get_luma_23: X2 family (diagonal + half-pel average)   */
/* ===================================================================== */
/*  Pass 1: identical to get_luma_22 -- raw H 6-tap -> int32 tmp_res.     */
/*  Pass 2: per output pixel, compute TWO values and bilinear-average:    */
/*    j_pos = clip((V_filter(tmp_res int32) + 512) >> 10)   -- diag hpel  */
/*    h_pel = clip((tmp_res[j + hpel_row_off][i]    + 16) >> 5) -- H hpel */
/*    out   = (j_pos + h_pel + 1) >> 1                                    */
/*  hpel_row_off = 2 for _21 (central row of V window)                    */
/*               = 3 for _23 (one row below central)                      */
/* ===================================================================== */

/*!
 ************************************************************************
 * \brief Pass-2 helper shared by get_luma_21_sse / get_luma_23_sse.
 *        Produces j_pos (diagonal half-pel) and h_pel (half-pel from
 *        tmp_res[j + hpel_row_off] via hpel_finish on the narrowed
 *        int32->int16 values), then _mm_avg_epu8 of the two.
 *
 *        Width-4: low-4 store via _mm_cvtsi128_si32 + memcpy.
 *        Width-8: _mm_storel_epi64.
 *        Width-16: _mm_storeu_si128.
 *
 *        Narrowing tmp_res int32 -> int16 via _mm_packs_epi32 is exact
 *        (no saturation) because the raw H 6-tap result fits in int16
 *        for byte inputs (range ~[-2550, +10710]).
 ************************************************************************
 */
static __inline void jm_x2_pass2_diag_plus_hpel(imgpel **block, int **tmp_res,
                                                int block_size_y, int block_size_x,
                                                int hpel_row_off,
                                                __m128i k512, __m128i k16,
                                                __m128i vmax, __m128i zero)
{
  int j;

  if (block_size_x == 4)
  {
    for (j = 0; j < block_size_y; j++)
    {
      const int *x0 = tmp_res[j];     const int *x1 = tmp_res[j + 1];
      const int *x2 = tmp_res[j + 2]; const int *x3 = tmp_res[j + 3];
      const int *x4 = tmp_res[j + 4]; const int *x5 = tmp_res[j + 5];
      const int *xh = tmp_res[j + hpel_row_off];
      imgpel    *dst = block[j];

      /* j_pos: V filter int32 -> +512/>>10 -> int16 -> clip -> byte */
      __m128i r32  = jm_vfilter4_pass2_int32(x0, x1, x2, x3, x4, x5, k512);
      __m128i r16  = _mm_packs_epi32(r32, r32);
      __m128i r16c = _mm_min_epi16(_mm_max_epi16(r16, zero), vmax);
      __m128i jpos = _mm_packus_epi16(r16c, r16c);

      /* h_pel: load int32 -> narrow to int16 (exact) -> hpel_finish -> byte */
      __m128i h32  = _mm_loadu_si128((const __m128i *)xh);
      __m128i h16  = _mm_packs_epi32(h32, h32);
      __m128i h16f = jm_hpel_finish_epi16(h16, k16, vmax, zero);
      __m128i hpel = _mm_packus_epi16(h16f, h16f);

      __m128i avg = _mm_avg_epu8(jpos, hpel);
      int tmp = _mm_cvtsi128_si32(avg);
      memcpy(dst, &tmp, 4);
    }
  }
  else if (block_size_x == 8)
  {
    for (j = 0; j < block_size_y; j++)
    {
      const int *x0 = tmp_res[j];     const int *x1 = tmp_res[j + 1];
      const int *x2 = tmp_res[j + 2]; const int *x3 = tmp_res[j + 3];
      const int *x4 = tmp_res[j + 4]; const int *x5 = tmp_res[j + 5];
      const int *xh = tmp_res[j + hpel_row_off];
      imgpel    *dst = block[j];

      __m128i r32a = jm_vfilter4_pass2_int32(x0+0, x1+0, x2+0, x3+0, x4+0, x5+0, k512);
      __m128i r32b = jm_vfilter4_pass2_int32(x0+4, x1+4, x2+4, x3+4, x4+4, x5+4, k512);
      __m128i r16  = _mm_packs_epi32(r32a, r32b);
      __m128i r16c = _mm_min_epi16(_mm_max_epi16(r16, zero), vmax);
      __m128i jpos = _mm_packus_epi16(r16c, r16c);

      __m128i h32a = _mm_loadu_si128((const __m128i *)(xh + 0));
      __m128i h32b = _mm_loadu_si128((const __m128i *)(xh + 4));
      __m128i h16  = _mm_packs_epi32(h32a, h32b);
      __m128i h16f = jm_hpel_finish_epi16(h16, k16, vmax, zero);
      __m128i hpel = _mm_packus_epi16(h16f, h16f);

      __m128i avg = _mm_avg_epu8(jpos, hpel);
      _mm_storel_epi64((__m128i *)dst, avg);
    }
  }
  else  /* block_size_x == 16 (guaranteed by pass-1 assert) */
  {
    for (j = 0; j < block_size_y; j++)
    {
      const int *x0 = tmp_res[j];     const int *x1 = tmp_res[j + 1];
      const int *x2 = tmp_res[j + 2]; const int *x3 = tmp_res[j + 3];
      const int *x4 = tmp_res[j + 4]; const int *x5 = tmp_res[j + 5];
      const int *xh = tmp_res[j + hpel_row_off];
      imgpel    *dst = block[j];

      /* j_pos: 4 int32×4 chunks -> 2 int16×8 -> clip -> 16 bytes */
      __m128i r32_0 = jm_vfilter4_pass2_int32(x0+0,  x1+0,  x2+0,  x3+0,  x4+0,  x5+0,  k512);
      __m128i r32_1 = jm_vfilter4_pass2_int32(x0+4,  x1+4,  x2+4,  x3+4,  x4+4,  x5+4,  k512);
      __m128i r32_2 = jm_vfilter4_pass2_int32(x0+8,  x1+8,  x2+8,  x3+8,  x4+8,  x5+8,  k512);
      __m128i r32_3 = jm_vfilter4_pass2_int32(x0+12, x1+12, x2+12, x3+12, x4+12, x5+12, k512);
      __m128i r16_lo  = _mm_packs_epi32(r32_0, r32_1);
      __m128i r16_hi  = _mm_packs_epi32(r32_2, r32_3);
      __m128i r16_loc = _mm_min_epi16(_mm_max_epi16(r16_lo, zero), vmax);
      __m128i r16_hic = _mm_min_epi16(_mm_max_epi16(r16_hi, zero), vmax);
      __m128i jpos    = _mm_packus_epi16(r16_loc, r16_hic);

      /* h_pel: 4 int32×4 loads -> 2 int16×8 -> finish each -> 16 bytes */
      __m128i h32_0 = _mm_loadu_si128((const __m128i *)(xh + 0));
      __m128i h32_1 = _mm_loadu_si128((const __m128i *)(xh + 4));
      __m128i h32_2 = _mm_loadu_si128((const __m128i *)(xh + 8));
      __m128i h32_3 = _mm_loadu_si128((const __m128i *)(xh + 12));
      __m128i h16_lo  = _mm_packs_epi32(h32_0, h32_1);
      __m128i h16_hi  = _mm_packs_epi32(h32_2, h32_3);
      __m128i h16_lof = jm_hpel_finish_epi16(h16_lo, k16, vmax, zero);
      __m128i h16_hif = jm_hpel_finish_epi16(h16_hi, k16, vmax, zero);
      __m128i hpel    = _mm_packus_epi16(h16_lof, h16_hif);

      __m128i avg = _mm_avg_epu8(jpos, hpel);
      _mm_storeu_si128((__m128i *)dst, avg);
    }
  }
}

void get_luma_21_sse(imgpel **block, imgpel **cur_imgY, int **tmp_res,
                     int block_size_y, int block_size_x, int x_pos,
                     int max_imgpel_value)
{
  const __m128i k5   = _mm_set1_epi16(5);
  const __m128i k20  = _mm_set1_epi16(20);
  const __m128i k16  = _mm_set1_epi16(16);
  const __m128i k512 = _mm_set1_epi32(512);
  const __m128i zero = _mm_setzero_si128();
  const __m128i vmax = _mm_set1_epi16((short)max_imgpel_value);

  assert(sizeof(imgpel) == 1);

  /* Pass 1: shared raw H 6-tap into int32 tmp_res. */
  jm_x2_pass1_hraw_to_tmpres(cur_imgY, tmp_res, block_size_y, block_size_x,
                             x_pos, k5, k20, zero);

  /* Pass 2: diagonal half-pel averaged with h-pel from tmp_res row j+2
   * (the central row of the V filter window, which IS the h-pel position
   * for output row j -- spec position 21 is (dx=2, dy=1)). */
  jm_x2_pass2_diag_plus_hpel(block, tmp_res, block_size_y, block_size_x,
                             /*hpel_row_off=*/2, k512, k16, vmax, zero);
}

void get_luma_23_sse(imgpel **block, imgpel **cur_imgY, int **tmp_res,
                     int block_size_y, int block_size_x, int x_pos,
                     int max_imgpel_value)
{
  const __m128i k5   = _mm_set1_epi16(5);
  const __m128i k20  = _mm_set1_epi16(20);
  const __m128i k16  = _mm_set1_epi16(16);
  const __m128i k512 = _mm_set1_epi32(512);
  const __m128i zero = _mm_setzero_si128();
  const __m128i vmax = _mm_set1_epi16((short)max_imgpel_value);

  assert(sizeof(imgpel) == 1);

  jm_x2_pass1_hraw_to_tmpres(cur_imgY, tmp_res, block_size_y, block_size_x,
                             x_pos, k5, k20, zero);

  /* Spec position 23 is (dx=2, dy=3): h-pel partner is one row below
   * the central row -- tmp_res[j + 3]. */
  jm_x2_pass2_diag_plus_hpel(block, tmp_res, block_size_y, block_size_x,
                             /*hpel_row_off=*/3, k512, k16, vmax, zero);
}

/* ===================================================================== */
/*   get_luma_12 / get_luma_32: 2Y family (diagonal + V half-pel average) */
/* ===================================================================== */
/*  Mirror of the X2 family with the H/V axes swapped:                    */
/*    Pass 1: RAW V 6-tap on bytes -> int32 tmp_res                       */
/*            block_size_y rows, each (block_size_x + 5) ints wide.       */
/*    Pass 2: H 6-tap on tmp_res int32 -> +512/>>10/clip -> j_pos byte.   */
/*            Also load v_pel = clip((tmp_res[j][vpel_col_off+i] + 16)    */
/*                                   >> 5).                               */
/*            out = (j_pos + v_pel + 1) >> 1                              */
/*  vpel_col_off = 2 for _12 (central column of H window).                */
/*               = 3 for _32 (one column right of central).               */
/*                                                                        */
/*  Tail handling for pass 1: block_size_x+5 is 9/13/21, never a          */
/*  multiple of 8. tmp_res rows are [21] wide (allocated [21][21]) so     */
/*  for width=16 (need 21) we can't over-compute past 21. Use an          */
/*  overlap-tail: process full 8-chunks, then a final 8-chunk shifted     */
/*  to end at need-1. Overlapping writes store identical values (the V    */
/*  filter at a given column is deterministic), so overwriting is safe.   */
/* ===================================================================== */

/*!
 ************************************************************************
 * \brief Pass-2 horizontal 6-tap on int32 lanes from a single tmp_res
 *        row at sliding offsets 0..5. Mirror of jm_vfilter4_pass2_int32
 *        for the swapped axis. Returns int32x4 after +512/>>10.
 ************************************************************************
 */
static __inline __m128i jm_hfilter4_pass2_int32(const int *xrow, __m128i k512)
{
  __m128i v0  = _mm_loadu_si128((const __m128i *)(xrow + 0));
  __m128i v1  = _mm_loadu_si128((const __m128i *)(xrow + 1));
  __m128i v2  = _mm_loadu_si128((const __m128i *)(xrow + 2));
  __m128i v3  = _mm_loadu_si128((const __m128i *)(xrow + 3));
  __m128i v4  = _mm_loadu_si128((const __m128i *)(xrow + 4));
  __m128i v5  = _mm_loadu_si128((const __m128i *)(xrow + 5));
  __m128i raw = jm_filter6tap_epi32(v0, v1, v2, v3, v4, v5);
  return jm_dhpel_finish_epi32(raw, k512);
}

/*!
 ************************************************************************
 * \brief Pass-1 helper for the 2Y family (12, 32): vertical RAW 6-tap on
 *        bytes, output (block_size_x + 5) int32s per row over block_size_y
 *        rows. Uses an overlap-tail when (block_size_x + 5) isn't a
 *        multiple of 8 (always, in practice: 9 / 13 / 21).
 ************************************************************************
 */
static __inline void jm_2y_pass1_vraw_to_tmpres(imgpel **cur_imgY, int **tmp_res,
                                                int block_size_y, int block_size_x,
                                                int x_pos,
                                                __m128i k5, __m128i k20,
                                                __m128i zero)
{
  const int need        = block_size_x + 5;          /* 9, 13, or 21 */
  const int full_chunks = need / 8;                  /* 1, 1, or 2 */
  const int tail        = need - full_chunks * 8;    /* 1, 5, or 5 */
  const int tail_start  = need - 8;                  /* 1, 5, or 13 */
  int j, i;

  assert((block_size_x == 4 || block_size_x == 8 || block_size_x == 16) &&
         "2Y pass-1 expects block_size_x in {4, 8, 16}");

  for (j = 0; j < block_size_y; j++)
  {
    const imgpel *r0 = &cur_imgY[j - 2][x_pos - 2];
    const imgpel *r1 = &cur_imgY[j - 1][x_pos - 2];
    const imgpel *r2 = &cur_imgY[j    ][x_pos - 2];
    const imgpel *r3 = &cur_imgY[j + 1][x_pos - 2];
    const imgpel *r4 = &cur_imgY[j + 2][x_pos - 2];
    const imgpel *r5 = &cur_imgY[j + 3][x_pos - 2];
    int          *dst = tmp_res[j];

    for (i = 0; i < full_chunks * 8; i += 8)
    {
      __m128i s16  = jm_vfilter8_epi16(r0+i, r1+i, r2+i, r3+i, r4+i, r5+i,
                                       k5, k20, zero);
      __m128i sign = _mm_srai_epi16(s16, 15);
      __m128i lo32 = _mm_unpacklo_epi16(s16, sign);
      __m128i hi32 = _mm_unpackhi_epi16(s16, sign);
      _mm_storeu_si128((__m128i *)(dst + i + 0), lo32);
      _mm_storeu_si128((__m128i *)(dst + i + 4), hi32);
    }
    if (tail > 0)
    {
      /* Overlap tail: covers lanes [tail_start .. need-1]. The first
       * (8 - tail) lanes of this chunk overlap with the prior full
       * chunk and are overwritten with identical values. */
      __m128i s16  = jm_vfilter8_epi16(r0+tail_start, r1+tail_start, r2+tail_start,
                                       r3+tail_start, r4+tail_start, r5+tail_start,
                                       k5, k20, zero);
      __m128i sign = _mm_srai_epi16(s16, 15);
      __m128i lo32 = _mm_unpacklo_epi16(s16, sign);
      __m128i hi32 = _mm_unpackhi_epi16(s16, sign);
      _mm_storeu_si128((__m128i *)(dst + tail_start + 0), lo32);
      _mm_storeu_si128((__m128i *)(dst + tail_start + 4), hi32);
    }
  }
}

/*!
 ************************************************************************
 * \brief Pass-2 helper for the 2Y family: H 6-tap on int32 tmp_res +
 *        v-pel averaging. Mirror of jm_x2_pass2_diag_plus_hpel with the
 *        H/V axes swapped.
 ************************************************************************
 */
static __inline void jm_2y_pass2_diag_plus_vpel(imgpel **block, int **tmp_res,
                                                int block_size_y, int block_size_x,
                                                int vpel_col_off,
                                                __m128i k512, __m128i k16,
                                                __m128i vmax, __m128i zero)
{
  int j;

  if (block_size_x == 4)
  {
    for (j = 0; j < block_size_y; j++)
    {
      const int *row = tmp_res[j];
      imgpel    *dst = block[j];

      /* j_pos: 1 chunk of 4 H-filter outputs -> int16 -> clip -> byte */
      __m128i r32  = jm_hfilter4_pass2_int32(row, k512);
      __m128i r16  = _mm_packs_epi32(r32, r32);
      __m128i r16c = _mm_min_epi16(_mm_max_epi16(r16, zero), vmax);
      __m128i jpos = _mm_packus_epi16(r16c, r16c);

      /* v_pel: load 4 int32 from tmp_res[j][vpel_col_off..], narrow, finish, byte */
      __m128i v32  = _mm_loadu_si128((const __m128i *)(row + vpel_col_off));
      __m128i v16  = _mm_packs_epi32(v32, v32);
      __m128i v16f = jm_hpel_finish_epi16(v16, k16, vmax, zero);
      __m128i vpel = _mm_packus_epi16(v16f, v16f);

      __m128i avg = _mm_avg_epu8(jpos, vpel);
      int tmp = _mm_cvtsi128_si32(avg);
      memcpy(dst, &tmp, 4);
    }
  }
  else if (block_size_x == 8)
  {
    for (j = 0; j < block_size_y; j++)
    {
      const int *row = tmp_res[j];
      imgpel    *dst = block[j];

      __m128i r32a = jm_hfilter4_pass2_int32(row + 0, k512);
      __m128i r32b = jm_hfilter4_pass2_int32(row + 4, k512);
      __m128i r16  = _mm_packs_epi32(r32a, r32b);
      __m128i r16c = _mm_min_epi16(_mm_max_epi16(r16, zero), vmax);
      __m128i jpos = _mm_packus_epi16(r16c, r16c);

      __m128i v32a = _mm_loadu_si128((const __m128i *)(row + vpel_col_off + 0));
      __m128i v32b = _mm_loadu_si128((const __m128i *)(row + vpel_col_off + 4));
      __m128i v16  = _mm_packs_epi32(v32a, v32b);
      __m128i v16f = jm_hpel_finish_epi16(v16, k16, vmax, zero);
      __m128i vpel = _mm_packus_epi16(v16f, v16f);

      __m128i avg = _mm_avg_epu8(jpos, vpel);
      _mm_storel_epi64((__m128i *)dst, avg);
    }
  }
  else  /* block_size_x == 16 (guaranteed by pass-1 assert) */
  {
    for (j = 0; j < block_size_y; j++)
    {
      const int *row = tmp_res[j];
      imgpel    *dst = block[j];

      __m128i r32_0 = jm_hfilter4_pass2_int32(row + 0,  k512);
      __m128i r32_1 = jm_hfilter4_pass2_int32(row + 4,  k512);
      __m128i r32_2 = jm_hfilter4_pass2_int32(row + 8,  k512);
      __m128i r32_3 = jm_hfilter4_pass2_int32(row + 12, k512);
      __m128i r16_lo  = _mm_packs_epi32(r32_0, r32_1);
      __m128i r16_hi  = _mm_packs_epi32(r32_2, r32_3);
      __m128i r16_loc = _mm_min_epi16(_mm_max_epi16(r16_lo, zero), vmax);
      __m128i r16_hic = _mm_min_epi16(_mm_max_epi16(r16_hi, zero), vmax);
      __m128i jpos    = _mm_packus_epi16(r16_loc, r16_hic);

      __m128i v32_0 = _mm_loadu_si128((const __m128i *)(row + vpel_col_off + 0));
      __m128i v32_1 = _mm_loadu_si128((const __m128i *)(row + vpel_col_off + 4));
      __m128i v32_2 = _mm_loadu_si128((const __m128i *)(row + vpel_col_off + 8));
      __m128i v32_3 = _mm_loadu_si128((const __m128i *)(row + vpel_col_off + 12));
      __m128i v16_lo  = _mm_packs_epi32(v32_0, v32_1);
      __m128i v16_hi  = _mm_packs_epi32(v32_2, v32_3);
      __m128i v16_lof = jm_hpel_finish_epi16(v16_lo, k16, vmax, zero);
      __m128i v16_hif = jm_hpel_finish_epi16(v16_hi, k16, vmax, zero);
      __m128i vpel    = _mm_packus_epi16(v16_lof, v16_hif);

      __m128i avg = _mm_avg_epu8(jpos, vpel);
      _mm_storeu_si128((__m128i *)dst, avg);
    }
  }
}

void get_luma_12_sse(imgpel **block, imgpel **cur_imgY, int **tmp_res,
                     int block_size_y, int block_size_x, int x_pos,
                     int shift_x, int max_imgpel_value)
{
  const __m128i k5   = _mm_set1_epi16(5);
  const __m128i k20  = _mm_set1_epi16(20);
  const __m128i k16  = _mm_set1_epi16(16);
  const __m128i k512 = _mm_set1_epi32(512);
  const __m128i zero = _mm_setzero_si128();
  const __m128i vmax = _mm_set1_epi16((short)max_imgpel_value);
  (void)shift_x;   /* SIMD uses cur_imgY[k] row indexing; see note for 02/01/03 */

  assert(sizeof(imgpel) == 1);

  jm_2y_pass1_vraw_to_tmpres(cur_imgY, tmp_res, block_size_y, block_size_x,
                             x_pos, k5, k20, zero);

  /* Spec position 12 is (dx=1, dy=2): v-pel partner is the central column
   * of the H window -- tmp_res[j][i + 2]. */
  jm_2y_pass2_diag_plus_vpel(block, tmp_res, block_size_y, block_size_x,
                             /*vpel_col_off=*/2, k512, k16, vmax, zero);
}

void get_luma_32_sse(imgpel **block, imgpel **cur_imgY, int **tmp_res,
                     int block_size_y, int block_size_x, int x_pos,
                     int shift_x, int max_imgpel_value)
{
  const __m128i k5   = _mm_set1_epi16(5);
  const __m128i k20  = _mm_set1_epi16(20);
  const __m128i k16  = _mm_set1_epi16(16);
  const __m128i k512 = _mm_set1_epi32(512);
  const __m128i zero = _mm_setzero_si128();
  const __m128i vmax = _mm_set1_epi16((short)max_imgpel_value);
  (void)shift_x;

  assert(sizeof(imgpel) == 1);

  jm_2y_pass1_vraw_to_tmpres(cur_imgY, tmp_res, block_size_y, block_size_x,
                             x_pos, k5, k20, zero);

  /* Spec position 32 is (dx=3, dy=2): v-pel partner is one column right
   * of central -- tmp_res[j][i + 3]. */
  jm_2y_pass2_diag_plus_vpel(block, tmp_res, block_size_y, block_size_x,
                             /*vpel_col_off=*/3, k512, k16, vmax, zero);
}

/* ===================================================================== */
/*   get_luma_11 / 13 / 31 / 33: pure diagonal quarter-pels               */
/* ===================================================================== */
/*  Bilinear average of two half-pel samples computed independently:      */
/*    H_hpel  = clip((H_filter(row j + h_row_off, cols x_pos-2..x_pos+3)  */
/*                    + 16) >> 5)                                         */
/*    V_hpel  = clip((V_filter(col x_pos + v_col_off, rows j-2..j+3)      */
/*                    + 16) >> 5)                                         */
/*    out     = (H_hpel + V_hpel + 1) >> 1   (i.e. _mm_avg_epu8)          */
/*                                                                        */
/*    pos   h_row_off   v_col_off                                         */
/*    ---   ----------  ----------                                        */
/*    11    0           0                                                 */
/*    13    1           0                                                 */
/*    31    0           1                                                 */
/*    33    1           1                                                 */
/*                                                                        */
/*  No tmp_res involved. Compose existing jm_hpel_h8_epi16 +              */
/*  jm_hpel_v8_epi16 building blocks.                                     */
/* ===================================================================== */

/*!
 ************************************************************************
 * \brief Shared body for the four diagonal qpels. h_row_off shifts the
 *        H half-pel source row by 0 or 1 (selects "upper" vs "lower"
 *        integer row for the H midpoint). v_col_off shifts the V
 *        half-pel source column by 0 or 1.
 ************************************************************************
 */
static __inline void jm_qpel_diag_common(imgpel **block, imgpel **cur_imgY,
                                         int block_size_y, int block_size_x,
                                         int x_pos, int max_imgpel_value,
                                         int h_row_off, int v_col_off)
{
  const __m128i k5   = _mm_set1_epi16(5);
  const __m128i k20  = _mm_set1_epi16(20);
  const __m128i k16  = _mm_set1_epi16(16);
  const __m128i zero = _mm_setzero_si128();
  const __m128i vmax = _mm_set1_epi16((short)max_imgpel_value);
  int j, i;

  assert(sizeof(imgpel) == 1);

  if (block_size_x == 4)
  {
    for (j = 0; j < block_size_y; j++)
    {
      const imgpel *hsrc = &cur_imgY[j + h_row_off][x_pos - 2];
      const imgpel *r0   = &cur_imgY[j - 2][x_pos + v_col_off];
      const imgpel *r1   = &cur_imgY[j - 1][x_pos + v_col_off];
      const imgpel *r2   = &cur_imgY[j    ][x_pos + v_col_off];
      const imgpel *r3   = &cur_imgY[j + 1][x_pos + v_col_off];
      const imgpel *r4   = &cur_imgY[j + 2][x_pos + v_col_off];
      const imgpel *r5   = &cur_imgY[j + 3][x_pos + v_col_off];

      __m128i h_short = jm_hpel_h8_epi16(hsrc, k5, k20, k16, vmax, zero);
      __m128i v_short = jm_hpel_v8_epi16(r0, r1, r2, r3, r4, r5,
                                         k5, k20, k16, vmax, zero);
      __m128i h_byte  = _mm_packus_epi16(h_short, h_short);
      __m128i v_byte  = _mm_packus_epi16(v_short, v_short);
      __m128i avg     = _mm_avg_epu8(h_byte, v_byte);
      int tmp = _mm_cvtsi128_si32(avg);
      memcpy(block[j], &tmp, 4);
    }
  }
  else  /* block_size_x in {8, 16} */
  {
    for (j = 0; j < block_size_y; j++)
    {
      const imgpel *hsrc = &cur_imgY[j + h_row_off][x_pos - 2];
      const imgpel *r0   = &cur_imgY[j - 2][x_pos + v_col_off];
      const imgpel *r1   = &cur_imgY[j - 1][x_pos + v_col_off];
      const imgpel *r2   = &cur_imgY[j    ][x_pos + v_col_off];
      const imgpel *r3   = &cur_imgY[j + 1][x_pos + v_col_off];
      const imgpel *r4   = &cur_imgY[j + 2][x_pos + v_col_off];
      const imgpel *r5   = &cur_imgY[j + 3][x_pos + v_col_off];
      imgpel       *dst  = block[j];

      for (i = 0; i < block_size_x; i += 8)
      {
        __m128i h_short = jm_hpel_h8_epi16(hsrc + i, k5, k20, k16, vmax, zero);
        __m128i v_short = jm_hpel_v8_epi16(r0+i, r1+i, r2+i, r3+i, r4+i, r5+i,
                                           k5, k20, k16, vmax, zero);
        __m128i h_byte  = _mm_packus_epi16(h_short, h_short);
        __m128i v_byte  = _mm_packus_epi16(v_short, v_short);
        __m128i avg     = _mm_avg_epu8(h_byte, v_byte);
        _mm_storel_epi64((__m128i *)(dst + i), avg);
      }
    }
  }
}

void get_luma_11_sse(imgpel **block, imgpel **cur_imgY, int block_size_y,
                     int block_size_x, int x_pos, int shift_x, int max_imgpel_value)
{
  (void)shift_x;   /* SIMD uses cur_imgY[k] row indexing */
  jm_qpel_diag_common(block, cur_imgY, block_size_y, block_size_x,
                      x_pos, max_imgpel_value,
                      /*h_row_off=*/0, /*v_col_off=*/0);
}

void get_luma_13_sse(imgpel **block, imgpel **cur_imgY, int block_size_y,
                     int block_size_x, int x_pos, int shift_x, int max_imgpel_value)
{
  (void)shift_x;
  jm_qpel_diag_common(block, cur_imgY, block_size_y, block_size_x,
                      x_pos, max_imgpel_value,
                      /*h_row_off=*/1, /*v_col_off=*/0);
}

void get_luma_31_sse(imgpel **block, imgpel **cur_imgY, int block_size_y,
                     int block_size_x, int x_pos, int shift_x, int max_imgpel_value)
{
  (void)shift_x;
  jm_qpel_diag_common(block, cur_imgY, block_size_y, block_size_x,
                      x_pos, max_imgpel_value,
                      /*h_row_off=*/0, /*v_col_off=*/1);
}

void get_luma_33_sse(imgpel **block, imgpel **cur_imgY, int block_size_y,
                     int block_size_x, int x_pos, int shift_x, int max_imgpel_value)
{
  (void)shift_x;
  jm_qpel_diag_common(block, cur_imgY, block_size_y, block_size_x,
                      x_pos, max_imgpel_value,
                      /*h_row_off=*/1, /*v_col_off=*/1);
}

/* ===================================================================== */
/*   get_chroma_0X / get_chroma_X0: single-axis chroma bilinear           */
/* ===================================================================== */
/*  Per-output bilinear interpolation:                                    */
/*    out[i] = (wA * a[i] + wB * b[i] + (1 << (total_scale - 1)))         */
/*              >> total_scale                                            */
/*  where (a, b) differ per kernel:                                       */
/*    0X  -- a = row j, b = row j+1   (Y-axis bilinear)                   */
/*    X0  -- a = col 0, b = col 1     (X-axis bilinear within one row)    */
/*                                                                        */
/*  H.264 weights are non-negative (4:2:0 chroma: wA + wB = 64,           */
/*  total_scale = 6), so the rounded sum fits in [0, 255] without         */
/*  needing an explicit clip -- packus_epi16 handles narrowing.           */
/*                                                                        */
/*  Output stride is HARDCODED at 16 bytes per row regardless of          */
/*  block_size_x (JM MC buffer convention; matches the scalar             */
/*  `block += 16`).                                                       */
/*                                                                        */
/*  block_size_x is 2, 4, or 8 for 4:2:0 chroma. We always compute 8      */
/*  lanes (over-reads are safe due to chroma reference padding) and       */
/*  store the low 8 / 4 / 2 bytes per row.                                */
/* ===================================================================== */

/*!
 ************************************************************************
 * \brief Compute 8 lanes of chroma bilinear: (wA*a + wB*b + round) >> s.
 *        Returns int16 lanes (low 8 valid). Pack to bytes at call site.
 *
 *        shift_count must be a __m128i with `total_scale` in its low
 *        64 bits (use _mm_cvtsi32_si128(total_scale)). _mm_srl_epi16
 *        uses the SAME shift for all lanes from the low quadword.
 ************************************************************************
 */
static __inline __m128i jm_chroma_bilinear8_epi16(const imgpel *a, const imgpel *b,
                                                  __m128i wA, __m128i wB,
                                                  __m128i round, __m128i shift_count,
                                                  __m128i zero)
{
  __m128i av  = _mm_loadl_epi64((const __m128i *)a);
  __m128i bv  = _mm_loadl_epi64((const __m128i *)b);
  __m128i a16 = _mm_unpacklo_epi8(av, zero);
  __m128i b16 = _mm_unpacklo_epi8(bv, zero);
  __m128i pa  = _mm_mullo_epi16(a16, wA);
  __m128i pb  = _mm_mullo_epi16(b16, wB);
  __m128i sum = _mm_add_epi16(pa, pb);
  sum = _mm_add_epi16(sum, round);
  return _mm_srl_epi16(sum, shift_count);
}

/*!
 ************************************************************************
 * \brief Pack the 8-lane int16 chroma result to bytes and store the low
 *        block_size_x bytes. The `if` on a loop-invariant value gets
 *        hoisted (loop unswitched) by both MSVC and GCC.
 ************************************************************************
 */
static __inline void jm_chroma_store_row(imgpel *dst, __m128i result_int16,
                                         int block_size_x)
{
  __m128i bytes = _mm_packus_epi16(result_int16, result_int16);
  if (block_size_x == 8)
  {
    _mm_storel_epi64((__m128i *)dst, bytes);
  }
  else if (block_size_x == 4)
  {
    int tmp = _mm_cvtsi128_si32(bytes);
    memcpy(dst, &tmp, 4);
  }
  else   /* block_size_x == 2 */
  {
    int tmp = _mm_cvtsi128_si32(bytes);
    memcpy(dst, &tmp, 2);   /* low 16 bits = first 2 output bytes (LE) */
  }
}

void get_chroma_0X_sse(imgpel *block, imgpel *cur_img, int span,
                       int block_size_y, int block_size_x,
                       int w00, int w01, int total_scale)
{
  const __m128i wA          = _mm_set1_epi16((short)w00);
  const __m128i wB          = _mm_set1_epi16((short)w01);
  const __m128i round       = _mm_set1_epi16((short)(1 << (total_scale - 1)));
  const __m128i shift_count = _mm_cvtsi32_si128(total_scale);
  const __m128i zero        = _mm_setzero_si128();
  const imgpel *cur_row = cur_img;
  const imgpel *nxt_row = cur_img + span;
  int j;

  assert(sizeof(imgpel) == 1);
  assert((block_size_x == 2 || block_size_x == 4 || block_size_x == 8) &&
         "chroma 0X expects block_size_x in {2, 4, 8}");

  for (j = 0; j < block_size_y; j++)
  {
    __m128i r = jm_chroma_bilinear8_epi16(cur_row, nxt_row, wA, wB,
                                          round, shift_count, zero);
    jm_chroma_store_row(block, r, block_size_x);
    block   += 16;             /* fixed JM MC stride */
    cur_row  = nxt_row;
    nxt_row += span;
  }
}

void get_chroma_X0_sse(imgpel *block, imgpel *cur_img, int span,
                       int block_size_y, int block_size_x,
                       int w00, int w10, int total_scale)
{
  const __m128i wA          = _mm_set1_epi16((short)w00);
  const __m128i wB          = _mm_set1_epi16((short)w10);
  const __m128i round       = _mm_set1_epi16((short)(1 << (total_scale - 1)));
  const __m128i shift_count = _mm_cvtsi32_si128(total_scale);
  const __m128i zero        = _mm_setzero_si128();
  const imgpel *cur_row = cur_img;
  int j;

  assert(sizeof(imgpel) == 1);
  assert((block_size_x == 2 || block_size_x == 4 || block_size_x == 8) &&
         "chroma X0 expects block_size_x in {2, 4, 8}");

  for (j = 0; j < block_size_y; j++)
  {
    /* X0 partner is one byte to the right within the SAME row. */
    __m128i r = jm_chroma_bilinear8_epi16(cur_row, cur_row + 1, wA, wB,
                                          round, shift_count, zero);
    jm_chroma_store_row(block, r, block_size_x);
    block   += 16;
    cur_row += span;
  }
}

/* ===================================================================== */
/*   get_chroma_XY: full 2D chroma bilinear (4-corner interpolation)      */
/* ===================================================================== */
/*  out[j][i] = (w00 * src[j  ][i  ] + w01 * src[j+1][i  ]                */
/*             + w10 * src[j  ][i+1] + w11 * src[j+1][i+1]                */
/*             + (1 << (total_scale - 1))) >> total_scale                 */
/*                                                                        */
/*  H.264 4:2:0: w00+w01+w10+w11 = 64, total_scale = 6, so max sum is     */
/*  64*255 + 32 (round) = 16352 -- fits in int16. Output stays in         */
/*  [0, 255]; packus_epi16 narrows int16 -> byte.                         */
/* ===================================================================== */

/*!
 ************************************************************************
 * \brief Compute 8 lanes of 2D chroma bilinear. The 4 corner loads are
 *        4 unaligned 8-byte reads from cur_row/nxt_row at offsets 0 and 1.
 ************************************************************************
 */
static __inline __m128i jm_chroma_bilinear_xy8_epi16(
    const imgpel *a, const imgpel *b, const imgpel *c, const imgpel *d,
    __m128i w00v, __m128i w01v, __m128i w10v, __m128i w11v,
    __m128i round, __m128i shift_count, __m128i zero)
{
  __m128i av  = _mm_loadl_epi64((const __m128i *)a);
  __m128i bv  = _mm_loadl_epi64((const __m128i *)b);
  __m128i cv  = _mm_loadl_epi64((const __m128i *)c);
  __m128i dv  = _mm_loadl_epi64((const __m128i *)d);
  __m128i a16 = _mm_unpacklo_epi8(av, zero);
  __m128i b16 = _mm_unpacklo_epi8(bv, zero);
  __m128i c16 = _mm_unpacklo_epi8(cv, zero);
  __m128i d16 = _mm_unpacklo_epi8(dv, zero);
  __m128i pa  = _mm_mullo_epi16(a16, w00v);
  __m128i pb  = _mm_mullo_epi16(b16, w01v);
  __m128i pc  = _mm_mullo_epi16(c16, w10v);
  __m128i pd  = _mm_mullo_epi16(d16, w11v);
  __m128i sum = _mm_add_epi16(_mm_add_epi16(pa, pb), _mm_add_epi16(pc, pd));
  sum = _mm_add_epi16(sum, round);
  return _mm_srl_epi16(sum, shift_count);
}

void get_chroma_XY_sse(imgpel *block, imgpel *cur_img, int span,
                       int block_size_y, int block_size_x,
                       int w00, int w01, int w10, int w11, int total_scale)
{
  const __m128i w00v        = _mm_set1_epi16((short)w00);
  const __m128i w01v        = _mm_set1_epi16((short)w01);
  const __m128i w10v        = _mm_set1_epi16((short)w10);
  const __m128i w11v        = _mm_set1_epi16((short)w11);
  const __m128i round       = _mm_set1_epi16((short)(1 << (total_scale - 1)));
  const __m128i shift_count = _mm_cvtsi32_si128(total_scale);
  const __m128i zero        = _mm_setzero_si128();
  const imgpel *cur_row = cur_img;
  const imgpel *nxt_row = cur_img + span;
  int j;

  assert(sizeof(imgpel) == 1);
  assert((block_size_x == 2 || block_size_x == 4 || block_size_x == 8) &&
         "chroma XY expects block_size_x in {2, 4, 8}");

  for (j = 0; j < block_size_y; j++)
  {
    /* Four corners of the 2x2 bilinear cell, all reachable via offsets
     * 0/+1 byte and 0/+span row from the current cell origin. */
    __m128i r = jm_chroma_bilinear_xy8_epi16(cur_row,     nxt_row,
                                             cur_row + 1, nxt_row + 1,
                                             w00v, w01v, w10v, w11v,
                                             round, shift_count, zero);
    jm_chroma_store_row(block, r, block_size_x);
    block   += 16;
    cur_row  = nxt_row;
    nxt_row += span;
  }
}

/* ===================================================================== */
/*   recon8x8: residual + prediction + clip for 8x8 block                */
/* ===================================================================== */
/*  Per-pixel: mb_rec[j][i] = clip(0, max,                                */
/*                                 mpr[j][i] + ((m7[j][i] + 32) >> 6))    */
/*  where the shift constant is DQ_BITS_8 (= 6) and the round constant is */
/*  1 << (DQ_BITS_8 - 1) = 32. Fixed 8x8 block (8 rows of 8 pixels each). */
/*                                                                        */
/*  Per row: load 8 int32 residuals (2 SSE2 regs of 4 each), add 32,      */
/*  arithmetic-shift right by 6, signed-saturate to int16 via             */
/*  _mm_packs_epi32, add 8 int16 prediction lanes, clip to [0, max_imgpel */
/*  _value] via max/min_epi16, unsigned-saturate to bytes via             */
/*  _mm_packus_epi16, store 8 bytes.                                      */
/* ===================================================================== */
void recon8x8_sse(int **m7, imgpel **mb_rec, imgpel **mpr,
                  int max_imgpel_value, int ioff)
{
  const __m128i round_v = _mm_set1_epi32(1 << (DQ_BITS_8 - 1));
  const __m128i zero    = _mm_setzero_si128();
  const __m128i vmax    = _mm_set1_epi16((short)max_imgpel_value);
  int j;

  assert(sizeof(imgpel) == 1);

  for (j = 0; j < 8; j++)
  {
    const int    *m_tr  = m7[j]      + ioff;
    const imgpel *m_prd = mpr[j]     + ioff;
    imgpel       *m_rec = mb_rec[j]  + ioff;

    /* Load 8 int32 residuals into two 4-lane vectors. */
    __m128i r_lo = _mm_loadu_si128((const __m128i *)(m_tr + 0));
    __m128i r_hi = _mm_loadu_si128((const __m128i *)(m_tr + 4));

    /* Round + arithmetic shift right by DQ_BITS_8 (immediate). */
    r_lo = _mm_srai_epi32(_mm_add_epi32(r_lo, round_v), DQ_BITS_8);
    r_hi = _mm_srai_epi32(_mm_add_epi32(r_hi, round_v), DQ_BITS_8);

    /* int32 -> int16 (signed sat). After the round+shift the values are
     * already in the int16 range so the saturation is a no-op. */
    __m128i res16 = _mm_packs_epi32(r_lo, r_hi);

    /* Load 8 prediction bytes -> int16. */
    __m128i pred_byte = _mm_loadl_epi64((const __m128i *)m_prd);
    __m128i pred16    = _mm_unpacklo_epi8(pred_byte, zero);

    /* mpr + residual, then clip to [0, max_imgpel_value]. */
    __m128i sum16 = _mm_add_epi16(pred16, res16);
    sum16 = _mm_max_epi16(sum16, zero);
    sum16 = _mm_min_epi16(sum16, vmax);

    /* int16 -> uint8 (unsigned sat) and store low 8 bytes. */
    __m128i sum8 = _mm_packus_epi16(sum16, sum16);
    _mm_storel_epi64((__m128i *)m_rec, sum8);
  }
}

/* ===================================================================== */
/*   Stage 4 -- sample_reconstruct: parameterized residual + pred + clip  */
/* ===================================================================== */
/*  Same per-pixel formula as recon8x8 but width / height / dq_bits are   */
/*  runtime values. Called from block.c at:                                */
/*    - 4x4 transform reconstruction (BLOCK_SIZE x BLOCK_SIZE = 4x4)       */
/*    - 16x16 luma reconstruction (MB_BLOCK_SIZE x MB_BLOCK_SIZE)          */
/*    - chroma reconstruction (8x8 for 4:2:0 16x16 MB)                     */
/*                                                                        */
/*  Per pixel: curImg[j][opix_x+i] = clip(0, max,                          */
/*    ((mb_rres[j][mb_x+i] + (1 << (dq_bits-1))) >> dq_bits) +             */
/*    mpr[j][mb_x+i])                                                      */
/*                                                                        */
/*  dq_bits is a runtime parameter so the shift uses _mm_sra_epi32 with a */
/*  count vector rather than the immediate srai variant.                  */
/* ===================================================================== */
void sample_reconstruct_sse(imgpel **curImg, imgpel **mpr, int **mb_rres,
                            int mb_x, int opix_x, int width, int height,
                            int max_imgpel_value, int dq_bits)
{
  const __m128i round_v     = _mm_set1_epi32(1 << (dq_bits - 1));
  const __m128i shift_count = _mm_cvtsi32_si128(dq_bits);
  const __m128i zero        = _mm_setzero_si128();
  const __m128i vmax        = _mm_set1_epi16((short)max_imgpel_value);
  int j;

  assert(sizeof(imgpel) == 1);

  if (width == 16)
  {
    for (j = 0; j < height; j++)
    {
      const int    *m7    = &mb_rres[j][mb_x];
      const imgpel *mpr_p = &mpr[j][mb_x];
      imgpel       *dst   = &curImg[j][opix_x];

      /* 16 int32 residuals as 4 chunks of 4. */
      __m128i r0 = _mm_loadu_si128((const __m128i *)(m7 +  0));
      __m128i r1 = _mm_loadu_si128((const __m128i *)(m7 +  4));
      __m128i r2 = _mm_loadu_si128((const __m128i *)(m7 +  8));
      __m128i r3 = _mm_loadu_si128((const __m128i *)(m7 + 12));
      r0 = _mm_sra_epi32(_mm_add_epi32(r0, round_v), shift_count);
      r1 = _mm_sra_epi32(_mm_add_epi32(r1, round_v), shift_count);
      r2 = _mm_sra_epi32(_mm_add_epi32(r2, round_v), shift_count);
      r3 = _mm_sra_epi32(_mm_add_epi32(r3, round_v), shift_count);

      /* int32 -> int16 (signed sat). After >>dq_bits the values fit. */
      __m128i res16_lo = _mm_packs_epi32(r0, r1);
      __m128i res16_hi = _mm_packs_epi32(r2, r3);

      /* 16 prediction bytes -> two int16 halves. */
      __m128i pred_byte = _mm_loadu_si128((const __m128i *)mpr_p);
      __m128i pred16_lo = _mm_unpacklo_epi8(pred_byte, zero);
      __m128i pred16_hi = _mm_unpackhi_epi8(pred_byte, zero);

      /* prediction + residual, clip, pack to bytes, store. */
      __m128i sum_lo = _mm_add_epi16(pred16_lo, res16_lo);
      __m128i sum_hi = _mm_add_epi16(pred16_hi, res16_hi);
      sum_lo = _mm_min_epi16(_mm_max_epi16(sum_lo, zero), vmax);
      sum_hi = _mm_min_epi16(_mm_max_epi16(sum_hi, zero), vmax);
      __m128i sum8 = _mm_packus_epi16(sum_lo, sum_hi);
      _mm_storeu_si128((__m128i *)dst, sum8);
    }
  }
  else if (width == 8)
  {
    for (j = 0; j < height; j++)
    {
      const int    *m7    = &mb_rres[j][mb_x];
      const imgpel *mpr_p = &mpr[j][mb_x];
      imgpel       *dst   = &curImg[j][opix_x];

      __m128i r0 = _mm_loadu_si128((const __m128i *)(m7 + 0));
      __m128i r1 = _mm_loadu_si128((const __m128i *)(m7 + 4));
      r0 = _mm_sra_epi32(_mm_add_epi32(r0, round_v), shift_count);
      r1 = _mm_sra_epi32(_mm_add_epi32(r1, round_v), shift_count);
      __m128i res16 = _mm_packs_epi32(r0, r1);

      __m128i pred_byte = _mm_loadl_epi64((const __m128i *)mpr_p);
      __m128i pred16    = _mm_unpacklo_epi8(pred_byte, zero);

      __m128i sum16 = _mm_add_epi16(pred16, res16);
      sum16 = _mm_min_epi16(_mm_max_epi16(sum16, zero), vmax);
      __m128i sum8  = _mm_packus_epi16(sum16, sum16);
      _mm_storel_epi64((__m128i *)dst, sum8);
    }
  }
  else  /* width == 4 (BLOCK_SIZE) */
  {
    for (j = 0; j < height; j++)
    {
      const int    *m7    = &mb_rres[j][mb_x];
      const imgpel *mpr_p = &mpr[j][mb_x];
      imgpel       *dst   = &curImg[j][opix_x];

      /* 4 int32 residuals into a single 4-lane vector. */
      __m128i r0 = _mm_loadu_si128((const __m128i *)m7);
      r0 = _mm_sra_epi32(_mm_add_epi32(r0, round_v), shift_count);
      __m128i res16 = _mm_packs_epi32(r0, r0);          /* low 4 lanes valid */

      /* 4 prediction bytes via cvtsi32_si128 + alignment-safe int read. */
      int pred_int;
      memcpy(&pred_int, mpr_p, 4);
      __m128i pred_byte = _mm_cvtsi32_si128(pred_int);
      __m128i pred16    = _mm_unpacklo_epi8(pred_byte, zero);

      __m128i sum16 = _mm_add_epi16(pred16, res16);
      sum16 = _mm_min_epi16(_mm_max_epi16(sum16, zero), vmax);
      __m128i sum8  = _mm_packus_epi16(sum16, sum16);
      int tmp = _mm_cvtsi128_si32(sum8);
      memcpy(dst, &tmp, 4);
    }
  }
}

/* ===================================================================== */
/*   Stage 4 -- get_block_00: integer-pel MC (full 16-byte row copy)      */
/* ===================================================================== */
/*  The scalar version is a memcpy(MB_BLOCK_SIZE) per row, unrolled by 2. */
/*  The SIMD version replaces each 16-byte memcpy with a 16-byte aligned- */
/*  -agnostic load/store. Output `block` is the 16-byte-strided MC temp;  */
/*  the source has at least 16 bytes of padding on each side so unaligned */
/*  loads are always safe.                                                */
/* ===================================================================== */
void get_block_00_sse(imgpel *block, imgpel *cur_img, int span, int block_size_y)
{
  int j;
  assert(sizeof(imgpel) == 1);
  for (j = 0; j < block_size_y; j += 2)
  {
    __m128i row0 = _mm_loadu_si128((const __m128i *)cur_img);
    cur_img += span;
    __m128i row1 = _mm_loadu_si128((const __m128i *)cur_img);
    cur_img += span;
    _mm_storeu_si128((__m128i *)block, row0);
    block += MB_BLOCK_SIZE;
    _mm_storeu_si128((__m128i *)block, row1);
    block += MB_BLOCK_SIZE;
  }
}

/* ===================================================================== */
/*   Stage 4 -- weighted_mc_prediction: single-reference weighted pred    */
/* ===================================================================== */
/*  result    = ((wp_scale * pel + (1 << (denom-1))) >> denom) + offset   */
/*  out[j][i] = clip(0, color_clip, result)                               */
/*                                                                        */
/*  block_size_x ranges over {2, 4, 8, 16}: luma uses 4/8/16, chroma      */
/*  uses 2/4/8. wp_scale fits in int16 (H.264 spec [-128, 127]). The      */
/*  product wp_scale * pel can be up to ~32k -- still int16. But the      */
/*  subsequent +round and +offset push us into int32 territory, so we     */
/*  sign-extend after the multiply.                                       */
/* ===================================================================== */

/* Process 8 lanes: pel16 -> ((wp*pel + round) >> denom) + offset, clipped
 * to [0, vmax]. Returns int16 lanes (low 8 valid) ready for packus. */
static __inline __m128i jm_wp_8lanes(__m128i pel16,
                                     __m128i wp_scale_v, __m128i round_v,
                                     __m128i shift_count, __m128i offset_v,
                                     __m128i zero, __m128i vmax)
{
  __m128i prod  = _mm_mullo_epi16(pel16, wp_scale_v);
  __m128i sign  = _mm_srai_epi16(prod, 15);             /* sign-extend lo/hi via unpack */
  __m128i lo32  = _mm_unpacklo_epi16(prod, sign);       /* 4 int32 */
  __m128i hi32  = _mm_unpackhi_epi16(prod, sign);       /* 4 int32 */
  lo32 = _mm_sra_epi32(_mm_add_epi32(lo32, round_v), shift_count);
  hi32 = _mm_sra_epi32(_mm_add_epi32(hi32, round_v), shift_count);
  lo32 = _mm_add_epi32(lo32, offset_v);
  hi32 = _mm_add_epi32(hi32, offset_v);
  /* Pack int32 -> int16 (signed sat; values fit in int16 since color_clip
   * is at most 255 for 8-bit) then clip explicit to [0, color_clip] in
   * case color_clip < 255. */
  __m128i res16 = _mm_packs_epi32(lo32, hi32);
  res16 = _mm_max_epi16(res16, zero);
  return _mm_min_epi16(res16, vmax);
}

void weighted_mc_prediction_sse(imgpel **mb_pred, imgpel **block,
                                int block_size_y, int block_size_x, int ioff,
                                int wp_scale, int wp_offset,
                                int weight_denom, int color_clip)
{
  /* Scalar rshift_rnd has an explicit `a > 0` guard: when weight_denom is
   * 0 it returns x unchanged (no rounding, no shift). H.264 spec allows
   * luma/chroma_log2_weight_denom == 0 and at least one Blu-ray stream
   * exercises it. We handle it in-band rather than delegating to scalar:
   *   - round_int = (denom > 0) ? (1 << (denom - 1)) : 0  (avoid UB)
   *   - _mm_sra_epi32(x, 0) is identity, so the shift step is a no-op
   *     when shift_count is zero.
   * Result: same code path bit-identical for denom == 0 and denom > 0. */
  const int round_int = (weight_denom > 0) ? (1 << (weight_denom - 1)) : 0;
  const __m128i wp_scale_v  = _mm_set1_epi16((short)wp_scale);
  const __m128i round_v     = _mm_set1_epi32(round_int);
  const __m128i shift_count = _mm_cvtsi32_si128(weight_denom);
  const __m128i offset_v    = _mm_set1_epi32(wp_offset);
  const __m128i zero        = _mm_setzero_si128();
  const __m128i vmax        = _mm_set1_epi16((short)color_clip);
  int j;

  assert(sizeof(imgpel) == 1);

  if (block_size_x == 16)
  {
    /* Two 8-lane halves per row, combine to 16 bytes via single packus. */
    for (j = 0; j < block_size_y; j++)
    {
      __m128i bytes   = _mm_loadu_si128((const __m128i *)block[j]);
      __m128i lo16   = _mm_unpacklo_epi8(bytes, zero);
      __m128i hi16   = _mm_unpackhi_epi8(bytes, zero);
      __m128i res_lo = jm_wp_8lanes(lo16, wp_scale_v, round_v, shift_count, offset_v, zero, vmax);
      __m128i res_hi = jm_wp_8lanes(hi16, wp_scale_v, round_v, shift_count, offset_v, zero, vmax);
      __m128i res8   = _mm_packus_epi16(res_lo, res_hi);
      _mm_storeu_si128((__m128i *)(&mb_pred[j][ioff]), res8);
    }
  }
  else
  {
    /* block_size_x in {2, 4, 8}: compute 8 lanes, store low N bytes.
     * Over-reads on block[j] are safe (the MC temp buffer is 16-wide). */
    for (j = 0; j < block_size_y; j++)
    {
      __m128i bytes = _mm_loadl_epi64((const __m128i *)block[j]);
      __m128i pel16 = _mm_unpacklo_epi8(bytes, zero);
      __m128i res16 = jm_wp_8lanes(pel16, wp_scale_v, round_v, shift_count, offset_v, zero, vmax);
      __m128i res8  = _mm_packus_epi16(res16, res16);
      imgpel *dst   = &mb_pred[j][ioff];
      if (block_size_x == 8)
      {
        _mm_storel_epi64((__m128i *)dst, res8);
      }
      else if (block_size_x == 4)
      {
        int tmp = _mm_cvtsi128_si32(res8);
        memcpy(dst, &tmp, 4);
      }
      else  /* block_size_x == 2 */
      {
        int tmp = _mm_cvtsi128_si32(res8);
        memcpy(dst, &tmp, 2);
      }
    }
  }
}

/* ===================================================================== */
/*   Stage 4 -- weighted_bi_prediction: bi-directional weighted pred      */
/* ===================================================================== */
/*  result = ((s0*b0 + s1*b1 + (1 << (denom-1))) >> denom) + offset       */
/*  out[i] = clip(0, color_clip, result)                                  */
/*                                                                        */
/*  Trick: pmaddwd computes pairs of int16 products and sums adjacent     */
/*  pairs into int32 lanes. Pack weights as [s0, s1, s0, s1, ...] and     */
/*  interleave (b0[i], b1[i]) pairs, then one madd gives the per-pixel    */
/*  weighted sum already in int32.                                        */
/*                                                                        */
/*  Caller advances mb_pred / block_l0 / block_l1 by block_size_x per     */
/*  row of work and then by (MB_BLOCK_SIZE - block_size_x) of fixup --    */
/*  the buffers are 16-strided regardless of block_size_x.                */
/* ===================================================================== */

/* Process 8 lanes: out = ((s0*b0 + s1*b1 + round) >> denom) + offset,
 * clipped to [0, vmax]. Returns int16 lanes (low 8 valid). */
static __inline __m128i jm_wbi_8lanes(const imgpel *b0, const imgpel *b1,
                                      __m128i wp_pair, __m128i round_v,
                                      __m128i shift_count, __m128i offset_v,
                                      __m128i zero, __m128i vmax)
{
  __m128i bv0   = _mm_loadl_epi64((const __m128i *)b0);
  __m128i bv1   = _mm_loadl_epi64((const __m128i *)b1);
  __m128i b0_16 = _mm_unpacklo_epi8(bv0, zero);
  __m128i b1_16 = _mm_unpacklo_epi8(bv1, zero);
  /* Interleave: pairs_lo = [b0[0], b1[0], b0[1], b1[1], b0[2], b1[2], b0[3], b1[3]] */
  __m128i pairs_lo = _mm_unpacklo_epi16(b0_16, b1_16);
  __m128i pairs_hi = _mm_unpackhi_epi16(b0_16, b1_16);
  /* madd: 4 int32 each = s0*b0[i] + s1*b1[i] for i in [0..3] and [4..7] */
  __m128i sum_lo32 = _mm_madd_epi16(pairs_lo, wp_pair);
  __m128i sum_hi32 = _mm_madd_epi16(pairs_hi, wp_pair);
  /* +round, >>denom, +offset */
  sum_lo32 = _mm_sra_epi32(_mm_add_epi32(sum_lo32, round_v), shift_count);
  sum_hi32 = _mm_sra_epi32(_mm_add_epi32(sum_hi32, round_v), shift_count);
  sum_lo32 = _mm_add_epi32(sum_lo32, offset_v);
  sum_hi32 = _mm_add_epi32(sum_hi32, offset_v);
  /* Pack to int16, clip to [0, vmax]. */
  __m128i res16 = _mm_packs_epi32(sum_lo32, sum_hi32);
  res16 = _mm_max_epi16(res16, zero);
  return _mm_min_epi16(res16, vmax);
}

void weighted_bi_prediction_sse(imgpel *mb_pred, imgpel *block_l0, imgpel *block_l1,
                                int block_size_y, int block_size_x,
                                int wp_scale_l0, int wp_scale_l1,
                                int wp_offset, int weight_denom, int color_clip)
{
  /* Weights packed as [s0, s1, s0, s1, ...] across the int16x8 register. */
  const __m128i wp_pair = _mm_set1_epi32(
      (int)(((unsigned int)((unsigned short)wp_scale_l1) << 16) |
            (unsigned int)((unsigned short)wp_scale_l0)));
  const __m128i round_v     = _mm_set1_epi32(1 << (weight_denom - 1));
  const __m128i shift_count = _mm_cvtsi32_si128(weight_denom);
  const __m128i offset_v    = _mm_set1_epi32(wp_offset);
  const __m128i zero        = _mm_setzero_si128();
  const __m128i vmax        = _mm_set1_epi16((short)color_clip);
  const int row_inc = MB_BLOCK_SIZE - block_size_x;
  int j;

  assert(sizeof(imgpel) == 1);

  if (block_size_x == 16)
  {
    for (j = 0; j < block_size_y; j++)
    {
      __m128i res_lo = jm_wbi_8lanes(block_l0,     block_l1,
                                     wp_pair, round_v, shift_count, offset_v, zero, vmax);
      __m128i res_hi = jm_wbi_8lanes(block_l0 + 8, block_l1 + 8,
                                     wp_pair, round_v, shift_count, offset_v, zero, vmax);
      __m128i res8   = _mm_packus_epi16(res_lo, res_hi);
      _mm_storeu_si128((__m128i *)mb_pred, res8);
      mb_pred  += 16 + row_inc;   /* block_size_x + row_inc = MB_BLOCK_SIZE */
      block_l0 += 16 + row_inc;
      block_l1 += 16 + row_inc;
    }
  }
  else
  {
    /* block_size_x in {2, 4, 8} */
    for (j = 0; j < block_size_y; j++)
    {
      __m128i res16 = jm_wbi_8lanes(block_l0, block_l1,
                                    wp_pair, round_v, shift_count, offset_v, zero, vmax);
      __m128i res8  = _mm_packus_epi16(res16, res16);
      if (block_size_x == 8)
      {
        _mm_storel_epi64((__m128i *)mb_pred, res8);
      }
      else if (block_size_x == 4)
      {
        int tmp = _mm_cvtsi128_si32(res8);
        memcpy(mb_pred, &tmp, 4);
      }
      else  /* block_size_x == 2 */
      {
        int tmp = _mm_cvtsi128_si32(res8);
        memcpy(mb_pred, &tmp, 2);
      }
      mb_pred  += block_size_x + row_inc;
      block_l0 += block_size_x + row_inc;
      block_l1 += block_size_x + row_inc;
    }
  }
}

#endif  /* IMGTYPE == 0 */
