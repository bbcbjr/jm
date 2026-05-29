
/*!
 ***************************************************************************
 * \file
 *    jm_simd.h
 *
 * \brief
 *    Stage 3 SIMD dispatch infrastructure for ldecod.
 *
 *    Design:
 *      - Runtime CPU feature detection (CPUID on x86).
 *      - A single global dispatch table holds function pointers to the
 *        "best available" implementation for each kernel.
 *      - jm_simd_init() probes the CPU and populates the table once at
 *        decoder startup. Default fills are scalar implementations, so
 *        a build with no SIMD kernels behaves identically to baseline.
 *      - Call sites use jm_simd.kernel_name(...) instead of calling the
 *        scalar function directly. The indirect call costs one mispredict
 *        on first use and zero thereafter (target stays stable for the
 *        lifetime of the decoder).
 *
 *    Verification harness (Stage 3 dev-time only):
 *      - When JM_SIMD_VERIFY is defined at build time, every dispatch
 *        call goes through a wrapper that runs BOTH the SIMD and scalar
 *        implementations and compares outputs byte-for-byte. Any mismatch
 *        triggers an assertion. Off by default; expensive.
 *
 *    Threading: jm_simd is read-only after jm_simd_init(). Safe to call
 *    from any number of worker threads concurrently.
 ***************************************************************************
 */

#ifndef _JM_SIMD_H_
#define _JM_SIMD_H_

#include "global.h"   /* for imgpel and shared types */

/*!
 ************************************************************************
 *               CPU FEATURE FLAGS
 *
 *  x86/x64 SIMD ISA flags returned by jm_cpu_detect().
 *  ARM NEON / Apple Silicon support is a future extension.
 ************************************************************************
 */
enum {
  JM_CPU_SSE2   = 1u << 0,   /* baseline for x86-64 */
  JM_CPU_SSE3   = 1u << 1,
  JM_CPU_SSSE3  = 1u << 2,
  JM_CPU_SSE41  = 1u << 3,
  JM_CPU_SSE42  = 1u << 4,
  JM_CPU_AVX    = 1u << 5,
  JM_CPU_AVX2   = 1u << 6
};

/*!
 ************************************************************************
 *               DISPATCH TABLE
 *
 *  Function pointers for each kernel. Populated by jm_simd_init().
 *  Add new kernels here as Stage 3b/3c land.
 *
 *  Signatures match the existing scalar declarations exactly so call
 *  sites can be rewritten with a textual `inverse4x4(` -> `jm_simd.inverse4x4(`
 *  swap.
 ************************************************************************
 */
/*!
 ************************************************************************
 *   MC kernel signature families
 *
 *   The 16 luma sub-pel kernels share 4 distinct signatures, the 3
 *   chroma kernels share 2 more. Typedefs make the dispatch table
 *   readable and let SIMD implementations match the scalar prototypes
 *   exactly.
 *
 *   Position naming: (dx, dy) where dx/dy in {0,1,2,3} are quarter-pel
 *   offsets. "X0" / "0X" denote families with one zero offset; "X2" /
 *   "2Y" denote families that need the tmp_res intermediate.
 ************************************************************************
 */
typedef void (*mc_luma_00_fn) (imgpel *block, imgpel *cur_img, int span, int block_size_y);
typedef void (*mc_luma_x0_fn) (imgpel **block, imgpel **cur_imgY, int block_size_y, int block_size_x, int x_pos, int max_imgpel_value);
typedef void (*mc_luma_0x_fn) (imgpel **block, imgpel **cur_imgY, int block_size_y, int block_size_x, int x_pos, int shift_x, int max_imgpel_value);
typedef void (*mc_luma_x2_fn) (imgpel **block, imgpel **cur_imgY, int **tmp_res, int block_size_y, int block_size_x, int x_pos, int max_imgpel_value);
typedef void (*mc_luma_2y_fn) (imgpel **block, imgpel **cur_imgY, int **tmp_res, int block_size_y, int block_size_x, int x_pos, int shift_x, int max_imgpel_value);

typedef void (*mc_chroma_0x_fn)(imgpel *block, imgpel *cur_img, int span, int block_size_y, int block_size_x, int w00, int w01_or_w10, int total_scale);
typedef void (*mc_chroma_xy_fn)(imgpel *block, imgpel *cur_img, int span, int block_size_y, int block_size_x, int w00, int w01, int w10, int w11, int total_scale);

/*!
 ************************************************************************
 *   dispatch signatures
 *
 *   recon8x8: residual + prediction add + clip for a single 8x8 block
 *             (source/app/ldecod/transform8x8.c)
 *   weighted_mc_prediction: per-pixel (wp_scale*pel + round) >> denom +
 *             offset, then clip. Single reference (P slice or B-slice
 *             single-direction). (source/app/ldecod/mc_prediction.c)
 *   weighted_bi_prediction: bi-directional weighted prediction --
 *             (s0*b0 + s1*b1 + round) >> denom + offset, clip. Used in
 *             B slices when weighted_bipred_idc is set.
 *             (source/app/ldecod/mc_prediction.c)
 ************************************************************************
 */
typedef void (*recon8x8_fn)(int **m7, imgpel **mb_rec, imgpel **mpr, int max_imgpel_value, int ioff);
typedef void (*weighted_mc_pred_fn)(imgpel **mb_pred, imgpel **block, int block_size_y, int block_size_x, int ioff, int wp_scale, int wp_offset, int weight_denom, int color_clip);
typedef void (*weighted_bi_pred_fn)(imgpel *mb_pred, imgpel *block_l0, imgpel *block_l1, int block_size_y, int block_size_x, int wp_scale_l0, int wp_scale_l1, int wp_offset, int weight_denom, int color_clip);

/* sample_reconstruct: same shape as recon8x8 but width/height/dq_bits are
 * runtime parameters. Lives in source/lib/lcommon/blk_prediction.c.
 * Called from block.c at the 4x4, 16x16, and chroma 8x8 sites. */
typedef void (*sample_reconstruct_fn)(imgpel **curImg, imgpel **mpr, int **mb_rres,
                                      int mb_x, int opix_x, int width, int height,
                                      int max_imgpel_value, int dq_bits);

typedef struct {
  /* Inverse transforms (source/lib/lcommon/transform.c) */
  void (*inverse4x4)(int **tblock, int **block, int pos_y, int pos_x);
  void (*inverse8x8)(int **tblock, int **block, int pos_x);

  /* MC luma sub-pel kernels (source/app/ldecod/mc_prediction.c).
   * 16 positions arranged in the standard (dx, dy) quarter-pel grid. */
  mc_luma_00_fn  get_block_00;   /* (0,0) integer pel - just block copy */
  mc_luma_x0_fn  get_luma_10;    /* (1,0) horizontal qpel */
  mc_luma_x0_fn  get_luma_20;    /* (2,0) horizontal hpel */
  mc_luma_x0_fn  get_luma_30;    /* (3,0) horizontal qpel */
  mc_luma_0x_fn  get_luma_01;    /* (0,1) vertical qpel */
  mc_luma_0x_fn  get_luma_02;    /* (0,2) vertical hpel */
  mc_luma_0x_fn  get_luma_03;    /* (0,3) vertical qpel */
  mc_luma_x2_fn  get_luma_21;    /* (2,1) */
  mc_luma_x2_fn  get_luma_22;    /* (2,2) diagonal hpel */
  mc_luma_x2_fn  get_luma_23;    /* (2,3) */
  mc_luma_2y_fn  get_luma_12;    /* (1,2) */
  mc_luma_2y_fn  get_luma_32;    /* (3,2) */
  mc_luma_0x_fn  get_luma_11;    /* (1,1) */
  mc_luma_0x_fn  get_luma_13;    /* (1,3) */
  mc_luma_0x_fn  get_luma_31;    /* (3,1) */
  mc_luma_0x_fn  get_luma_33;    /* (3,3) */

  /* MC chroma kernels */
  mc_chroma_0x_fn get_chroma_0X;
  mc_chroma_0x_fn get_chroma_X0;
  mc_chroma_xy_fn get_chroma_XY;

  /* Residual reconstruction and weighted prediction */
  recon8x8_fn           recon8x8;
  sample_reconstruct_fn sample_reconstruct;
  weighted_mc_pred_fn   weighted_mc_prediction;
  weighted_bi_pred_fn   weighted_bi_prediction;

  /* Bookkeeping: which features the selected implementations require */
  unsigned int features_selected;
  /* All CPU features actually detected (may exceed features_selected if
   * we have no SIMD impl yet for some advanced ISA) */
  unsigned int features_detected;
} jm_simd_dispatch_t;

extern jm_simd_dispatch_t jm_simd;

/*!
 ************************************************************************
 * \brief Detect available CPU features. Returns a bitmask of JM_CPU_*.
 *        Pure function; safe to call any number of times.
 ************************************************************************
 */
extern unsigned int jm_cpu_detect(void);

/*!
 ************************************************************************
 * \brief Initialize the dispatch table. Called once at decoder startup
 *        from OpenDecoder(). Safe to call multiple times (idempotent).
 *
 *        After this call, every jm_simd.kernel pointer is non-NULL and
 *        points to the best implementation for the detected CPU.
 ************************************************************************
 */
extern void jm_simd_init(void);

/*!
 ************************************************************************
 * \brief Print a one-line summary of selected vs available SIMD features
 *        to stdout. Useful for debug / -v decoder runs.
 ************************************************************************
 */
extern void jm_simd_print_info(void);

#endif  /* _JM_SIMD_H_ */
