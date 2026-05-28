
/*!
 ***************************************************************************
 * \file
 *    jm_simd.c
 *
 * \brief
 *    Stage 3 SIMD dispatch table + CPU feature detection. See jm_simd.h
 *    for the design and usage rules.
 ***************************************************************************
 */

#include <assert.h>
#include <stdio.h>
#include <string.h>

#include "global.h"
#include "jm_simd.h"
#include "transform_sse.h"
#include "transform_avx2.h"
#include "mc_prediction.h"  /* for scalar MC kernel extern decls (Stage 3c-0) */

/* Inverse transform kernel declarations come from transform.h:
 *   scalar:  inverse4x4 / inverse8x8        (transform.c)
 *   SIMD:    inverse4x4_sse                 (transform_simd.c)
 *            inverse8x8_sse2 / inverse8x8_avx2
 *
 * IMGTYPE-agnostic (operate on int** coefficient arrays, not pixels). */
#include "transform.h"

/* SIMD MC kernels (mc_kernels_sse.c). Stage 3c-1+ add these one at a time.
 * Only built for IMGTYPE == 0 (8-bit imgpel); the kernels use byte-vs-short
 * intrinsics and int16 intermediates that don't generalize to 10/12-bit. */
#if IMGTYPE == 0
extern void get_luma_20_sse(imgpel **block, imgpel **cur_imgY, int block_size_y,
                            int block_size_x, int x_pos, int max_imgpel_value);
extern void get_luma_02_sse(imgpel **block, imgpel **cur_imgY, int block_size_y,
                            int block_size_x, int x_pos, int shift_x, int max_imgpel_value);
extern void get_luma_10_sse(imgpel **block, imgpel **cur_imgY, int block_size_y,
                            int block_size_x, int x_pos, int max_imgpel_value);
extern void get_luma_30_sse(imgpel **block, imgpel **cur_imgY, int block_size_y,
                            int block_size_x, int x_pos, int max_imgpel_value);
extern void get_luma_01_sse(imgpel **block, imgpel **cur_imgY, int block_size_y,
                            int block_size_x, int x_pos, int shift_x, int max_imgpel_value);
extern void get_luma_03_sse(imgpel **block, imgpel **cur_imgY, int block_size_y,
                            int block_size_x, int x_pos, int shift_x, int max_imgpel_value);
extern void get_luma_22_sse(imgpel **block, imgpel **cur_imgY, int **tmp_res,
                            int block_size_y, int block_size_x, int x_pos,
                            int max_imgpel_value);
extern void get_luma_21_sse(imgpel **block, imgpel **cur_imgY, int **tmp_res,
                            int block_size_y, int block_size_x, int x_pos,
                            int max_imgpel_value);
extern void get_luma_23_sse(imgpel **block, imgpel **cur_imgY, int **tmp_res,
                            int block_size_y, int block_size_x, int x_pos,
                            int max_imgpel_value);
extern void get_luma_12_sse(imgpel **block, imgpel **cur_imgY, int **tmp_res,
                            int block_size_y, int block_size_x, int x_pos,
                            int shift_x, int max_imgpel_value);
extern void get_luma_32_sse(imgpel **block, imgpel **cur_imgY, int **tmp_res,
                            int block_size_y, int block_size_x, int x_pos,
                            int shift_x, int max_imgpel_value);
extern void get_luma_11_sse(imgpel **block, imgpel **cur_imgY, int block_size_y,
                            int block_size_x, int x_pos, int shift_x, int max_imgpel_value);
extern void get_luma_13_sse(imgpel **block, imgpel **cur_imgY, int block_size_y,
                            int block_size_x, int x_pos, int shift_x, int max_imgpel_value);
extern void get_luma_31_sse(imgpel **block, imgpel **cur_imgY, int block_size_y,
                            int block_size_x, int x_pos, int shift_x, int max_imgpel_value);
extern void get_luma_33_sse(imgpel **block, imgpel **cur_imgY, int block_size_y,
                            int block_size_x, int x_pos, int shift_x, int max_imgpel_value);
extern void get_chroma_0X_sse(imgpel *block, imgpel *cur_img, int span,
                              int block_size_y, int block_size_x,
                              int w00, int w01, int total_scale);
extern void get_chroma_X0_sse(imgpel *block, imgpel *cur_img, int span,
                              int block_size_y, int block_size_x,
                              int w00, int w10, int total_scale);
extern void get_chroma_XY_sse(imgpel *block, imgpel *cur_img, int span,
                              int block_size_y, int block_size_x,
                              int w00, int w01, int w10, int w11, int total_scale);
#endif

/* The single global dispatch table. Read-only after jm_simd_init(). */
jm_simd_dispatch_t jm_simd;

/* ===================================================================== */
/*                    CPU FEATURE DETECTION (x86/x64)                     */
/* ===================================================================== */

#if defined(_MSC_VER) && (defined(_M_IX86) || defined(_M_X64))
  #include <intrin.h>
  #define JM_HAVE_CPUID 1
  static void jm__cpuid(int leaf, int sub, int regs[4])
  {
    __cpuidex(regs, leaf, sub);
  }
#elif defined(__GNUC__) && (defined(__i386__) || defined(__x86_64__))
  #include <cpuid.h>
  #define JM_HAVE_CPUID 1
  static void jm__cpuid(int leaf, int sub, int regs[4])
  {
    __cpuid_count(leaf, sub,
                  regs[0], regs[1], regs[2], regs[3]);
  }
#else
  #define JM_HAVE_CPUID 0
#endif

unsigned int jm_cpu_detect(void)
{
  unsigned int features = 0;

#if JM_HAVE_CPUID
  int regs[4] = {0, 0, 0, 0};

  /* leaf 1: standard feature flags (EDX/ECX) */
  jm__cpuid(1, 0, regs);
  /* EDX bit 26: SSE2 */
  if (regs[3] & (1 << 26)) features |= JM_CPU_SSE2;
  /* ECX bit 0:  SSE3   */
  if (regs[2] & (1 << 0))  features |= JM_CPU_SSE3;
  /* ECX bit 9:  SSSE3  */
  if (regs[2] & (1 << 9))  features |= JM_CPU_SSSE3;
  /* ECX bit 19: SSE4.1 */
  if (regs[2] & (1 << 19)) features |= JM_CPU_SSE41;
  /* ECX bit 20: SSE4.2 */
  if (regs[2] & (1 << 20)) features |= JM_CPU_SSE42;
  /* ECX bit 28: AVX (must also check OSXSAVE bit 27 + XCR0 to confirm
   * OS state-save support; we skip that for now -- any modern OS supports
   * AVX state save if the CPU has it) */
  if (regs[2] & (1 << 28)) features |= JM_CPU_AVX;

  /* leaf 7 sub-leaf 0: extended feature flags (EBX) */
  jm__cpuid(7, 0, regs);
  /* EBX bit 5: AVX2 */
  if (regs[1] & (1 << 5))  features |= JM_CPU_AVX2;
#endif

  return features;
}

/* ===================================================================== */
/*                        DISPATCH TABLE INIT                             */
/* ===================================================================== */

void jm_simd_init(void)
{
  unsigned int feat = jm_cpu_detect();
  unsigned int selected = 0;

  /* ===== Inverse transforms ===== */
  jm_simd.inverse4x4 = inverse4x4;
  jm_simd.inverse8x8 = inverse8x8;

  /* Stage 3b: select SIMD variants when supported. Selection order goes
   * from highest-quality to lowest, with the last applicable assignment
   * winning. This way a CPU with AVX2 gets the AVX2 8x8, while one with
   * only SSE2 still gets the SSE2 8x8. */
  if (feat & JM_CPU_SSE2)
  {
    jm_simd.inverse4x4 = inverse4x4_sse;     /* SSE2 4x4 IDCT */
    jm_simd.inverse8x8 = inverse8x8_sse2;    /* SSE2 8x8 IDCT */
    selected |= JM_CPU_SSE2;
  }
  if (feat & JM_CPU_AVX2)
  {
    jm_simd.inverse8x8 = inverse8x8_avx2;    /* AVX2 8x8 IDCT (upgrade) */
    selected |= JM_CPU_AVX2;
  }

  /* ===== MC luma sub-pel kernels (Stage 3c-0: all scalar for now) =====
   * Function pointers populated from the scalar implementations declared
   * in mc_prediction.h. SIMD replacements drop in as Stage 3c sub-phases. */
  jm_simd.get_block_00 = get_block_00;
  jm_simd.get_luma_10  = get_luma_10;
  jm_simd.get_luma_20  = get_luma_20;
  jm_simd.get_luma_30  = get_luma_30;
  jm_simd.get_luma_01  = get_luma_01;
  jm_simd.get_luma_02  = get_luma_02;
  jm_simd.get_luma_03  = get_luma_03;
  jm_simd.get_luma_21  = get_luma_21;
  jm_simd.get_luma_22  = get_luma_22;
  jm_simd.get_luma_23  = get_luma_23;
  jm_simd.get_luma_12  = get_luma_12;
  jm_simd.get_luma_32  = get_luma_32;
  jm_simd.get_luma_11  = get_luma_11;
  jm_simd.get_luma_13  = get_luma_13;
  jm_simd.get_luma_31  = get_luma_31;
  jm_simd.get_luma_33  = get_luma_33;

  /* ===== MC chroma kernels ===== */
  jm_simd.get_chroma_0X = get_chroma_0X;
  jm_simd.get_chroma_X0 = get_chroma_X0;
  jm_simd.get_chroma_XY = get_chroma_XY;

  /* Stage 3c-1 / 3c-2: SSSE3 half-pel luma filters. Universally
   * available on x86-64 (SSSE3 introduced 2006). The MC SIMD kernels
   * only exist for 8-bit imgpel builds; high-bit-depth builds fall
   * back to scalar transparently. */
#if IMGTYPE == 0
  if (feat & JM_CPU_SSSE3)
  {
    /* Belt-and-braces: confirm at runtime that this TU's imgpel is the
     * 1-byte form the SIMD kernels assume. If a future build mismatches
     * IMGTYPE across TUs, the static_assert in mc_kernels_sse.c is the
     * first line of defense; this assert is the second. */
    assert(sizeof(imgpel) == 1);
    jm_simd.get_luma_20 = get_luma_20_sse;   /* horizontal half-pel */
    jm_simd.get_luma_02 = get_luma_02_sse;   /* vertical   half-pel */
    jm_simd.get_luma_10 = get_luma_10_sse;   /* horizontal qpel  (dx=1) */
    jm_simd.get_luma_30 = get_luma_30_sse;   /* horizontal qpel  (dx=3) */
    jm_simd.get_luma_01 = get_luma_01_sse;   /* vertical   qpel  (dy=1) */
    jm_simd.get_luma_03 = get_luma_03_sse;   /* vertical   qpel  (dy=3) */
    jm_simd.get_luma_22 = get_luma_22_sse;   /* diagonal half-pel (2D, tmp_res) */
    jm_simd.get_luma_21 = get_luma_21_sse;   /* X2 family: diag-hpel + h-hpel avg */
    jm_simd.get_luma_23 = get_luma_23_sse;   /* X2 family: diag-hpel + h-hpel avg */
    jm_simd.get_luma_12 = get_luma_12_sse;   /* 2Y family: diag-hpel + v-hpel avg */
    jm_simd.get_luma_32 = get_luma_32_sse;   /* 2Y family: diag-hpel + v-hpel avg */
    jm_simd.get_luma_11 = get_luma_11_sse;   /* diagonal qpel (1,1) */
    jm_simd.get_luma_13 = get_luma_13_sse;   /* diagonal qpel (1,3) */
    jm_simd.get_luma_31 = get_luma_31_sse;   /* diagonal qpel (3,1) */
    jm_simd.get_luma_33 = get_luma_33_sse;   /* diagonal qpel (3,3) */
    jm_simd.get_chroma_0X = get_chroma_0X_sse;  /* Y-axis chroma bilinear */
    jm_simd.get_chroma_X0 = get_chroma_X0_sse;  /* X-axis chroma bilinear */
    jm_simd.get_chroma_XY = get_chroma_XY_sse;  /* 2D chroma bilinear (4 corners) */
    selected |= JM_CPU_SSSE3;
  }
#endif

  jm_simd.features_detected = feat;
  jm_simd.features_selected = selected;
}

void jm_simd_print_info(void)
{
  unsigned int f = jm_simd.features_detected;
  printf("SIMD: detected:");
  if (f == 0)               printf(" (none / non-x86 build)");
  if (f & JM_CPU_SSE2)      printf(" SSE2");
  if (f & JM_CPU_SSE3)      printf(" SSE3");
  if (f & JM_CPU_SSSE3)     printf(" SSSE3");
  if (f & JM_CPU_SSE41)     printf(" SSE4.1");
  if (f & JM_CPU_SSE42)     printf(" SSE4.2");
  if (f & JM_CPU_AVX)       printf(" AVX");
  if (f & JM_CPU_AVX2)      printf(" AVX2");
  printf("  selected:");
  if (jm_simd.features_selected == 0)
    printf(" scalar-only");
  else
  {
    if (jm_simd.features_selected & JM_CPU_SSE2)  printf(" SSE2(inverse4x4,inverse8x8)");
    if (jm_simd.features_selected & JM_CPU_AVX2)  printf(" AVX2(inverse8x8)");
#if IMGTYPE == 0
    if (jm_simd.features_selected & JM_CPU_SSSE3) printf(" SSSE3(ALL_LUMA_MC,ALL_CHROMA_MC)");
#endif
  }
  printf("\n");
}
