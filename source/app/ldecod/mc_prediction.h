
/*!
 *************************************************************************************
 * \file mc_prediction.h
 *
 * \brief
 *    definitions for motion compensated prediction
 *
 * \author
 *      Main contributors (see contributors.h for copyright, 
 *                         address and affiliation details)
 *      - Alexis Michael Tourapis  <alexismt@ieee.org>
 *
 *************************************************************************************
 */

#ifndef _MC_PREDICTION_H_
#define _MC_PREDICTION_H_

#include "global.h"
#include "mbuffer.h"

extern int  allocate_pred_mem(Slice *currSlice);
extern void free_pred_mem    (Slice *currSlice);

extern void get_block_luma(StorablePicture *curr_ref, int x_pos, int y_pos, int block_size_x, int block_size_y, imgpel **block,
                           int shift_x,int maxold_x,int maxold_y,int **tmp_res,int max_imgpel_value,imgpel no_ref_value,Macroblock *currMB);

extern void intra_cr_decoding    (Macroblock *currMB, int yuv);
extern void prepare_direct_params(Macroblock *currMB, StorablePicture *dec_picture, MotionVector *pmvl0, MotionVector *pmvl1,char *l0_rFrame, char *l1_rFrame);
extern void perform_mc           (Macroblock *currMB, ColorPlane pl, StorablePicture *dec_picture, int pred_dir, int i, int j, int block_size_x, int block_size_y);

/* Scalar MC kernel declarations (formerly static; now
 * externally addressable so they can be installed into the jm_simd
 * dispatch table). SIMD reimplementations declared the same way. */
extern void get_block_00 (imgpel *block, imgpel *cur_img, int span, int block_size_y);
extern void get_luma_10  (imgpel **block, imgpel **cur_imgY, int block_size_y, int block_size_x, int x_pos, int max_imgpel_value);
extern void get_luma_20  (imgpel **block, imgpel **cur_imgY, int block_size_y, int block_size_x, int x_pos, int max_imgpel_value);
extern void get_luma_30  (imgpel **block, imgpel **cur_imgY, int block_size_y, int block_size_x, int x_pos, int max_imgpel_value);
extern void get_luma_01  (imgpel **block, imgpel **cur_imgY, int block_size_y, int block_size_x, int x_pos, int shift_x, int max_imgpel_value);
extern void get_luma_02  (imgpel **block, imgpel **cur_imgY, int block_size_y, int block_size_x, int x_pos, int shift_x, int max_imgpel_value);
extern void get_luma_03  (imgpel **block, imgpel **cur_imgY, int block_size_y, int block_size_x, int x_pos, int shift_x, int max_imgpel_value);
extern void get_luma_21  (imgpel **block, imgpel **cur_imgY, int **tmp_res, int block_size_y, int block_size_x, int x_pos, int max_imgpel_value);
extern void get_luma_22  (imgpel **block, imgpel **cur_imgY, int **tmp_res, int block_size_y, int block_size_x, int x_pos, int max_imgpel_value);
extern void get_luma_23  (imgpel **block, imgpel **cur_imgY, int **tmp_res, int block_size_y, int block_size_x, int x_pos, int max_imgpel_value);
extern void get_luma_12  (imgpel **block, imgpel **cur_imgY, int **tmp_res, int block_size_y, int block_size_x, int x_pos, int shift_x, int max_imgpel_value);
extern void get_luma_32  (imgpel **block, imgpel **cur_imgY, int **tmp_res, int block_size_y, int block_size_x, int x_pos, int shift_x, int max_imgpel_value);
extern void get_luma_11  (imgpel **block, imgpel **cur_imgY, int block_size_y, int block_size_x, int x_pos, int shift_x, int max_imgpel_value);
extern void get_luma_13  (imgpel **block, imgpel **cur_imgY, int block_size_y, int block_size_x, int x_pos, int shift_x, int max_imgpel_value);
extern void get_luma_31  (imgpel **block, imgpel **cur_imgY, int block_size_y, int block_size_x, int x_pos, int shift_x, int max_imgpel_value);
extern void get_luma_33  (imgpel **block, imgpel **cur_imgY, int block_size_y, int block_size_x, int x_pos, int shift_x, int max_imgpel_value);

extern void get_chroma_0X(imgpel *block, imgpel *cur_img, int span, int block_size_y, int block_size_x, int w00, int w01, int total_scale);
extern void get_chroma_X0(imgpel *block, imgpel *cur_img, int span, int block_size_y, int block_size_x, int w00, int w10, int total_scale);
extern void get_chroma_XY(imgpel *block, imgpel *cur_img, int span, int block_size_y, int block_size_x, int w00, int w01, int w10, int w11, int total_scale);

/* Scalar fallbacks for the residual reconstruction and weighted
 * prediction kernels. recon8x8 lives in transform8x8.c; the two weighted
 * prediction helpers live in mc_prediction.c. All three were `static`
 */
extern void recon8x8(int **m7, imgpel **mb_rec, imgpel **mpr, int max_imgpel_value, int ioff);
extern void weighted_mc_prediction(imgpel **mb_pred, imgpel **block, int block_size_y, int block_size_x, int ioff, int wp_scale, int wp_offset, int weight_denom, int color_clip);
extern void weighted_bi_prediction(imgpel *mb_pred, imgpel *block_l0, imgpel *block_l1, int block_size_y, int block_size_x, int wp_scale_l0, int wp_scale_l1, int wp_offset, int weight_denom, int color_clip);

#endif

