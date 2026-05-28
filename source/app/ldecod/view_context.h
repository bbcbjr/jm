
/*!
 ***************************************************************************
 * \file
 *    view_context.h
 *
 * \brief
 *    Per-view (per-DPB-layer) decode state. Each MVC view owns one of
 *    these and threaded view-parallel decoding (Stage 2) walks each
 *    worker through its own ViewContext, leaving the shared
 *    VideoParameters with read-mostly state only.
 *
 *    Scaffolding-only at this milestone (M3-scaffold): the struct is
 *    allocated alongside each DPB layer but no fields have been migrated
 *    off VideoParameters yet. Field migration happens group-by-group in
 *    subsequent M3 steps; see STAGE2_M2_DESIGN.md.
 ***************************************************************************
 */

#ifndef _VIEW_CONTEXT_H_
#define _VIEW_CONTEXT_H_

#include "global.h"

typedef struct view_context
{
  // ---- identity ----------------------------------------------------------
  int                       view_id;          //!< 0 = base, 1 = dependent
  int                       layer_id;         //!< currently == view_id
  struct video_par         *p_Vid;            //!< upward pointer to shared VP

  // ---- current picture --------------------------------------------------
  struct storable_picture  *dec_picture;
  struct storable_picture  *dec_picture_JV[MAX_PLANE];

  // ---- slice collection (per-picture) -----------------------------------
  struct slice            **ppSliceList;
  int                       iSliceNumOfCurrPic;
  int                       iNumOfSlicesAllocated;
  int                       iNumOfSlicesDecoded;
  unsigned int              num_dec_mb;

  // ---- lookahead carry-over for next picture ----------------------------
  struct slice                  *pNextSlice;
  pic_parameter_set_rbsp_t      *pNextPPS;
  int                            newframe;
  int                            bFrameInit;

  // ---- cached-shortcut pointers (owning storage is p_EncodePar[layer_id])
  struct macroblock_dec    *mb_data;
  struct macroblock_dec    *mb_data_JV[MAX_PLANE];
  char                     *intra_block;
  char                     *intra_block_JV[MAX_PLANE];
  byte                    **ipredmode;
  byte                    **ipredmode_JV[MAX_PLANE];
  int                     **siblock;
  int                     **siblock_JV[MAX_PLANE];
  byte                  ****nz_coeff;
  BlockPos                 *PicPos;
  int                      *qp_per_matrix;
  int                      *qp_rem_matrix;
  imgpel                  **imgY_ref;
  imgpel                 ***imgUV_ref;

  // ---- current picture dimensions (refreshed in init_picture) ----------
  unsigned int              PicHeightInMbs;
  unsigned int              PicSizeInMbs;
  unsigned int              PicWidthInMbs;
  unsigned int              FrameHeightInMbs;
  unsigned int              FrameSizeInMbs;
  unsigned int              oldFrameSizeInMbs;

  // ---- current picture metadata ----------------------------------------
  PictureStructure          structure;
  int                       type;             //!< I / P / B / SI / SP

  // ---- per-view frame_num & DPB continuity -----------------------------
  unsigned int              pre_frame_num;
  unsigned int              previous_frame_num;
  int                       last_dec_poc;
  int                       last_dec_layer_id;
  int                       dpb_layer_id;
  int                       last_has_mmco_5;
  int                       last_pic_bottom_field;

  // ---- POC mode 0 state ------------------------------------------------
  signed int                PrevPicOrderCntMsb;
  unsigned int              PrevPicOrderCntLsb;

  // ---- POC mode 1 state ------------------------------------------------
  signed int                ExpectedPicOrderCnt;
  signed int                PicOrderCntCycleCnt;
  signed int                FrameNumInPicOrderCntCycle;
  unsigned int              PreviousFrameNum;
  unsigned int              FrameNumOffset;
  int                       ExpectedDeltaPerPicOrderCntCycle;
  int                       ThisPOC;
  int                       PreviousFrameNumOffset;

  // ---- per-slice snapshots of active parameter sets --------------------
  // refreshed at the top of every slice's processing by snapshot_active_ps()
  seq_parameter_set_rbsp_t        *active_sps;
  pic_parameter_set_rbsp_t        *active_pps;
  subset_seq_parameter_set_rbsp_t *active_subset_sps;
  int                              ChromaArrayType;
  int                              no_output_of_prior_pics_flag;

  // ---- per-view reporting strings & boundary detection -----------------
  char                      cslice_type[9];
  struct old_slice_par     *old_slice;
  int                       non_conforming_stream;

  // ---- per-view random-access / recovery state -------------------------
  int                       recovery_point;
  int                       recovery_point_found;
  int                       recovery_frame_num;
  int                       recovery_frame_cnt;
  int                       recovery_poc;
  int                       recovery_flag;

  // ---- per-view redundant-slice state ----------------------------------
  int                       Is_primary_correct;
  int                       Is_redundant_correct;

  // ---- per-view error concealment scratch (ERC currently disabled) -----
  struct concealment_node  *concealment_head;
  struct concealment_node  *concealment_end;
  int                       IDR_concealment_flag;
  int                       conceal_mode;
  int                       conceal_slice_type;
  int                       earlier_missing_poc;
  unsigned int              frame_to_conceal;
  int                       last_ref_pic_poc;
  int                       ref_poc_gap;
  int                       poc_gap;

  // ---- output staging --------------------------------------------------
  ImageData                 tempData3;

  // ---- threading (slot reserved; populated in M4) ----------------------
  // pthread_t              worker_tid;
  // queue_t               *inbox_nalus;     // NALUs for this view
  // queue_t               *outbox_pictures; // completed pictures, POC-tagged
  // pthread_mutex_t        dpb_mutex;       // for inter-view sync wait
  // pthread_cond_t         dpb_cond;
  // int                    last_published_poc;

} ViewContext;

/*!
 ************************************************************************
 * \brief
 *    VCTX(slice)    : the ViewContext that owns currSlice's per-view state
 *    VCTX_MB(mb)    : same, addressed via the macroblock's parent slice
 *
 *    Most call-sites already have one of these pointers in scope, so
 *    field migrations are a textual p_Vid->FIELD --> VCTX(...)->FIELD
 *    rewrite. The very few entry-point functions that have neither in
 *    scope take an explicit (ViewContext *vctx) parameter.
 ************************************************************************
 */
#define VCTX(slice)    ((slice)->p_Vid->p_view_ctx[(slice)->layer_id])
#define VCTX_MB(mb)    VCTX((mb)->p_Slice)
/* For DPB-bound helpers (ERC, store/flush) that don't carry a Slice: each
   DecodedPictureBuffer carries layer_id + a p_Vid back-pointer, so the same
   indirection works. */
#define VCTX_DPB(dpb)  ((dpb)->p_Vid->p_view_ctx[(dpb)->layer_id])

/*!
 ************************************************************************
 * \brief
 *    Allocate a per-view decode context for the given layer. Called
 *    once per layer at decoder init, alongside DPB / EncodePar / LayerPar.
 ************************************************************************
 */
extern struct view_context *alloc_view_context (struct video_par *p_Vid, int layer_id);

/*!
 ************************************************************************
 * \brief
 *    Tear down a per-view decode context. Called once per layer at
 *    decoder shutdown, alongside DPB / EncodePar / LayerPar free.
 ************************************************************************
 */
extern void                 free_view_context  (struct view_context *vctx);

#endif  // _VIEW_CONTEXT_H_
