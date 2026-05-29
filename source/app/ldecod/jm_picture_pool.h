
/*!
 ***************************************************************************
 * \file
 *    jm_picture_pool.h
 *
 * \brief
 *    Per-decoder buffer pool that recycles the heavy buffers attached to
 *    each StorablePicture (imgY, imgUV, mv_info, mb_field). The
 *    StorablePicture struct itself (~1 KB) is still calloc/free'd per
 *    frame -- it's small and its state needs fresh init. Only the big
 *    buffers (~4 MB per 1080p picture) participate in the pool.
 *
 * \par Why
 *    Post Stage 3c, alloc_storable_picture is ~18% of decode time. The
 *    cost is in the OS allocator's malloc/free path for the multi-MB
 *    pixel buffers, not the zero-fill (Phase A confirmed: nozero variants
 *    delivered negligible gain because the malloc itself dominated). A
 *    pool eliminates the malloc/free calls entirely by recycling
 *    pre-allocated buffer sets across frames.
 *
 * \par Scope
 *    The pool serves only FRAME-structured pictures at a fixed canonical
 *    size (set lazily by the first acquire). Anything else (fields,
 *    separate_colour_plane / JV mode, size mismatch) falls back to the
 *    direct alloc path. For Blu-ray MVC this covers ~100% of allocations
 *    because Blu-ray is frame-mode 4:2:0 only.
 *
 * \par Lifecycle
 *    picture_buffer_pool_create  - called from OpenDecoder.
 *    picture_buffer_pool_destroy - called from FinitDecoder, frees ALL
 *                                  slots (caller must ensure no live
 *                                  pointers remain).
 *    try_acquire / release       - per-picture from alloc_storable_picture /
 *                                  free_storable_picture.
 *
 * \par Threading
 *    All public functions take the pool's internal mutex. Current decoder
 *    has only one thread that calls alloc/free (the decode thread); the
 *    demux thread does not touch pictures. Pool is thread-safe by
 *    construction so future view-parallel (M4-P3) can use it unchanged.
 ***************************************************************************
 */

#ifndef _JM_PICTURE_POOL_H_
#define _JM_PICTURE_POOL_H_

#include "global.h"
#include "jm_threads.h"

#define JM_PICTURE_POOL_DEFAULT_CAPACITY 32

struct picture_buffer_pool;   /* forward */

/*! Per-slot record. The buffers are owned by the slot (allocated once
 *  on first acquire, reused thereafter). The slot is not aware of the
 *  StorablePicture struct that borrowed its buffers -- StorablePicture
 *  stores a back-pointer to the slot in its _buffer_slot field. */
typedef struct picture_buffer_slot
{
  /* Heavy buffers (the whole point of the pool). */
  imgpel                       **imgY;        /* NULL until first alloc */
  imgpel                      ***imgUV;       /* NULL if YUV400          */
  struct pic_motion_params     **mv_info;
  byte                          *mb_field;    /* motion.mb_field         */

  /* Size & layout this slot was allocated for. Set on first alloc. */
  int size_x, size_y, size_x_cr, size_y_cr;
  int iLumaStride,   iLumaExpandedHeight;
  int iChromaStride, iChromaExpandedHeight;
  int iLumaPadY, iLumaPadX;
  int iChromaPadY, iChromaPadX;
  int chroma_format_idc;

  int                            in_use;       /* 0=free, 1=checked out  */
  struct picture_buffer_pool    *parent_pool;  /* back-pointer for release */
} PictureBufferSlot;

typedef struct picture_buffer_pool
{
  PictureBufferSlot **slots;       /* slots[capacity], lazily filled */
  int                 capacity;
  int                 count;       /* number of slots created so far */

  /* Canonical match key (set by first successful acquire). */
  int canonical_set;
  int canonical_size_x, canonical_size_y;
  int canonical_size_x_cr, canonical_size_y_cr;
  int canonical_chroma_format_idc;
  int canonical_iLumaPadY, canonical_iLumaPadX;
  int canonical_iChromaPadY, canonical_iChromaPadX;

  jm_mutex_t          lock;

  /* Stats (printed at decoder shutdown). */
  unsigned int        stat_hits;       /* reused an existing slot */
  unsigned int        stat_creates;    /* allocated a new slot (first time) */
  unsigned int        stat_misses;     /* fell back to direct alloc */
} PictureBufferPool;

/*! Create an empty pool. Slot buffers are allocated lazily on first use.
 *  capacity <= 0 means "use JM_PICTURE_POOL_DEFAULT_CAPACITY". */
extern PictureBufferPool *picture_buffer_pool_create(int capacity);

/*! Free ALL slots and the pool. Caller MUST ensure no StorablePicture
 *  still references a slot's buffers. */
extern void picture_buffer_pool_destroy(PictureBufferPool *pool);

/*! Try to acquire a slot matching the requested size/format. Returns the
 *  slot (caller should copy slot->imgY/imgUV/mv_info/mb_field into the
 *  StorablePicture's corresponding fields and set _buffer_slot to the
 *  returned pointer) or NULL on:
 *    - structure != FRAME (fields not pooled)
 *    - separate_colour_plane_flag != 0 (JV mode not pooled)
 *    - size/padding mismatch against canonical
 *    - pool is full
 */
extern PictureBufferSlot *picture_buffer_pool_try_acquire(
    PictureBufferPool *pool, VideoParameters *p_Vid,
    PictureStructure structure,
    int size_x, int size_y, int size_x_cr, int size_y_cr);

/*! Return a slot to the pool. The slot's buffers are NOT freed; they
 *  stay live for the next acquirer. */
extern void picture_buffer_pool_release(PictureBufferSlot *slot);

/*! Print one-line stats (hits / creates / misses) to stdout. */
extern void picture_buffer_pool_print_info(const PictureBufferPool *pool);

#endif /* _JM_PICTURE_POOL_H_ */
