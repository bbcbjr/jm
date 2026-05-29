
/*!
 ***************************************************************************
 * \file
 *    jm_picture_pool.c
 *
 * \brief
 *    Implementation of the per-decoder picture buffer pool. See
 *    jm_picture_pool.h for design rationale and usage rules.
 ***************************************************************************
 */

#include <stdio.h>

#include "global.h"
#include "memalloc.h"
#include "jm_picture_pool.h"

/* ===================================================================== */
/*                              CREATE / DESTROY                          */
/* ===================================================================== */

PictureBufferPool *picture_buffer_pool_create(int capacity)
{
  PictureBufferPool *pool;

  if (capacity <= 0)
    capacity = JM_PICTURE_POOL_DEFAULT_CAPACITY;

  pool = (PictureBufferPool *)calloc(1, sizeof(*pool));
  if (!pool)
    return NULL;

  pool->slots = (PictureBufferSlot **)calloc(capacity, sizeof(PictureBufferSlot *));
  if (!pool->slots)
  {
    free(pool);
    return NULL;
  }

  pool->capacity      = capacity;
  pool->count         = 0;
  pool->canonical_set = 0;
  jm_mutex_init(&pool->lock);
  return pool;
}

/* Free the buffers held by a slot. Mirrors the buffer-free portion of
 * free_storable_picture, in the same order, with the same free helpers. */
static void slot_free_buffers(PictureBufferSlot *slot)
{
  if (slot->mv_info)
  {
    free_mem2Dmp(slot->mv_info);
    slot->mv_info = NULL;
  }
  if (slot->mb_field)
  {
    free(slot->mb_field);
    slot->mb_field = NULL;
  }
  if (slot->imgY)
  {
    free_mem2Dpel_pad(slot->imgY, slot->iLumaPadY, slot->iLumaPadX);
    slot->imgY = NULL;
  }
  if (slot->imgUV)
  {
    free_mem3Dpel_pad(slot->imgUV, 2, slot->iChromaPadY, slot->iChromaPadX);
    slot->imgUV = NULL;
  }
}

void picture_buffer_pool_destroy(PictureBufferPool *pool)
{
  int i;
  if (!pool)
    return;

  for (i = 0; i < pool->count; i++)
  {
    if (pool->slots[i])
    {
      slot_free_buffers(pool->slots[i]);
      free(pool->slots[i]);
      pool->slots[i] = NULL;
    }
  }

  jm_mutex_destroy(&pool->lock);
  free(pool->slots);
  free(pool);
}

/* ===================================================================== */
/*                              ACQUIRE / RELEASE                         */
/* ===================================================================== */

/* All sub-checks for "does this acquire request match the pool's canonical
 * key?" -- runs while the lock is held. */
static int canonical_match(const PictureBufferPool *pool,
                           int size_x, int size_y, int size_x_cr, int size_y_cr,
                           int chroma_format_idc,
                           int pad_y, int pad_x,
                           int chroma_pad_y, int chroma_pad_x)
{
  return pool->canonical_size_x            == size_x
      && pool->canonical_size_y            == size_y
      && pool->canonical_size_x_cr         == size_x_cr
      && pool->canonical_size_y_cr         == size_y_cr
      && pool->canonical_chroma_format_idc == chroma_format_idc
      && pool->canonical_iLumaPadY         == pad_y
      && pool->canonical_iLumaPadX         == pad_x
      && pool->canonical_iChromaPadY       == chroma_pad_y
      && pool->canonical_iChromaPadX       == chroma_pad_x;
}

/* First-time slot creation: allocate the heavy buffers using the _nozero
 * variants (decoder fully overwrites them every frame). Mirrors the order
 * and helpers used by alloc_storable_picture's direct path. */
static PictureBufferSlot *slot_create_and_alloc_buffers(
    PictureBufferPool *pool,
    int size_x, int size_y, int size_x_cr, int size_y_cr,
    int chroma_format_idc,
    int pad_y, int pad_x, int chroma_pad_y, int chroma_pad_x)
{
  PictureBufferSlot *slot;
  int mb_count;

  slot = (PictureBufferSlot *)calloc(1, sizeof(*slot));
  if (!slot)
    return NULL;

  get_mem2Dpel_pad(&slot->imgY, size_y, size_x, pad_y, pad_x);
  if (chroma_format_idc != YUV400)
    get_mem3Dpel_pad(&slot->imgUV, 2, size_y_cr, size_x_cr, chroma_pad_y,
                     chroma_pad_x);

  get_mem2Dmp(&slot->mv_info, (size_y >> BLOCK_SHIFT), (size_x >> BLOCK_SHIFT));

  /* mb_field is a small per-MB byte array. Kept calloc-style (zero-init)
   * for safety because some neighbor lookups during slice header parsing
   * might read it before MB-level decode writes it. ~8 KB for 1080p so
   * the zero-fill cost is negligible. Exact implementation of alloc_pic_motion */
  mb_count = (size_y >> BLOCK_SHIFT) * (size_x >> BLOCK_SHIFT);
  slot->mb_field = (byte *)calloc(mb_count, sizeof(byte));

  slot->size_x               = size_x;
  slot->size_y               = size_y;
  slot->size_x_cr            = size_x_cr;
  slot->size_y_cr            = size_y_cr;
  slot->iLumaStride          = size_x    + 2 * pad_x;
  slot->iLumaExpandedHeight  = size_y    + 2 * pad_y;
  slot->iChromaStride        = size_x_cr + 2 * chroma_pad_x;
  slot->iChromaExpandedHeight= size_y_cr + 2 * chroma_pad_y;
  slot->iLumaPadY            = pad_y;
  slot->iLumaPadX            = pad_x;
  slot->iChromaPadY          = chroma_pad_y;
  slot->iChromaPadX          = chroma_pad_x;
  slot->chroma_format_idc    = chroma_format_idc;
  slot->parent_pool          = pool;
  return slot;
}

PictureBufferSlot *picture_buffer_pool_try_acquire(
    PictureBufferPool *pool, VideoParameters *p_Vid,
    PictureStructure structure,
    int size_x, int size_y, int size_x_cr, int size_y_cr)
{
  int chroma_format_idc;
  int pad_y, pad_x, chroma_pad_y, chroma_pad_x;
  int i, free_idx;
  PictureBufferSlot *slot = NULL;

  /* Eligibility gates -- these are cheap and don't need the lock. */
  if (!pool)
    return NULL;
  if (structure != FRAME)
    return NULL;                                  /* pool serves frames only  */
  if (p_Vid->separate_colour_plane_flag != 0)
    return NULL;                                  /* JV mode falls back       */

  chroma_format_idc = p_Vid->active_sps->chroma_format_idc;
  pad_y             = p_Vid->iLumaPadY;
  pad_x             = p_Vid->iLumaPadX;
  chroma_pad_y      = p_Vid->iChromaPadY;
  chroma_pad_x      = p_Vid->iChromaPadX;

  jm_mutex_lock(&pool->lock);

  /* Canonical key: lazy-set by the first acquire, then enforced. */
  if (!pool->canonical_set)
  {
    pool->canonical_set                 = 1;
    pool->canonical_size_x              = size_x;
    pool->canonical_size_y              = size_y;
    pool->canonical_size_x_cr           = size_x_cr;
    pool->canonical_size_y_cr           = size_y_cr;
    pool->canonical_chroma_format_idc   = chroma_format_idc;
    pool->canonical_iLumaPadY           = pad_y;
    pool->canonical_iLumaPadX           = pad_x;
    pool->canonical_iChromaPadY         = chroma_pad_y;
    pool->canonical_iChromaPadX         = chroma_pad_x;
  }
  else if (!canonical_match(pool, size_x, size_y, size_x_cr, size_y_cr,
                            chroma_format_idc, pad_y, pad_x,
                            chroma_pad_y, chroma_pad_x))
  {
    pool->stat_misses++;
    jm_mutex_unlock(&pool->lock);
    return NULL;
  }

  /* Find a free slot. */
  free_idx = -1;
  for (i = 0; i < pool->count; i++)
  {
    if (!pool->slots[i]->in_use)
    {
      free_idx = i;
      break;
    }
  }

  if (free_idx >= 0)
  {
    /* Recycle. */
    slot = pool->slots[free_idx];
    slot->in_use = 1;
    pool->stat_hits++;
  }
  else if (pool->count < pool->capacity)
  {
    /* First-time alloc for this slot. */
    slot = slot_create_and_alloc_buffers(pool,
                                         size_x, size_y, size_x_cr, size_y_cr,
                                         chroma_format_idc,
                                         pad_y, pad_x,
                                         chroma_pad_y, chroma_pad_x);
    if (slot)
    {
      slot->in_use = 1;
      pool->slots[pool->count++] = slot;
      pool->stat_creates++;
    }
    else
    {
      pool->stat_misses++;
    }
  }
  else
  {
    /* Pool full -- caller falls back to direct alloc. */
    pool->stat_misses++;
  }

  jm_mutex_unlock(&pool->lock);
  return slot;
}

void picture_buffer_pool_release(PictureBufferSlot *slot)
{
  PictureBufferPool *pool;
  if (!slot)
    return;
  pool = slot->parent_pool;
  if (!pool)
    return;                                      /* defensive: detached slot */

  jm_mutex_lock(&pool->lock);
  slot->in_use = 0;
  /* NOTE: We do NOT zero imgY/imgUV/mv_info between uses -- the decoder
   * fully overwrites them every frame (proven by Phase A bit-identical
   * verification). mb_field is left intact too; if a future build needs
   * defensive zero, add memset here gated by a #define. */
  jm_mutex_unlock(&pool->lock);
}

/* ===================================================================== */
/*                                  STATS                                  */
/* ===================================================================== */

void picture_buffer_pool_print_info(const PictureBufferPool *pool)
{
  if (!pool)
  {
    printf("PicturePool: (none)\n");
    return;
  }
  printf("PicturePool: capacity=%d created=%d hits=%u creates=%u misses=%u (hit rate %.1f%%)\n",
         pool->capacity, pool->count,
         pool->stat_hits, pool->stat_creates, pool->stat_misses,
         (pool->stat_hits + pool->stat_creates + pool->stat_misses > 0)
           ? (100.0 * (double)pool->stat_hits /
              (double)(pool->stat_hits + pool->stat_creates + pool->stat_misses))
           : 0.0);
}
