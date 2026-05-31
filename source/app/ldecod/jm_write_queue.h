
/*!
 ***************************************************************************
 * \file
 *    jm_write_queue.h
 *
 * \brief
 *    Thread-safe FIFO of write requests
 *    + writer thread that drains them by calling write_out_picture
 *    synchronously on the worker. Decoder thread enqueues the picture
 *    pointer (after taking an extra refcount via
 *    storable_picture_addref); writer thread runs the cropping
 *    (img2buf) + disk I/O + final free_storable_picture.
 *
 *    Ownership / lifecycle
 *    ---------------------
 *    - storable_picture_addref(p) is called BEFORE jm_write_queue_push.
 *      The picture's _ref_count is now (DPB ref + writer ref) = at
 *      least 2.
 *    - jm_write_queue_push transfers the *additional* reference into
 *      the queue. Push must not fail in steady state -- the queue is
 *      bounded but backpressure is handled by the picture pool
 *      blocking acquire, not by queue fullness.
 *    - Writer pops, calls write_out_picture (which does crop +
 *      img2buf + io_write), then free_storable_picture (which drops
 *      the writer's reference -- struct + pool slot are released only
 *      when the count reaches zero, i.e. when the DPB has also
 *      released its reference).
 *    - On push failure (queue closed at shutdown), caller drops the
 *      addref it just took and falls back to synchronous
 *      write_out_picture.
 *
 *    Threading
 *    ---------
 *    - Single producer (decode thread) / single consumer (writer
 *      thread) today.
 *    - Mutex + 2 condvars (not_full, not_empty). push blocks while
 *      full unless closed; pop blocks while empty unless closed and
 *      drained. close wakes both.
 ***************************************************************************
 */

#ifndef _JM_WRITE_QUEUE_H_
#define _JM_WRITE_QUEUE_H_

#include <stddef.h>

struct storable_picture;
struct video_par;

/*! Per-request payload. Just three pointers/ints -- no pixel data,
 *  no metadata snapshot. The writer dereferences req->picture
 *  directly because the storable_picture_addref taken before push
 *  keeps it alive, and runs the full write_out_picture (including
 *  pDecOuputPic slot pickup). pDecOuputPic list safety is provided
 *  by jm_writer_drain at every DecodeOneFrame boundary -- decoder
 *  and writer never mutate the list concurrently. */
typedef struct write_request
{
  struct storable_picture *picture;  /* refcount held by request */
  int                      p_out;    /* output fd; -1 = skip writes */
  struct video_par        *p_Vid;    /* decoder params (img2buf, etc.) */
} write_request_t;

struct jm_write_queue;
typedef struct jm_write_queue jm_write_queue_t;

extern jm_write_queue_t *jm_write_queue_create (int capacity);

/*! Destroy the queue. Releases the refcount on any pictures still in
 *  the queue (their pool slots can then be reclaimed normally).
 *  MUST NOT be called while the writer thread is running. */
extern void              jm_write_queue_destroy(jm_write_queue_t *q);

/*! Push a request. Blocks if full. Returns 0 on success, -1 if the
 *  queue is closed (caller retains the addref it took -- must
 *  free_storable_picture or fall back to sync). */
extern int               jm_write_queue_push   (jm_write_queue_t *q,
                                                const write_request_t *req);

/*! Pop a request. Blocks if empty. Returns 0 on success and fills
 *  *out; returns -1 if closed AND empty (clean EOS). */
extern int               jm_write_queue_pop    (jm_write_queue_t *q,
                                                write_request_t *out);

/*! Mark closed. Subsequent pushes return -1; subsequent pops drain
 *  then return -1. */
extern void              jm_write_queue_close  (jm_write_queue_t *q);

/*! Block until queue is empty AND the writer is not mid-request.
 *  Call this from the decoder thread before observing the state of
 *  any pre-claimed DecodedPicList slot. */
extern void              jm_write_queue_drain  (jm_write_queue_t *q);

/*! Spawn / stop the writer worker thread. */
extern void              jm_writer_start(struct video_par *p_Vid);
extern void              jm_writer_stop (struct video_par *p_Vid);

/*! Convenience: drain the per-VideoParameters writer queue if one
 *  exists. No-op if writes are sync. */
extern void              jm_writer_drain(struct video_par *p_Vid);

#endif  /* _JM_WRITE_QUEUE_H_ */
