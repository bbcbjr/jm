
/*!
 ***************************************************************************
 * \file
 *    jm_nalu_queue.h
 *
 * \brief
 *    Thread-safe bounded FIFO of NALU_t pointers. Producer/consumer
 *    primitive for Stage 2 view-parallel decoding (M4-P1+).
 *
 *    Ownership rules:
 *      - jm_nalu_queue_push() transfers ownership of the NALU into the
 *        queue. Caller must not free the NALU after a successful push.
 *      - jm_nalu_queue_pop() transfers ownership back to the caller. The
 *        caller MUST FreeNALU() it when done.
 *      - jm_nalu_queue_destroy() frees any NALUs still in the queue.
 *
 *    Concurrency:
 *      - push blocks while full unless the queue is closed (then it
 *        returns -1 and the NALU is NOT consumed -- caller must free).
 *      - pop blocks while empty unless the queue is closed and drained
 *        (then it returns NULL).
 *      - close wakes both push and pop waiters. Subsequent pops drain
 *        any remaining NALUs and then return NULL on each call.
 *
 *    M4-P1 usage: single-threaded. The decode thread pumps one NALU at a
 *    time and immediately pops, so push/pop never actually block.
 *    M4-P2+ usage: a separate demux thread runs jm_nalu_queue_push() in
 *    a loop, and the decode thread blocks in pop until each NALU arrives.
 ***************************************************************************
 */

#ifndef _JM_NALU_QUEUE_H_
#define _JM_NALU_QUEUE_H_

/* Forward declaration of NALU_t to avoid pulling in nalu.h from every
 * file that touches the queue field on VideoParameters. */
struct nalu_t;
typedef struct nalu_t NALU_t;

struct jm_nalu_queue;
typedef struct jm_nalu_queue jm_nalu_queue_t;

/*!
 ************************************************************************
 * \brief Allocate a new queue with the given capacity (number of NALU
 *        slots). Returns NULL on allocation failure.
 ************************************************************************
 */
extern jm_nalu_queue_t * jm_nalu_queue_create(int capacity);

/*!
 ************************************************************************
 * \brief Tear down the queue. Frees any NALUs still buffered.
 ************************************************************************
 */
extern void              jm_nalu_queue_destroy(jm_nalu_queue_t *q);

/*!
 ************************************************************************
 * \brief Push a NALU into the queue. Blocks if full. Returns 0 on
 *        success, -1 if the queue is already closed (NALU NOT consumed
 *        in that case).
 ************************************************************************
 */
extern int               jm_nalu_queue_push(jm_nalu_queue_t *q, NALU_t *nalu);

/*!
 ************************************************************************
 * \brief Pop a NALU. Blocks if empty. Returns NULL only if the queue is
 *        closed AND empty (end-of-stream consumed).
 ************************************************************************
 */
extern NALU_t *          jm_nalu_queue_pop(jm_nalu_queue_t *q);

/*!
 ************************************************************************
 * \brief Mark the queue as closed. Subsequent pushes return -1.
 *        Subsequent pops drain remaining items, then return NULL.
 ************************************************************************
 */
extern void              jm_nalu_queue_close(jm_nalu_queue_t *q);

/*!
 ************************************************************************
 * \brief Snapshot inspectors for the single-threaded M4-P1 pump pattern.
 *        Once an actual demux thread exists (M4-P2+), the decode side
 *        will use the blocking pop instead and these go unused.
 ************************************************************************
 */
extern int               jm_nalu_queue_is_empty(jm_nalu_queue_t *q);
extern int               jm_nalu_queue_is_closed(jm_nalu_queue_t *q);

/*!
 ************************************************************************
 *                        DEMUX THREAD (M4-P2+)
 *
 *  jm_demux_start(p_Vid) spawns a thread that reads NALUs from the
 *  bitstream (annex_b or RTP) and pushes them onto p_Vid->nalu_queue.
 *  Exits naturally on EOS (closes the queue and returns).
 *
 *  jm_demux_stop(p_Vid) closes the queue (in case the thread is blocked
 *  in push) and joins the demux thread. Safe to call even if start was
 *  never called (no-op) or the thread already exited on its own.
 *
 *  Both helpers take VideoParameters* because the demux thread reads
 *  via read_next_nalu(p_Vid, ...) which needs the bitstream state on
 *  p_Vid (annex_b / BitStreamFile / NALUCount / LastAccessUnitExists).
 *
 *  Threading contract: ONLY the demux thread touches p_Vid->annex_b,
 *  p_Vid->BitStreamFile, p_Vid->NALUCount, p_Vid->LastAccessUnitExists
 *  while the demux is running. The decode thread(s) only touch the
 *  queue's pop side.
 ************************************************************************
 */
struct video_par;
extern void              jm_demux_start(struct video_par *p_Vid);
extern void              jm_demux_stop (struct video_par *p_Vid);

#endif  /* _JM_NALU_QUEUE_H_ */
