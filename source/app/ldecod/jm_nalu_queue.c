
/*!
 ***************************************************************************
 * \file
 *    jm_nalu_queue.c
 *
 * \brief
 *    Thread-safe bounded FIFO of NALU_t pointers. See jm_nalu_queue.h
 *    for ownership and concurrency rules.
 ***************************************************************************
 */

#include <stdlib.h>
#include <string.h>

#include "global.h"          /* for NALU_t and VideoParameters */
#include "nalu.h"            /* for AllocNALU / FreeNALU / read_next_nalu */
#include "memalloc.h"        /* for no_mem_exit */
#include "jm_threads.h"
#include "jm_nalu_queue.h"

struct jm_nalu_queue
{
  NALU_t   **buf;        /* ring buffer of NALU_t* slots */
  int        cap;        /* capacity */
  int        head;       /* next index to pop from */
  int        tail;       /* next index to push to */
  int        count;      /* number of items currently in the queue */
  int        closed;     /* 1 once jm_nalu_queue_close() has been called */
  jm_mutex_t m;
  jm_cond_t  not_full;
  jm_cond_t  not_empty;
};

jm_nalu_queue_t * jm_nalu_queue_create(int capacity)
{
  jm_nalu_queue_t *q;

  if (capacity <= 0) return NULL;

  q = (jm_nalu_queue_t *)calloc(1, sizeof(*q));
  if (q == NULL) return NULL;

  q->buf = (NALU_t **)calloc((size_t)capacity, sizeof(NALU_t *));
  if (q->buf == NULL) { free(q); return NULL; }

  q->cap     = capacity;
  q->head    = 0;
  q->tail    = 0;
  q->count   = 0;
  q->closed  = 0;
  jm_mutex_init(&q->m);
  jm_cond_init (&q->not_full);
  jm_cond_init (&q->not_empty);
  return q;
}

void jm_nalu_queue_destroy(jm_nalu_queue_t *q)
{
  if (q == NULL) return;

  /* Free any NALUs still buffered. No locking needed: by contract the
   * caller has joined any threads using this queue before destroy. */
  while (q->count > 0)
  {
    FreeNALU(q->buf[q->head]);
    q->buf[q->head] = NULL;
    q->head = (q->head + 1) % q->cap;
    q->count--;
  }

  jm_cond_destroy (&q->not_empty);
  jm_cond_destroy (&q->not_full);
  jm_mutex_destroy(&q->m);
  free(q->buf);
  free(q);
}

int jm_nalu_queue_push(jm_nalu_queue_t *q, NALU_t *nalu)
{
  if (q == NULL || nalu == NULL) return -1;

  jm_mutex_lock(&q->m);
  while (q->count == q->cap && !q->closed)
  {
    jm_cond_wait(&q->not_full, &q->m);
  }
  if (q->closed)
  {
    jm_mutex_unlock(&q->m);
    return -1;
  }
  q->buf[q->tail] = nalu;
  q->tail = (q->tail + 1) % q->cap;
  q->count++;
  jm_cond_signal(&q->not_empty);
  jm_mutex_unlock(&q->m);
  return 0;
}

NALU_t * jm_nalu_queue_pop(jm_nalu_queue_t *q)
{
  NALU_t *nalu;

  if (q == NULL) return NULL;

  jm_mutex_lock(&q->m);
  while (q->count == 0 && !q->closed)
  {
    jm_cond_wait(&q->not_empty, &q->m);
  }
  if (q->count == 0)
  {
    /* closed and drained */
    jm_mutex_unlock(&q->m);
    return NULL;
  }
  nalu = q->buf[q->head];
  q->buf[q->head] = NULL;
  q->head = (q->head + 1) % q->cap;
  q->count--;
  jm_cond_signal(&q->not_full);
  jm_mutex_unlock(&q->m);
  return nalu;
}

void jm_nalu_queue_close(jm_nalu_queue_t *q)
{
  if (q == NULL) return;

  jm_mutex_lock(&q->m);
  q->closed = 1;
  /* wake both waiters: pushers see "closed", pop sees "drain then exit" */
  jm_cond_broadcast(&q->not_empty);
  jm_cond_broadcast(&q->not_full);
  jm_mutex_unlock(&q->m);
}

int jm_nalu_queue_is_empty(jm_nalu_queue_t *q)
{
  int r;
  if (q == NULL) return 1;
  jm_mutex_lock(&q->m);
  r = (q->count == 0);
  jm_mutex_unlock(&q->m);
  return r;
}

int jm_nalu_queue_is_closed(jm_nalu_queue_t *q)
{
  int r;
  if (q == NULL) return 1;
  jm_mutex_lock(&q->m);
  r = q->closed;
  jm_mutex_unlock(&q->m);
  return r;
}

/* ===================================================================== */
/*                  DEMUX THREAD (M4-P2)                                  */
/* ===================================================================== */

/*!
 ************************************************************************
 * \brief Pump exactly one NALU from the bitstream into the queue.
 *        Returns 1 on success, 0 on EOS (the queue is closed in that
 *        case) or if the queue was closed by another thread mid-pump.
 *
 *        Called in a loop by demux_thread_main().
 ************************************************************************
 */
static int pump_one_nalu(VideoParameters *p_Vid)
{
  NALU_t *n = AllocNALU(MAX_CODED_FRAME_SIZE);
  if (n == NULL)
    no_mem_exit("pump_one_nalu");
  if (read_next_nalu(p_Vid, n) == 0)
  {
    FreeNALU(n);
    jm_nalu_queue_close(p_Vid->nalu_queue);
    return 0;   /* EOS */
  }
  if (jm_nalu_queue_push(p_Vid->nalu_queue, n) != 0)
  {
    /* Queue was closed under us by jm_demux_stop(). Exit cleanly. */
    FreeNALU(n);
    return 0;
  }
  return 1;
}

/*!
 ************************************************************************
 * \brief Demux thread main loop. Pumps until EOS or queue closed.
 ************************************************************************
 */
static void *demux_thread_main(void *arg)
{
  VideoParameters *p_Vid = (VideoParameters *)arg;
  while (pump_one_nalu(p_Vid))
  {
    /* keep pumping */
  }
  return NULL;
}

/*!
 ************************************************************************
 * \brief Spawn the demux thread. The handle is stored on p_Vid for
 *        later jm_demux_stop().
 ************************************************************************
 */
void jm_demux_start(VideoParameters *p_Vid)
{
  if (p_Vid == NULL || p_Vid->nalu_queue == NULL)
    return;
  p_Vid->demux_thread_running = 1;
  if (jm_thread_create(&p_Vid->demux_thread, demux_thread_main, p_Vid) != 0)
  {
    p_Vid->demux_thread_running = 0;
    no_mem_exit("jm_demux_start: failed to create thread");
  }
}

/*!
 ************************************************************************
 * \brief Stop the demux thread. Closes the queue (unblocks any push
 *        waiting), then joins. Safe to call repeatedly.
 ************************************************************************
 */
void jm_demux_stop(VideoParameters *p_Vid)
{
  if (p_Vid == NULL || !p_Vid->demux_thread_running)
    return;
  /* Close the queue first: if demux is blocked in push (full + nobody
   * popping), this wakes it and the next push returns -1, ending the
   * pump_one_nalu loop. If demux already exited on EOS, this is a
   * no-op (queue is already closed). */
  if (p_Vid->nalu_queue)
    jm_nalu_queue_close(p_Vid->nalu_queue);
  jm_thread_join(p_Vid->demux_thread);
  p_Vid->demux_thread_running = 0;
}

